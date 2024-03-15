# -*- coding: utf-8 -*-
import logging
import os
import sys
from dataclasses import asdict
from logging import Logger
from typing import Any, Dict, List, Tuple, Union

import datasets
import transformers
import wandb
from datasets import DatasetDict, load_dataset
from transformers import TrainingArguments

from util.const import GPT2_EN_PATH, GPT2_ZH_PATH, PROJ_NAME
from util.dp_arguments import DataTrainingArguments, Gpt2Arguments, OlmoArguments


def post_process_gpt2_args(gpt2_args: Gpt2Arguments) -> None:
	"""
	Post-process the parsed model arguments based on specific conditions.
	"""
	# Set the GPT2 model path to GPT2-ZH if the model is Chinese.
	if gpt2_args.chinese:
		gpt2_args.gpt2_name_or_path = GPT2_ZH_PATH
		gpt2_args.config_name = GPT2_ZH_PATH
		gpt2_args.tokenizer_name = GPT2_ZH_PATH

	if gpt2_args.randomized:
		gpt2_args.gpt2_name_or_path = None
		gpt2_args.config_name = GPT2_EN_PATH
		gpt2_args.tokenizer_name = GPT2_EN_PATH


def post_process_training_args(training_args: TrainingArguments, wandb_proj_name: str = '', serial: str = '') -> None:
	"""
	Post-process the parsed HF training arguments
	"""
	training_args.report_to = ["wandb"]
	training_args.logging_steps = 50
	training_args.load_best_model_at_end = True
	training_args.metric_for_best_model = "eval_accuracy"
	training_args.greater_is_better = True
	training_args.save_total_limit = 1
	training_args.output_dir = os.path.join(training_args.output_dir, wandb_proj_name, serial)  # Modify output dir


def transform_dict(config_dict: Dict, expand: bool = True):
	"""
	General function to transform any dictionary into wandb config acceptable format
	(This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
	The expand argument is used to expand iterables into dictionaries so that these configs can be used when compare across runs
	"""
	ret: Dict[str, Any] = {}
	for k, v in config_dict.items():
		if v is None or isinstance(v, (int, float, str)):
			ret[k] = v
		elif isinstance(v, (list, tuple, set)):
			# Need to check if item in iterable is YAML-friendly
			t = transform_dict(dict(enumerate(v)), expand)
			# Transform back to iterable if expand is False
			ret[k] = t if expand else [t[i] for i in range(len(v))]
		elif isinstance(v, dict):
			ret[k] = transform_dict(v, expand)
		else:
			# Transform to YAML-friendly (str) format
			# Need to handle both Classes, Callables, Object Instances
			# Custom Classes might not have great __repr__ so __name__ might be better in these cases
			vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
			ret[k] = f"{v.__module__}:{vname}"
	return ret


def set_wandb(model_args: Union[Gpt2Arguments, OlmoArguments], data_args: DataTrainingArguments,
              training_args: TrainingArguments) -> Tuple[str, str]:
	"""
	Set the wandb project name and wandb env based on the model arguments.

	:return [wandb project name, serial name for each experiment]
	"""
	if isinstance(model_args, Gpt2Arguments):
		model_name = 'GPT2'
	elif isinstance(model_args, OlmoArguments):
		model_name = 'OLMO'
	else:
		raise NotImplementedError(f"Model type {type(model_args)} not supported so far.")

	# Serial name for each experiment
	serial = f"Epoch{int(training_args.num_train_epochs)}-LR{training_args.learning_rate}-"
	if model_args.randomized:
		serial += "Randomized-"
	else:
		serial += "Pretrained-"

	if model_args.dev:
		serial += "Dev"
	else:
		serial += "Test"

	# wandb project name
	if model_args.use_mlp:
		wandb_proj_name = f"{PROJ_NAME}-{data_args.task}-{model_name}-MLP-Dim{model_args.mlp_dim}-Layer{model_args.mlp_layers}"
	else:
		wandb_proj_name = f"{PROJ_NAME}-{data_args.task}-{model_name}-LR-Dim{model_args.mlp_dim}-Layer{model_args.mlp_layers}"

	if model_args.onehot:
		wandb_proj_name += "-OneHot"
		training_args.output_dir += "OneHot/"

	if isinstance(model_args, Gpt2Arguments) and model_args.chinese:
		wandb_proj_name += "-Chinese"
		training_args.output_dir += "Chinese/"

	os.environ["WANDB_PROJECT"] = wandb_proj_name
	wandb.init(
		project=wandb_proj_name,
		name=serial,
		)
	wandb.log(transform_dict(asdict(model_args)))
	wandb.log(transform_dict(asdict(data_args)))

	return wandb_proj_name, serial


def set_logger(training_args: TrainingArguments) -> Logger:
	logger: Logger = logging.getLogger(__name__)
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
		)
	log_level = training_args.get_process_log_level()
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	# Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}\n device: {training_args.device}\n n_gpu: {training_args.n_gpu} \n"
		f"distributed training: {bool(training_args.local_rank != -1)}\n 16-bits training: {training_args.fp16}"
		)
	logger.info(f"Training/evaluation parameters {training_args}")

	return logger


def get_total_gpus() -> int:
	"""
	Get total number of GPUs in the server
	:return: number of GPUs
	"""
	import subprocess

	sp = subprocess.Popen(['nvidia-smi', '--list-gpus'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out_str = sp.communicate()
	out_list = out_str[0].decode("utf-8").split('\n')
	# Subtract one as the last line is empty
	num_gpus = len(out_list) - 1
	print(f"... {num_gpus} GPUs found")
	return num_gpus


def get_idle_gpus(num_gpus: int = 2) -> List[int]:
	"""
	Get idle GPUs in the server
	:param num_gpus: requested number of GPUs
	:return: list of idle GPU IDs
	"""
	import operator
	import subprocess

	total_gpus = get_total_gpus()
	if num_gpus > total_gpus:
		raise ValueError(f'Requested number of GPUs ({num_gpus}) exceeds available GPUs ({total_gpus})')

	sp = subprocess.Popen(
		['nvidia-smi', '--format=csv', '--query-gpu=utilization.gpu'], stdout=subprocess.PIPE, stderr=subprocess.PIPE
		)
	out_str = sp.communicate()
	out_list = out_str[0].decode("utf-8").split('\n')
	gpu_utilization = []
	for i, gpu in enumerate(out_list[1:-1]):
		utilization = int(gpu.split(' ')[0])
		gpu_utilization.append((i, utilization))
	sorted_gpus = sorted(gpu_utilization, key=operator.itemgetter(1))
	idle_gpus = [gpu[0] for gpu in sorted_gpus[:num_gpus]]
	return idle_gpus


def set_gpu_env(num_gpus: int = 1):
	"""
	Set GPU environments in the server
	:param num_gpus: number of GPUs to use
	:return: PyTorch device
	"""
	import os
	import torch

	idle_gpus = get_idle_gpus(num_gpus)
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, idle_gpus))
	print(f"... Available GPUs {idle_gpus}")
	# list available GPUs
	gpu_list = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
	print(f"... {len(gpu_list)} visible 'logical' GPUs: {gpu_list}")
	# Set up GPUs for multi-GPU training
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"... using {device}")

	return device


def load_raw_dataset(model_args: Union[Gpt2Arguments, OlmoArguments], data_args: DataTrainingArguments,
                     training_args: TrainingArguments, logger: Logger) -> DatasetDict:
	data_files = {}
	logger.info("Loading data for {}".format(data_args.task))
	if training_args.do_train:
		data_files["train"] = os.path.join(data_args.data_dir, data_args.task, 'train.json')

	if model_args.dev:
		data_files["validation"] = os.path.join(data_args.data_dir, data_args.task, 'development.json')
	else:
		data_files["validation"] = os.path.join(data_args.data_dir, data_args.task, 'test.json')
	data_files["test"] = os.path.join(data_args.data_dir, data_args.task, 'test.json')

	raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
	return raw_datasets


def record_num_of_params(model, logger: Logger) -> None:
	"""
	Record the number of trainable parameters and total parameters of the model and log it on wandb.

	:param model: An LLM
	:param logger: A logger
	"""
	num_trainable_params = model.num_parameters(only_trainable=True)
	num_total_params = model.num_parameters()
	logger.info(f"Number of parameters to train (without adapters): {num_trainable_params}")
	logger.info(f"Total number of parameters (without adapters): {num_total_params}")
	wandb.run.summary["num_trainable_params"] = num_trainable_params
	wandb.run.summary["num_total_params"] = num_total_params


def convert_span(result, pre_tokenized_str, span):
	char_start = pre_tokenized_str[span[0]][1][0]
	char_end = pre_tokenized_str[span[1]][1][1] - 1
	start = result.char_to_token(char_start)
	end = result.char_to_token(char_end)
	return [start, end]


def load_optimizer(model, training_args: TrainingArguments):
	"""
	Load the optimizer for the model based on the training arguments.

	:param model:
	:param training_args:
	:return: Pptimizer
	"""
	if training_args.do_train:
		no_decay = ["bias", "LayerNorm.weight"]
		optimizer_grouped_parameters = [
			{
				"params"      : [p for n, p in model.named_parameters() if
				                 not any(nd in n for nd in no_decay) and p.requires_grad],
				"weight_decay": training_args.weight_decay,
				"lr"          : training_args.learning_rate
				},
			{
				"params"      : [p for n, p in model.named_parameters() if
				                 any(nd in n for nd in no_decay) and p.requires_grad],
				"weight_decay": 0.0,
				"lr"          : training_args.learning_rate
				},
			]

		from torch.optim import AdamW
		optimizer = AdamW(optimizer_grouped_parameters)
	else:
		optimizer = None

	return optimizer


def compute_metrics(eval_pred):
	accuracy, _ = eval_pred
	accuracy = accuracy.sum(axis=0)
	accuracy = accuracy[0] / accuracy[1]
	return {"accuracy": accuracy}


def get_label_and_id_mapping(task: str) -> Tuple[Dict[str, int], Dict[int, str]]:
	"""
	Get label to id and id to label mapping from the pre-defined data directory.

	:param task: Task name
	:return: label to id mapping, id to label mapping
	"""
	from util.const import LABEL_DICT
	label2id = {label: i for i, label in enumerate(LABEL_DICT[task])}
	id2label = {i: label for label, i in label2id.items()}
	return label2id, id2label
