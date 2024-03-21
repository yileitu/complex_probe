# -*- coding: utf-8 -*-
# import hf_olmo # Must be imported to load olmo model, ignore no reference error.
import os
import random

import pandas as pd
import torch
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoConfig, AutoTokenizer, EarlyStoppingCallback, GPTNeoXForCausalLM, HfArgumentParser, \
	Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments, default_data_collator, set_seed

from modeling_pythia_dp import PythiaForDiagnosticProbing
from util.const import IS_UNARY, MAX_TARGET
from util.dp_arguments import DataTrainingArguments, PythiaArguments
from util.func import compute_metrics, convert_span, get_label_and_id_mapping, load_optimizer, load_raw_dataset, \
	post_process_training_args, record_num_of_params, set_gpu_env, set_logger, set_wandb
from util.probe_config import ProbeConfig

parser = HfArgumentParser((PythiaArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model_args: PythiaArguments
data_args: DataTrainingArguments
training_args: TrainingArguments

# Setups
wandb_proj_name, serial = set_wandb(model_args, data_args, training_args)
set_seed(training_args.seed)
logger = set_logger(training_args)
device = set_gpu_env(num_gpus=data_args.n_gpu)

# Post-process args
model_args.model_path = f"EleutherAI/pythia-{model_args.scale}-deduped"
model_args.cache_dir = os.path.join(model_args.cache_dir, f"pythia-{model_args.scale}-deduped", model_args.revision)
post_process_training_args(training_args=training_args, wandb_proj_name=wandb_proj_name, serial=serial)

# Load Pythia config
pythia_config_kwargs = {
	"cache_dir"     : model_args.cache_dir,
	"revision"      : model_args.revision,
	"use_auth_token": True,
	"use_cache"     : True,
	}
pythia_config = AutoConfig.from_pretrained(model_args.model_path, **pythia_config_kwargs)
label2id, id2label = get_label_and_id_mapping(task=data_args.task)
pythia_config.label2id = label2id
pythia_config.id2label = id2label
print("Pythia config: ", pythia_config)

# Load tokenizer
tokenizer_kwargs = {
	"cache_dir": model_args.cache_dir,
	"revision" : model_args.revision,
	}
tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, **tokenizer_kwargs)
pre_tokenizer = WhitespaceSplit()
tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.pad_token = tokenizer.eos_token
print("Pad token: ", tokenizer.pad_token)
print("Pad token ID: ", tokenizer.pad_token_id)
print("Vocab size of Config before tokenization: ", pythia_config.vocab_size)
print("Vocab size of Tokenizer before tokenization: ", len(tokenizer))

# Load Pythia model
pythia = GPTNeoXForCausalLM.from_pretrained(pretrained_model_name_or_path=model_args.model_path, **pythia_config_kwargs)
print(type(pythia))
probe_config = ProbeConfig()
probe_config.mlp_dropout = model_args.mlp_dropout
probe_config.mlp_dim = model_args.mlp_dim
probe_config.mlp_layers = model_args.mlp_layers
probe_config.unary = IS_UNARY[data_args.task]
probe_config.use_mlp = model_args.use_mlp
probe_config.num_labels = len(label2id)
probe_config.onehot = model_args.onehot
probe_config.label2id = label2id
probe_config.id2label = id2label
if probe_config.onehot:
	logger.info("Using onehot embeddings.")
model = PythiaForDiagnosticProbing(pythia=pythia, pythia_config=pythia_config, probe_config=probe_config)
record_num_of_params(model, logger)
print("Embedding size of Olmo: ", pythia.get_input_embeddings().weight.shape)
print(f"Model embedding dimension: {pythia.get_input_embeddings().embedding_dim}")
print("Output size of Olmo: ", pythia.get_output_embeddings().weight.shape)
print(probe_config)

# # Dataset
# Load raw dataset
raw_datasets = load_raw_dataset(model_args, data_args, training_args, logger)

# Preprocessing the datasets.
# First we tokenize all the texts.
if training_args.do_train:
	column_names = raw_datasets["train"].column_names
else:
	column_names = raw_datasets["validation"].column_names


# NOTE: To determine max_length, max_length_train, max_length_val, max_length_test
def pre_tokenize_function(example):
	"""
	Determine MAX_LENGTH for model of different languages
	"""
	result = tokenizer(example['text'])
	return result


pre_tokenized_datasets = raw_datasets.map(
	pre_tokenize_function, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names,
	load_from_cache_file=False, desc="Running tokenizer on dataset"
	)
max_length_train = max(len(x['input_ids']) for x in pre_tokenized_datasets["train"])
max_length_val = max(len(x['input_ids']) for x in pre_tokenized_datasets["validation"])
max_length_test = max(len(x['input_ids']) for x in pre_tokenized_datasets["test"])
max_length = max(max_length_train, max_length_val, max_length_test)
print("Max length of input in Train: ", max_length_train)
print("Max length of input in Validation: ", max_length_val)
print("Max length of input in Test: ", max_length_test)
print("Max length of input: ", max_length)
del pre_tokenized_datasets


def tokenize_function(example):
	result = tokenizer(example['text'], padding="max_length", max_length=max_length)
	pre_tokenized_str = pre_tokenizer.pre_tokenize_str(example['text'])

	num_targets = len(example['targets'])
	num_to_pad = MAX_TARGET[data_args.task] - num_targets
	# NOTE: 与jiaoda代码有改动
	# num_to_pad = max_length - num_targets
	pad_spans = [[-1, -1]] * num_to_pad
	pad_labels = [-1] * num_to_pad

	# NOTE: span1s, span2s are custom parameter names of forward().
	result['span1s'] = [convert_span(result, pre_tokenized_str, target['span1']) for target in example['targets']]
	result['span1s'].extend(pad_spans)
	result['labels'] = [label2id[target['label']] for target in example['targets']]
	result['labels'].extend(pad_labels)
	if not probe_config.unary:
		result['span2s'] = [convert_span(result, pre_tokenized_str, target['span2']) for target in example['targets']]
		result['span2s'].extend(pad_spans)
	return result


with training_args.main_process_first(desc="dataset map tokenization"):

	tokenized_datasets = raw_datasets.map(
		tokenize_function, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names,
		load_from_cache_file=False, desc="Running tokenizer on dataset"
		)

if training_args.do_train:
	if "train" not in tokenized_datasets:
		raise ValueError("--do_train requires a train dataset")
	train_dataset = tokenized_datasets["train"]
	if data_args.max_train_samples is not None:
		train_dataset = train_dataset.select(random.sample(range(len(train_dataset)), data_args.max_train_samples))
		total = 0
		for example in train_dataset:
			for label in example['labels']:
				if label != -1:
					total += 1
		logger.info("Total number of samples: {}".format(total))

if training_args.do_eval:
	if "validation" not in tokenized_datasets:
		raise ValueError("--do_eval requires a validation dataset")
	eval_dataset = tokenized_datasets["validation"]
	if data_args.max_eval_samples is not None:
		eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

# Optimizer
optimizer = load_optimizer(model, training_args)

# Save eval results
eval_results_df = pd.DataFrame(columns=["epoch", "eval_accuracy", "eval_loss"])


class SaveEvalResultsCallback(TrainerCallback):
	def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
		global eval_results_df
		_metrics = kwargs.pop("metrics")
		cur_epoch: int = int(state.epoch)

		if state.is_world_process_zero:
			eval_result = {
				"epoch"        : cur_epoch,
				"eval_accuracy": _metrics["eval_accuracy"],
				"eval_loss"    : _metrics["eval_loss"]
				}
			eval_result_df = pd.DataFrame([eval_result])
			eval_results_df = pd.concat([eval_results_df, eval_result_df])
			eval_results_df.to_csv(os.path.join(args.output_dir, f"eval_results.csv"), index=False)


# Trainer
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset if training_args.do_train else None,
	eval_dataset=eval_dataset if training_args.do_eval else None,
	tokenizer=tokenizer,
	data_collator=default_data_collator,  # Data collator will default to DataCollatorWithPadding, so we change it.
	optimizers=(optimizer, None),
	compute_metrics=compute_metrics,
	callbacks=[SaveEvalResultsCallback(), EarlyStoppingCallback(early_stopping_patience=5)],
	)

# Training
if training_args.do_train:
	checkpoint = None
	if training_args.resume_from_checkpoint is not None:
		checkpoint = training_args.resume_from_checkpoint
	train_result = trainer.train(resume_from_checkpoint=checkpoint)
	trainer.save_model(output_dir=training_args.output_dir)  # Saves the tokenizer too for easy upload
	max_train_samples = (data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
	metrics = train_result.metrics
	metrics["train_samples"] = min(max_train_samples, len(train_dataset))
	trainer.log_metrics("train", metrics)

# Evaluation
if training_args.do_eval:
	logger.info("*** Evaluate ***")
	logger.info(
		f'Layer weights: {torch.stack([p for n, p in model.scalar_mix.named_parameters() if "scalar" in n]).flatten()}'
		)

	metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
	max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
	metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

	trainer.log_metrics("eval", metrics)

eval_results_df.to_csv(os.path.join(training_args.output_dir, "eval_results.csv"), index=False)
