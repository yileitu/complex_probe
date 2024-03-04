# -*- coding: utf-8 -*-

from pytorch_transformers import AutoConfig
from transformers import HfArgumentParser, TrainingArguments, set_seed

from util.dp_arguments import DataTrainingArguments, Gpt2Arguments
from util.func import load_raw_dataset, post_process_gpt2_args, post_process_training_args, set_gpu_env, set_logger, \
	set_wandb

parser = HfArgumentParser((Gpt2Arguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Setups
wandb_proj_name, serial = set_wandb(training_args, model_args, data_args)
set_seed(training_args.seed)
logger = set_logger(training_args)
device = set_gpu_env(num_gpus=model_args.n_gpu)

post_process_gpt2_args(gpt2_args=model_args)
post_process_training_args(training_args=training_args, wandb_proj_name=wandb_proj_name, serial=serial)

load_raw_dataset(model_args, data_args, training_args, logger)
label2id = {label: i for i, label in enumerate(LABEL_DICT[data_args.task])}

# Load GPT2 config
config_kwargs = {
	"cache_dir"     : model_args.cache_dir,
	"revision"      : model_args.model_revision,
	"use_auth_token": True if model_args.use_auth_token else None,
	}
if model_args.config_name:
	config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
elif model_args.gpt2_name_or_path:
	config = AutoConfig.from_pretrained(model_args.gpt2_name_or_path, **config_kwargs)
	logger.info(f"Model config loaded from pretrained ckpt {model_args.gpt2_name_or_path}")
config.num_labels = len(label2id)
config.saturated = model_args.saturated
config.onehot = model_args.onehot
if config.onehot:
	logger.info("Using onehot embeddings.")
config.chinese = model_args.chinese
if config.chinese:
	logger.info("Using GPT2-Chinese.")
