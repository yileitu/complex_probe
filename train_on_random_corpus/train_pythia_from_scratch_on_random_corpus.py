# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime

from datasets import load_dataset

script_dir = os.path.dirname(__file__)  # 获取当前脚本文件的目录
parent_dir = os.path.dirname(script_dir)  # 获取父目录
sys.path.insert(0, parent_dir)  # 将父目录添加到sys.path

from dataclasses import asdict

import wandb
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, GPTNeoXForCausalLM, HfArgumentParser, \
	Trainer, TrainingArguments, set_seed

from arguments import MyArguments
from util.func import set_gpu_env, set_logger, transform_dict

# Parse arguments
parser = HfArgumentParser((MyArguments, TrainingArguments))
my_args, training_args = parser.parse_args_into_dataclasses()
my_args: MyArguments
training_args: TrainingArguments

# Setup wandb
wandb_proj_name = f"TrainOnRandomCorpus-{my_args.model_name.upper()}"
serial = f"Epoch{int(training_args.num_train_epochs)}-LR{training_args.learning_rate}-Seed{training_args.seed}"
os.environ["WANDB_PROJECT"] = wandb_proj_name
wandb.init(
	project=wandb_proj_name,
	name=serial,
	)

# Setup training args
training_args.report_to = ["wandb"]
training_args.logging_steps = 50
training_args.run_name = serial
training_args.save_total_limit = 1
training_args.save_strategy = "no"
training_args.evaluation_strategy = "no"
wandb.log(transform_dict(asdict(my_args)))

# Misc Setup
set_seed(training_args.seed)
logger = set_logger(training_args=training_args)
device = set_gpu_env(num_gpus=my_args.n_gpu)

# Load tokenizer and model
if my_args.model_name == 'pythia':
	pythia_config_kwargs = {
		"cache_dir"     : my_args.cache_dir,
		"revision"      : my_args.revision,
		"use_auth_token": True,
		"use_cache"     : True,
		}
	pythia_hf_path = f"EleutherAI/pythia-{my_args.scale}-deduped"
	model = GPTNeoXForCausalLM.from_pretrained(pythia_hf_path, **pythia_config_kwargs)
	tokenizer = AutoTokenizer.from_pretrained(pythia_hf_path)
	tokenizer.pad_token = tokenizer.eos_token
else:
	raise ValueError(f"Unsupported model name: {my_args.model_name}")
model.to(device)

# Load and tokenize random corpus
corpus_path = f"generated_corpus_len{my_args.sentence_len}_num{my_args.sentence_num}.txt"
dataset = load_dataset('text', data_files={'train': corpus_path})


def tokenize_function(examples):
	return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=my_args.max_length)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load labeled data
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
	model=model,
	args=training_args,
	data_collator=data_collator,
	train_dataset=tokenized_datasets["train"],
	)

trainer.train()
current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
save_dir = f"len{my_args.sentence_len}_num{my_args.sentence_num}-{current_timestamp}"
timestamped_dir = os.path.join(training_args.output_dir, save_dir)
trainer.save_model(output_dir=timestamped_dir)
tokenizer.save_pretrained(save_directory=timestamped_dir)
logger.info(f"Saved model and tokenizer to {timestamped_dir}")
