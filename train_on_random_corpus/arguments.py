# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MyArguments:
	"""
	Custom arguments for fine-tuning.
	"""
	max_length: Optional[int] = field(
		default=64,
		metadata={"help": "The maximum input sequence length after tokenization."},
		)
	data_dir: str = field(
		default=None,
		metadata={"help": "Where to read corpus data."}
		)
	cache_dir: Optional[str] = field(
		default='cache/',
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
		)
	n_gpu: Optional[int] = field(
		default=1,
		metadata={"help": "Number of GPUs to use."},
		)
	model_name: Optional[str] = field(
		default='gpt2',
		metadata={"help": "Model name. Only support 'pythia' for now."},
		)
	model_path: Optional[str] = field(
		default=None,
		metadata={"help": "Path to finetuned model checkpoint."},
		)
	revision: Optional[str] = field(
		default=None,
		metadata={
			"help": "The specific model version (checkpoint) to use. Only apply to Pythia model."
			},
		)
	scale: Optional[str] = field(
		default="160m",
		metadata={
			"help": "Pythia model scale"
			},
		)
	sentence_len: Optional[int] = field(
		default=15,
		metadata={
			"help": "Sentence length in the random corpus."
			},
		)
	sentence_num: Optional[int] = field(
		default=10,
		metadata={
			"help": "Number of sentences in the random corpus."
			},
		)
