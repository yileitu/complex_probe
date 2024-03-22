# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Gpt2Arguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
	"""
	gpt2_name_or_path: Optional[str] = field(
		default=None,
		metadata={
			"help": "The model checkpoint for weights initialization."
			        "Don't set if you want to train a model from scratch."
			},
		)
	chinese: bool = field(
		default=False,
		metadata={
			"help": "Whether to use GPT2-Chinese model."
			},
		)
	model_path: Optional[str] = field(
		default=None,
		metadata={
			"help": "Path to trained model."
			        "Don't set if you want to train a model from scratch."
			},
		)
	config_overrides: Optional[str] = field(
		default=None,
		metadata={
			"help": "Override some existing default config settings when a model is trained from scratch. Example: "
			        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
			},
		)
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
		)
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
		)
	cache_dir: Optional[str] = field(
		default='cache/',
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
		)
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
		)
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
			        "with private models)."
			},
		)
	use_mlp: bool = field(
		default=True,
		metadata={
			"help": "use mlp or linear regression"
			},
		)
	mlp_dropout: Optional[float] = field(
		default=0.2,
		metadata={"help": "Dropout in MLP model."},
		)
	mlp_dim: Optional[int] = field(
		default=512,
		metadata={"help": "Dimension of hidden states of MLP model."},
		)
	mlp_layers: Optional[int] = field(
		default=1,
		metadata={"help": "The number of layers of MLP model."},
		)
	num_of_heads: Optional[int] = field(
		default=96,
		metadata={"help": "Number of heads left unpruned."},
		)
	pruning_lr: Optional[float] = field(
		default=0.1,
		metadata={"help": "Learning rate for head importance variables."},
		)
	do_prune: Optional[bool] = field(
		default=False,
		metadata={"help": "Whether heads are pruned."},
		)
	randomized: bool = field(
		default=False,
		metadata={
			"help": "If true, load the architecture of the model only, without pretrained weights. "
			        "By default (randomized=False), load the whole pretrained model."
			},
		)
	dev: bool = field(
		default=False,
		metadata={
			"help": "If true, use development dataset to do evaluation. Otherwise use test dataset."
			},
		)
	verbose: int = field(
		default=0,
		metadata={
			"help": "How to group wandb experiments."
			},
		)
	saturated: bool = field(
		default=False,
		metadata={
			"help": "Saturated attention mode."
			},
		)
	onehot: bool = field(
		default=False,
		metadata={
			"help": "If true, extract the embeddings from GPT2 and then pass them as input to the probe."
			},
		)
	branch: Optional[str] = field(
		default=None,
		metadata={
			"help": "The specific model version (checkpoint) to use. Only apply to OLMo model."
			},
		)


@dataclass
class OlmoArguments:
	"""
	Olmo Arguments
	"""
	branch: Optional[str] = field(
		default=None,
		metadata={
			"help": "The specific model version (checkpoint) to use. Only apply to OLMo model."
			},
		)
	model_path: Optional[str] = field(
		default=None,
		metadata={
			"help": "Path to trained model."
			        "Don't set if you want to train a model from scratch."
			},
		)
	cache_dir: Optional[str] = field(
		default='cache/',
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
		)
	use_mlp: bool = field(
		default=True,
		metadata={
			"help": "Use mlp if True, otherwise use linear regression."
			},
		)
	mlp_dropout: Optional[float] = field(
		default=0.0,
		metadata={"help": "Dropout in MLP model."},
		)
	mlp_dim: Optional[int] = field(
		default=512,
		metadata={"help": "Dimension of hidden states of MLP model."},
		)
	mlp_layers: Optional[int] = field(
		default=1,
		metadata={"help": "The number of layers of MLP model."},
		)
	num_of_heads: Optional[int] = field(
		default=96,
		metadata={"help": "Number of heads left unpruned."},
		)
	dev: bool = field(
		default=False,
		metadata={
			"help": "If true, use development dataset to do evaluation. Otherwise use test dataset."
			},
		)
	onehot: bool = field(
		default=False,
		metadata={
			"help": "If true, extract the embeddings from GPT2 and then pass them as input to the probe."
			},
		)


@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""
	n_gpu: int = field(
		default=1,
		metadata={
			"help": "Number of GPUs to use."
			},
		)
	data_dir: Optional[str] = field(
		default=None, metadata={"help": "Where data is stored"}
		)
	task: Optional[str] = field(
		default='ner',
		metadata={"help": "Tasks, one or more of {pos, const, coref, ner, srl}."},
		)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
			        "value if set."
			},
		)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
			        "value if set."
			},
		)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
		)
	preprocessing_num_workers: Optional[int] = field(
		default=None,
		metadata={"help": "The number of processes to use for the preprocessing."},
		)


@dataclass
class PythiaArguments:
	"""
	Pythia Arguments
	"""
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
	model_path: Optional[str] = field(
		default=None,
		metadata={
			"help": "HuggingFace path or local path to checkpoints."
			},
		)
	cache_dir: Optional[str] = field(
		default='cache/',
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
		)
	use_mlp: bool = field(
		default=True,
		metadata={
			"help": "Use mlp if True, otherwise use linear regression."
			},
		)
	mlp_dropout: Optional[float] = field(
		default=0.0,
		metadata={"help": "Dropout in MLP model."},
		)
	mlp_dim: Optional[int] = field(
		default=512,
		metadata={"help": "Dimension of hidden states of MLP model."},
		)
	mlp_layers: Optional[int] = field(
		default=1,
		metadata={"help": "The number of layers of MLP model."},
		)
	num_of_heads: Optional[int] = field(
		default=96,
		metadata={"help": "Number of heads left unpruned."},
		)
	dev: bool = field(
		default=False,
		metadata={
			"help": "If true, use development dataset to do evaluation. Otherwise use test dataset."
			},
		)
	onehot: bool = field(
		default=False,
		metadata={
			"help": "If true, extract the embeddings from GPT2 and then pass them as input to the probe."
			},
		)
