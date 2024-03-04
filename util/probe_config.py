# -*- coding: utf-8 -*-
from transformers import PretrainedConfig


class ProbeConfig(PretrainedConfig):
	model_type = "probe_on_llm"

	def __init__(self, **kwargs):
		config = {}
		config.update(kwargs)
		super().__init__(**config)
