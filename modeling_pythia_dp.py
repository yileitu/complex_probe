from copy import deepcopy
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import wandb
from allennlp_light.modules import scalar_mix
from allennlp_light.modules.span_extractors import SelfAttentiveSpanExtractor
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPTNeoXForCausalLM

from util.probe_config import ProbeConfig
from util.func import DiagnosticProbingOutputs


class PythiaForDiagnosticProbing(GPTNeoXForCausalLM):
	def __init__(self, pythia, pythia_config, probe_config: ProbeConfig):
		super().__init__(config=pythia_config)
		self.model = pythia

		if probe_config.onehot is False:
			for param in self.model.parameters():
				param.requires_grad = False
		else:
			for param in self.model.parameters():
				param.requires_grad = True
			print("Onehot is True. All parameters are trainable.")

		# Model parallel
		self.model_parallel = False
		self.device_map = None

		# Configs from probe
		self.probe_config = probe_config
		self.unary = probe_config.unary
		self.num_labels = probe_config.num_labels
		self.mlp_dropout = probe_config.mlp_dropout
		self.mlp_dim = probe_config.mlp_dim
		self.mlp_layers: int = probe_config.mlp_layers
		self.use_mlp = probe_config.use_mlp
		self.onehot: bool = probe_config.onehot

		# Configs from Pythia
		self.pythia_config = pythia_config
		self.n_embd = pythia_config.hidden_size
		self.scalar_mix = scalar_mix.ScalarMix(pythia_config.num_hidden_layers)
		self.vocab_size = pythia_config.vocab_size
		# self.onehot_scalar_mix = scalar_mix.ScalarMix(1)

		if self.onehot is False:
			self.proj1 = nn.Conv1d(self.n_embd, self.mlp_dim, kernel_size=1)
			print("Projection 1: ", self.proj1)
		else:
			self.proj1 = nn.Conv1d(self.vocab_size, self.mlp_dim, kernel_size=1)
		self.span_extractor1 = SelfAttentiveSpanExtractor(self.mlp_dim)
		self.input_dim = self.span_extractor1.get_output_dim()
		if not self.unary:
			if self.onehot is False:
				self.proj2 = nn.Conv1d(self.n_embd, self.mlp_dim, kernel_size=1)
			else:
				self.proj2 = nn.Conv1d(self.vocab_size, self.mlp_dim, kernel_size=1)
			self.span_extractor2 = SelfAttentiveSpanExtractor(self.mlp_dim)
			self.input_dim += self.span_extractor2.get_output_dim()

		wandb.run.summary["input_layer_dim"] = self.input_dim
		print("Input layer dim: ", self.input_dim)

		if not self.use_mlp:
			lin_module_list = []
			if self.mlp_layers == 1:
				self.classifier = nn.Sequential(
					nn.Linear(self.input_dim, self.mlp_dim),
					nn.Linear(self.mlp_dim, self.num_labels)
					)
			elif self.mlp_layers >= 2:
				lin_module_list.append(nn.Linear(self.input_dim, self.mlp_dim))
				for _ in range(self.mlp_layers - 1):
					lin_module_list.append(nn.Linear(self.mlp_dim, self.mlp_dim))
				lin_module_list.append(nn.Linear(self.mlp_dim, self.num_labels))
				self.classifier = nn.Sequential(*lin_module_list)
		else:
			input_layer_list = [
				nn.Linear(self.input_dim, self.mlp_dim),
				nn.ReLU(),
				nn.LayerNorm(self.mlp_dim),
				nn.Dropout(self.mlp_dropout),
				]
			output_layer_list = [nn.Linear(self.mlp_dim, self.num_labels)]
			if self.mlp_layers == 1:
				classifier_module_list = deepcopy(input_layer_list) + deepcopy(output_layer_list)
			elif self.mlp_layers >= 2:
				classifier_module_list = deepcopy(input_layer_list)
				for _ in range(self.mlp_layers - 1):
					classifier_module_list.append(nn.Linear(self.mlp_dim, self.mlp_dim))
					classifier_module_list.append(nn.ReLU())
					classifier_module_list.append(nn.LayerNorm(self.mlp_dim))
					classifier_module_list.append(nn.Dropout(self.mlp_dropout))
				classifier_module_list += deepcopy(output_layer_list)
			else:
				raise ValueError(f"The num of MLP layers should be a positive integer. Your input is {self.mlp_layer}")
			self.classifier = nn.Sequential(*classifier_module_list)
			print("MLP Architecture: ", self.classifier)

	# self.w = nn.Parameter(torch.empty([config.num_hidden_layers, config.num_hidden_layers]))
	# nn.init.xavier_uniform(self.w)
	# self.num_of_heads = None
	# self.use_dsp = False

	def forward(
			self,
			input_ids: Optional[torch.LongTensor] = None,
			attention_mask: Optional[torch.FloatTensor] = None,
			position_ids: Optional[torch.LongTensor] = None,
			inputs_embeds: Optional[torch.FloatTensor] = None,
			head_mask: Optional[torch.FloatTensor] = None,
			past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
			labels: Optional[torch.LongTensor] = None,
			use_cache: Optional[bool] = None,
			output_attentions: Optional[bool] = None,
			output_hidden_states: Optional[bool] = True,
			return_dict: Optional[bool] = None,
			span1s=None,
			span2s=None,
			):
		return_dict = return_dict if return_dict is not None else self.pythia_config.use_return_dict

		# print(f"Input batch_size from input_ids: {input_ids.size(0)}") if input_ids is not None else None
		# print(f"Input batch_size from span1s: {span1s.size(0)}") if span1s is not None else None
		# print(f"Input batch_size from span2s: {span2s.size(0)}") if span2s is not None else None

		# if self.use_dsp:
		# 	head_mask = STEFunction.apply(self.w.view(-1), self.num_of_heads).view_as(self.w)
		# 	self.apply_masks(head_mask)

		if self.onehot is False:
			base_model_outputs = self.model(
				input_ids,
				attention_mask=attention_mask,
				position_ids=position_ids,
				head_mask=head_mask,
				inputs_embeds=inputs_embeds,
				past_key_values=past_key_values,
				use_cache=use_cache,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
				)
			if not self.use_mlp:
				contextual_embeddings = base_model_outputs[0]
			else:
				all_hidden_states = base_model_outputs.hidden_states[1:]
				contextual_embeddings = self.scalar_mix(all_hidden_states)
			# print("Shape of all_hidden_states: ", len(all_hidden_states), [h.size() for h in all_hidden_states])
			# print("Shape of contextual_embeddings: ", contextual_embeddings.size())
		else:
			contextual_embeddings = torch.nn.functional.one_hot(input_ids, num_classes=self.config.vocab_size).half()

		span_mask = span1s[:, :, 0] != -1
		se_proj1 = self.proj1(contextual_embeddings.transpose(1, 2)).transpose(2, 1).contiguous()
		span1_emb = self.span_extractor1(se_proj1, span1s, span_indices_mask=span_mask.long())
		if not self.unary:
			se_proj2 = self.proj2(contextual_embeddings.transpose(1, 2)).transpose(2, 1).contiguous()
			span2_emb = self.span_extractor2(se_proj2, span2s, span_indices_mask=span_mask.long())
			span_emb = torch.cat([span1_emb, span2_emb], dim=2)
		else:
			span_emb = span1_emb

		logits = self.classifier(span_emb)
		loss_fct = CrossEntropyLoss()
		loss = loss_fct(logits[span_mask], labels[span_mask])

		corrections = logits[span_mask].argmax(-1) == labels[span_mask]
		correct_counts = corrections.sum()
		total_counts = len(corrections)
		accuracy = torch.tensor([[correct_counts, total_counts]], device=corrections.device)

		if not return_dict:
			output = (accuracy,)
			return ((loss,) + output) if loss is not None else output

		return DiagnosticProbingOutputs(
			loss=loss,
			logits=accuracy,
			)
