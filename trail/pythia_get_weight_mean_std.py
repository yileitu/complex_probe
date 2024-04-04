# -*- coding: utf-8 -*-
import math

from transformers import GPTNeoXForCausalLM

model = GPTNeoXForCausalLM.from_pretrained(
	"EleutherAI/pythia-160m-deduped",
	revision="step0",
	)

model_dim = 768  # Pythia-6.9b

# compute right std values of the two init methods
# reference https://github.com/EleutherAI/gpt-neox/blob/v1.0/megatron/model/init_functions.py#L101-L118

small_init_std = (2 / (5 * model_dim)) ** 0.5
wang_init_std = 2 / (32 * math.sqrt(model_dim))
print('small_init_std:', small_init_std)
print('wang_init_std:', wang_init_std)

for n, p in model.named_parameters():
	print(n, p.shape, f"Mean: {p.mean().item()} Std: {p.std().item()}")
