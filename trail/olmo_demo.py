# from huggingface_hub import list_repo_refs
# from transformers import AutoModelForCausalLM

# # allenai/OLMo-7B
# BRANCHES_BY_STEPS = ["step0-tokens0B", "step111000-tokens491B", "step222000-tokens982B", "step333000-tokens1473B",
#                      "step444000-tokens1964B", "step555000-tokens2455B"]  # Divided by 5
# BRANCHES_BY_TOKENS = ["step0-tokens0B", "step111000-tokens491B", "step223000-tokens986B", "step334000-tokens1478B",
#                       "step446000-tokens1973B", 'step557000-tokens2464B']  # Divided by 5
#
# out = list_repo_refs("allenai/OLMo-1B")
# branches = [b.name for b in
#             out.branches]  # 'step556000-tokens2460B', 'step557000-tokens2464B', 'step0-tokens0B', 'main'

# allenai/OLMo-1B
BRANCHES_BY_TOKENS = ["step20000-tokens84B", "step117850-tokens494B", "step330000-tokens1384B", "step466000-tokens1955B",
                      "step601000-tokens2521B", 'step738000-tokens3095B']  # Divided by 5

# out = list_repo_refs("allenai/OLMo-1B")
# branches = [b.name for b in
#             out.branches]  # 'step556000-tokens2460B', 'step557000-tokens2464B', 'step0-tokens0B', 'main'
#
#
#
# def extract_numbers(s):
# 	"""
# 	从给定的字符串中提取step和tokens后的数字。
# 	假设字符串格式为'step{step_number}-tokens{token_number}B'。
# 	"""
# 	step_part, tokens_part = s.split('-')  # 分割'step'部分和'tokens'部分
# 	step_number = int(step_part.replace('step', ''))  # 提取并转换step数字
# 	token_number = int(tokens_part.replace('tokens', '').replace('B', ''))  # 提取并转换tokens数字
# 	return step_number, token_number
#
#
# sorted_branches = sorted(
# 	branches, key=lambda x: (float('inf'), float('inf')) if x in ['main', 'step0-tokens0B'] else extract_numbers(
# 		x
# 		)
# 	)
# print(sorted_branches)
#

import hf_olmo
from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)
# Print type of olmo
print(f"Type of olmo: {type(olmo)}")
