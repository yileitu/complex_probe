import os
import re

import pandas as pd

MODEL: str = "gpt2"
TASK: str = "ner"
IS_MLP: bool = True

# The main directory where all the experiments are stored
if IS_MLP:
	main_dir = f'../outputs/{MODEL}/mlp/{TASK}'
else:
	main_dir = f'../outputs/{MODEL}/lr/{TASK}'

# 正则表达式来检查第一层和第二层文件夹的格式
first_level_dir_pattern = re.compile(r'ner-GPT2-MLP-Dim\d+-Layer\d+')
second_level_dir_pattern = re.compile(
	r'Epoch\d+-LR(?:[0-9]*\.?[0-9]+(?:e-?\d+)?)-(Pretrained|Randomized)-Dev'
	)
learning_rate_pattern = re.compile(r'LR([0-9]*\.?[0-9]+(?:e-?\d+)?)')
number_pattern = re.compile(r'\d+')


def extract_numbers(s):
	return number_pattern.findall(s)


def extract_learning_rate(dirname):
	match = learning_rate_pattern.search(dirname)
	if match:
		return match.group(1)
	return None


def is_valid_first_level_dir(dirname):
	return bool(first_level_dir_pattern.fullmatch(dirname))


def is_valid_second_level_dir(dirname):
	return bool(second_level_dir_pattern.fullmatch(dirname))


# This function will find all first-level directories
def find_first_level_dirs(main_dir):
	temp_all_dirs = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
	return [d for d in temp_all_dirs if is_valid_first_level_dir(d)]


# This function reads the CSV and extracts the required information
def get_eval_results(csv_file):
	df = pd.read_csv(csv_file)
	val_acc_info = df.iloc[-2]  # Second-to-last line
	test_acc_info = df.iloc[-1]  # Last line
	return val_acc_info, test_acc_info


#  Extract learning rate and group (Pretrained or Randomized) for GPT2
def extract_lr_and_group(sub_dir_name):
	match = re.search(r'LR([0-9e.-]+)-(Pretrained|Randomized)', sub_dir_name)
	if match:
		return match.group(1), match.group(2)
	return None, None


# Find all the first-level directories
first_level_dirs = find_first_level_dirs(main_dir)

best_results = {
	'Pretrained': {},
	'Randomized': {}
	}

for dir_name in first_level_dirs:
	dim, layer = dir_name.split('-')[3:5]
	dim = extract_numbers(dim)[0]
	layer = extract_numbers(layer)[0]

	second_level_dirs = [d for d in os.listdir(os.path.join(main_dir, dir_name)) if is_valid_second_level_dir(d)]
	print(f"{dir_name}\t\t{second_level_dirs}")

	max_val_acc = 0
	best_lr = None
	max_epoch = None
	test_acc = None
	# Process each second-level directory
	for sub_dir in second_level_dirs:
		lr, group = extract_lr_and_group(sub_dir)
		if (lr is None) or (group is None):
			continue
		eval_csv_path = os.path.join(main_dir, dir_name, sub_dir, 'eval_results.csv')
		val_acc_info, test_acc_info = get_eval_results(eval_csv_path)
		try:
			# Update the best results for each group if the current validation accuracy is higher
			if layer not in best_results[group] or \
					val_acc_info['eval_accuracy'] > best_results[group][layer]['max_val_acc']:
				best_results[group][layer] = {
					'max_val_acc'  : val_acc_info['eval_accuracy'],
					'learning_rate': lr,
					'max_epoch'    : val_acc_info['epoch'],
					'test_accuracy': test_acc_info['eval_accuracy'],
					'group'        : group,
					'dim'          : dim
					}
		except Exception as e:
			print(f"Error processing {eval_csv_path}. Error: {e}")

# Convert the best results to a DataFrame
summary_data = []
for group, layers in best_results.items():
	for layer, data in layers.items():
		summary_data.append(
			{
				'group'          : data['group'],
				'layer'          : layer,
				'dim'            : data['dim'],
				'max_val_acc'    : data['max_val_acc'],
				'learning_rate'  : data['learning_rate'],
				'converged_epoch': data['max_epoch'],
				'test_acc'       : data['test_accuracy']
				}
			)

summary_df = pd.DataFrame(summary_data)
summary_df['dim'] = summary_df['dim'].astype(int)
summary_df['layer'] = summary_df['layer'].astype(int)
summary_df.sort_values(by=['group', 'layer'], ascending=[True, True], inplace=True)

# Save the results to a CSV file
summary_csv_path = f'{main_dir}/probing_results_summary.xlsx'
summary_df.to_excel(summary_csv_path, index=False)
