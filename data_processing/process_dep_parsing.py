# -*- coding: utf-8 -*-
import json
import os.path


def parse_conllu_to_json_with_span2(input_path: str, stats_path: str = None) -> list:
	"""
	Parse UD CoNLL-U format to JSON format with 2-span format for dependency parsing.
	
	:param input_path: Path to the input CoNLL-U file
	:param stats_path: Path to the output file to write statistics
	:return: List of dependency parsing datapoints in JSON format
	"""
	sentences_json = []
	total_data_points = 0
	discarded_data_points = 0
	current_sentence = {"text": "", "targets": []}
	discard_current_sentence = False  # Flag to indicate if the current sentence should be discarded

	with open(input_path, encoding='utf-8') as file:
		for line in file:
			line = line.strip()
			if line.startswith("#"):
				continue  # Ignore comment lines
			elif line == "":
				total_data_points += 1
				if not discard_current_sentence and current_sentence["text"]:  # End of sentence, and it's not discarded
					sentences_json.append(current_sentence)
				else:
					discarded_data_points += 1
				# Reset for the next sentence
				current_sentence = {"text": "", "targets": []}
				discard_current_sentence = False
			else:
				try:
					parts = line.split('\t')
					# TODO: Handle implicit structure
					# Handle multiword tokens and implicit structure by skipping
					if "-" in parts[0] or "." in parts[0] or len(parts) != 10:
						continue
					if len(parts) != 10 or "-" in parts[0]:  # Skip non-standard lines and multiword tokens
						raise ValueError("Invalid line format")
					token_id = int(parts[0]) - 1  # Token index, starting from 0
					word = parts[1]
					head = int(parts[6]) - 1  # Head index, starting from 0
					label = parts[7]  # Dependency label
					if current_sentence["text"]:
						current_sentence["text"] += " " + word
					else:
						current_sentence["text"] = word
					# Add dependency target
					if head >= 0:  # Exclude root dependencies
						current_sentence["targets"].append(
							{
								"span1": [token_id, token_id],
								"span2": [head, head],
								"label": label
								}
							)
				except ValueError as e:
					print(f"Error parsing line: {line}")
					print(e)
					discard_current_sentence = True  # Mark the current sentence for discarding due to an error

		# Handle the last sentence case
		if not discard_current_sentence and current_sentence["text"]:
			sentences_json.append(current_sentence)
		elif current_sentence["text"]:
			total_data_points += 1
			discarded_data_points += 1

	# Write statistics to a text file
	if stats_path:
		with open(stats_path, 'w', encoding='utf-8') as file:
			file.write(f"Total data points: {total_data_points}\n")
			file.write(f"Discarded data points: {discarded_data_points}\n")
			file.write(f"Retained data points: {total_data_points - discarded_data_points}\n")

	return sentences_json


if __name__ == "__main__":
	input_dir = '../dataset/UD/UD_English-EWT-master'
	output_dir = '../dataset/UD/'

	input_train = os.path.join(input_dir, 'en_ewt-ud-train.conllu')
	train_json = parse_conllu_to_json_with_span2(
		input_train, stats_path=os.path.join(output_dir, 'en_ewt-ud-train_stats.txt')
		)
	output_train = os.path.join(output_dir, 'en_ewt-ud-train.json')
	with open(output_train, 'w', encoding='utf-8') as f:
		json.dump(train_json, f, indent=2, ensure_ascii=False)

	input_test = os.path.join(input_dir, 'en_ewt-ud-test.conllu')
	test_json = parse_conllu_to_json_with_span2(
		input_test, stats_path=os.path.join(output_dir, 'en_ewt-ud-test_stats.txt')
		)
	output_test = os.path.join(output_dir, 'en_ewt-ud-test.json')
	with open(output_test, 'w', encoding='utf-8') as f:
		json.dump(test_json, f, indent=2, ensure_ascii=False)

	input_dev = os.path.join(input_dir, 'en_ewt-ud-dev.conllu')
	dev_json = parse_conllu_to_json_with_span2(
		input_dev, stats_path=os.path.join(output_dir, 'en_ewt-ud-dev_stats.txt')
		)
	output_dev = os.path.join(output_dir, 'en_ewt-ud-dev.json')
	with open(output_dev, 'w', encoding='utf-8') as f:
		json.dump(dev_json, f, indent=2, ensure_ascii=False)
