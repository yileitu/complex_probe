from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

SCALE: str = "160m"
MODEL_PATH: str = f"EleutherAI/pythia-{SCALE}-deduped"
REVISION: str = "step0"
CACHE_DIR: str = "../cache/"

set_seed(21946520)
tokenizer_kwargs = {
	"cache_dir": CACHE_DIR,
	"revision" : REVISION,
	}
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, **tokenizer_kwargs)
vocab = tokenizer.get_vocab()


def generate_sentence(vocab, eos_token_id, length: int = 10) -> Tuple[str, np.ndarray]:
	"""
	Randomly generate a sentence of a given length.

	:param vocab: Vocabulary of the tokenizer
	:param eos_token_id: ID of the end-of-sentence token
	:param length: Fixed length of the sentence to generate
	:return: A tuple of the generated sentence and its token IDs.
	"""
	sentence_ids = np.random.choice(list(vocab.values()), size=length)
	sentence_ids = np.append(sentence_ids, eos_token_id)
	sentence = tokenizer.decode(sentence_ids, skip_special_tokens=False)

	return sentence, sentence_ids


def generate_corpus(tokenizer: AutoTokenizer, num_sentences: int = 100000, sentence_length: int = 10) -> List[str]:
	corpus = []
	all_special_ids = set(tokenizer.all_special_ids)
	eos_token_id = tokenizer.eos_token_id
	vocab_without_specials = {token: idx for token, idx in tokenizer.get_vocab().items() if idx not in all_special_ids}
	for _ in tqdm(range(num_sentences)):
		sentence, _ = generate_sentence(vocab=vocab_without_specials, eos_token_id=eos_token_id, length=sentence_length)
		corpus.append(sentence)
	return corpus


def validate_sentence(sentence, original_ids):
	# 将生成的句子转换回ID
	encoded_ids = tokenizer.encode(sentence, add_special_tokens=False)
	print("Encoded IDs:", encoded_ids)

	return np.array_equal(original_ids, encoded_ids)


SENTENCE_LEN = 15
SENTENCE_NUM = 1000000
corpus = generate_corpus(tokenizer=tokenizer, num_sentences=SENTENCE_NUM, sentence_length=SENTENCE_LEN)

# 存储语料库到文本文件
corpus_name = f'generated_corpus_len{SENTENCE_LEN}_num{SENTENCE_NUM}.txt'
with open(corpus_name, 'w', encoding='utf-8') as f:
	for sentence in corpus:
		f.write(sentence + '\n')

print("Corpus generated and saved.")
