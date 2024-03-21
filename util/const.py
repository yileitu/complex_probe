# -*- coding: utf-8 -*-
GPT2_EN_PATH = "gpt2"
GPT2_ZH_PATH = "uer/gpt2-chinese-cluecorpussmall"
OLMO_1B_HF_PATH = "allenai/OLMo-1B"

PROJ_NAME = "SP"

LABEL_DICT = {}
LABEL2ID = {}
LABEL_DICT['ner'] = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE',
                     'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT',
                     'QUANTITY', 'TIME', 'WORK_OF_ART']
LABEL_DICT['pos'] = ['$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX',
                     'CC', 'CD', 'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
                     'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
                     'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                     'WDT', 'WP', 'WP$', 'WRB', '``']
LABEL_DICT['const'] = ['ADJP', 'ADVP', 'CONJP', 'EMBED', 'FRAG', 'INTJ', 'LST',
                       'META', 'NAC', 'NML', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'S', 'SBAR',
                       'SBARQ', 'SINV', 'SQ', 'TOP', 'UCP', 'VP', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP',
                       'X']
LABEL_DICT['coref'] = ['False', 'True']
LABEL_DICT['srl'] = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA',
                     'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-DSP',
                     'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB', 'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG',
                     'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-PRX', 'ARGM-REC', 'ARGM-TMP',
                     'C-ARG0', 'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADJ', 'C-ARGM-ADV',
                     'C-ARGM-CAU', 'C-ARGM-COM', 'C-ARGM-DIR', 'C-ARGM-DIS', 'C-ARGM-DSP', 'C-ARGM-EXT',
                     'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-MOD', 'C-ARGM-NEG', 'C-ARGM-PRP', 'C-ARGM-TMP',
                     'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARG5', 'R-ARGM-ADV', 'R-ARGM-CAU',
                     'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-EXT', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR',
                     'R-ARGM-MOD', 'R-ARGM-PNC', 'R-ARGM-PRD', 'R-ARGM-PRP', 'R-ARGM-TMP']

for task in LABEL_DICT:
	LABEL2ID[task] = {label: "label" + str(i) for i, label in enumerate(LABEL_DICT[task])}

MAX_LENGTH = {'pos': 350, 'const': 350, 'ner': 350, 'coref': 280, 'srl': 350}
MAX_TARGET = {'pos': 275, 'const': 175, 'ner': 71, 'coref': 300, 'srl': 11}
IS_UNARY = {'pos': True, 'const': True, 'ner': True, 'coref': False, 'srl': False}