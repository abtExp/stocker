from assign import vars
from assign.models import SPEAKER_ENCODER, SENTENCE_ENCODER
from assign.utils.data_feeder import data_loader

from transformers import BertModel, BertTokenizer

import numpy as np

import tensorflow as tf

from argparse import ArgumentParser


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('mode', help='modality')

	args = parser.parse_args()

	tokenizer = None
	text_encoder = None
	speech_encoder = None

	if args.mode == 'text' or args.mode == 'both':
		text_encoder = SENTENCE_ENCODER(vars)
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	if args.mode == 'audio' or args.mode == 'both':
		speech_encoder = SPEAKER_ENCODER(vars, graph=tf.get_default_graph())

	tx, ty, _ = data_loader(vars, encoder=speech_encoder, tokenizer=tokenizer, model=text_encoder, load_mode=args.mode)

	if args.mode != 'both':
		print(tx.shape)
	else:
		print(tx[0].shape)
		print(tx[1].shape)

	print(ty.shape)