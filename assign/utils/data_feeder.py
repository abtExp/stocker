import librosa
import numpy as np
import pandas as pd

import shutil

from os import listdir
from os.path import exists

import pickle

import torch

import keras.preprocessing.text as text
import keras.preprocessing.sequence as seq
from keras.utils import to_categorical

from transformers import BertModel, BertTokenizer

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split

from ..utils.data_utils import *

from ..models.sentence_encoder import SENTENCE_ENCODER


def data_loader(vars, mode='train', encoder=None, tokenizer=None, model=None, load_mode='text'):
	data_folder = vars.DATA_PATH+'data'
	batch_size = 0

	if mode == 'train':
		data_folder = data_folder+'/train/'
		batch_size = vars.TRAIN_BATCH_SIZE
	elif mode == 'valid':
		data_folder = data_folder+'/valid/'
		batch_size = vars.VALID_BATCH_SIZE
	else:
		data_folder = data_folder+'/test/'

	txts = []
	auds = []
	prices = []

	all_datas = listdir(data_folder)

	while len(prices) < batch_size:
		idx = np.random.choice(np.arange(0, len(all_datas)), batch_size, replace=False)[0]
		folder = all_datas[idx]
		company, start_date = folder.split('_')

		if load_mode == 'audio' or load_mode == 'both':
			aud_features = []

		# Loading Text Features
		if load_mode == 'text' or load_mode == 'both':
			with open(data_folder+folder+'/Text.txt') as f:
				sentences = f.read()
				sentences = sentences.split('\n')
				sentences = sentences[:vars.MAX_SENTENCES]
				features = []
				for sentence in sentences:
					tokens = tokenizer.encode(sentence, max_length=vars.MAX_SENTENCE_LENGTH, pad_to_max_length=True)
					# tokens = torch.tensor([tokens])
					outputs = model.predict(np.array([tokens], dtype=np.long))
					# return
					features.append(outputs)

			features = np.array(features)

			if len(features) < vars.MAX_SENTENCES:
				features = np.concatenate((features, np.zeros((vars.MAX_SENTENCES-len(features), *np.shape(features[0])))))


		# Loading Audio Features
		if load_mode == 'audio' or load_mode == 'both':
			cntr = 0
			for i in listdir(data_folder+folder+'/Audio/'):
				aud = load_audio(vars, data_folder+folder+'/Audio/'+i)
				aud = encoder.predict(aud)
				aud_features.append(aud)
				cntr += 1
				if cntr >= vars.MAX_SENTENCES:
					break
			aud_features = np.array(aud_features)

			if len(aud_features) > 0:
				if len(aud_features) < vars.MAX_SENTENCES:
					aud_features = np.concatenate((aud_features, np.zeros((vars.MAX_SENTENCES-len(aud_features), *np.shape(aud_features[0])))))


		labels = load_target(vars, company, start_date)

		if type(labels) == np.ndarray:
			is_avail = False
			is_aud_avail = False

			if load_mode == 'audio' or load_mode == 'both':
				if len(aud_features) > 0:
					if load_mode == 'audio':
						is_avail = True
					else:
						is_aud_avail = True

					auds.append(aud_features)

			if load_mode == 'text' or load_mode == 'both':
				if len(features) > 0:
					if load_mode == 'both':
						if is_aud_avail:
							is_avail = True
					else:
						is_avail = True

					xts.append(features)

			if is_avail:
				prices.append(labels)

	if load_mode == 'audio':
		return np.expand_dims(auds, axis=-1), np.array(prices)
	elif load_mode == 'text':
		return np.array(txts), np.array(prices)
	else:
		return [np.array(txts), np.expand_dims(auds, axis=-1)], np.array(prices)