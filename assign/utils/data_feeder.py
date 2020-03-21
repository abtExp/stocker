import librosa
import numpy as np
import pandas as pd

import shutil

from os import listdir
from os.path import exists

import pickle

import torch

from ..utils.data_utils import load_audio, load_target

from ..models.sentence_encoder import SENTENCE_ENCODER


def data_loader(vars, mode='train', folder='', encoder=None, tokenizer=None, model=None, load_mode='text'):
	data_folder = vars.DATA_PATH+'data__'
	batch_size = 0

	print(folder)

	# if len(folder) == 0:
	if mode == 'train':
		data_folder = data_folder+'/train/'
		batch_size = vars.TRAIN_BATCH_SIZE
	elif mode == 'valid':
		data_folder = data_folder+'/valid/'
		batch_size = vars.VALID_BATCH_SIZE
	else:
		data_folder = data_folder+'/test/'
		batch_size = 1
	# else:
	# 	batch_size = 1

	txts = []
	auds = []
	prices = []
	err = ''

	all_datas = listdir(data_folder)

	while len(prices) < batch_size:
		if len(folder) == 0:
			idx = np.random.choice(np.arange(0, len(all_datas)), batch_size, replace=False)[0]
			folder = all_datas[idx]

		company, start_date = folder.split('_')
		# else:
		# 	if mode == 'test':
		# 		company, start_date = folder[folder.rindex('/')+1:].split('_')

		if load_mode == 'audio' or load_mode == 'both':
			aud_features = []

		# Loading Text Features
		if load_mode == 'text' or load_mode == 'both':
			with open(data_folder+folder+'/Text.txt') as f:
				try:
					sentences = f.read()
					sentences = sentences.split('\n')
					sentences = sentences[:vars.MAX_SENTENCES]
					features = []
					for sentence in sentences:
						tokens = tokenizer.encode(sentence, max_length=vars.MAX_SENTENCE_LENGTH, pad_to_max_length=True)
						outputs = model.predict(np.array([tokens], dtype=np.long))
						features.append(outputs)

				except Exception as e:
					# print('Can\'t read! : ', e)
					err = 'Text Data Read Error : {}'.format(data_folder+folder)
					features = []

			features = np.array(features)

			if len(features) < vars.MAX_SENTENCES and len(features) > 0:
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
			if cntr == 0:
				err = 'No Audio Found: {}'.format(data_folder+folder)
			aud_features = np.array(aud_features)

			if len(aud_features) > 0:
				if len(aud_features) < vars.MAX_SENTENCES:
					aud_features = np.concatenate((aud_features, np.zeros((vars.MAX_SENTENCES-len(aud_features), *np.shape(aud_features[0])))))


		labels, error = load_target(vars, company, start_date)

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

					txts.append(features)

			if is_avail:
				prices.append(labels)
			else:
				folder = ''
				if mode == 'test':
					err = 'Can\'t read target : {}, {}'.format(data_folder+folder, err)
					return np.array([]), [], err
		else:
			folder = ''
			err = 'Can\'t read target : {}'.format(data_folder+folder, error)
			if mode == 'test':
				return np.array([]), [], err

	if load_mode == 'audio':
		return np.expand_dims(auds, axis=-1), np.array(prices), err
	elif load_mode == 'text':
		return np.array(txts), np.array(prices), err
	else:
		return [np.array(txts), np.expand_dims(auds, axis=-1)], np.array(prices), err