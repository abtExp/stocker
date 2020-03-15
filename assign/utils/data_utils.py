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

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split


import numpy as np
import librosa

import subprocess


def convert_to_wav(input_file):
	output_file = input_file[:input_file.rindex('.')]+'.wav'
	command = 'ffmpeg -y -i {} -acodec pcm_s16le -ac 1 -ar 16000 {} -nostats -hide_banner -loglevel quiet'.format(input_file, output_file)
	subprocess.call(command, shell=True)


def length_normalize(audio, sample_len):
	if len(audio) < sample_len:
		sample = np.zeros(sample_len)
		sample[:len(audio)] = audio
	else:
		sample = audio[:sample_len]

	return sample


def load_audio(vars, aud_path):
	X, _ = librosa.load(aud_path, res_type='kaiser_fast', sr=vars.FRAME_RATE, mono=True)
	features = []

	max_aud_len = int(vars.FRAME_RATE * vars.MAX_AUDIO_DURATION)

	aud = length_normalize(X, max_aud_len)

	num_frames_per_segment = int(vars.MAX_SEGMENT_LENGTH * vars.FRAME_RATE)

	num_samples = int(max_aud_len // num_frames_per_segment)

	for i in range(num_samples):
		sample = aud[int(i*num_frames_per_segment) : int((i+1)*num_frames_per_segment)]
		mfccs = np.mean(librosa.feature.mfcc(y=sample, sr=vars.FRAME_RATE, n_mfcc=13), axis=0)
		features.append(mfccs)

	features = np.array(features)
	features =np.expand_dims(features, axis=2)

	return features


def dataset_creator(vars):
	data_path = vars.DATA_PATH
	train_test_split = vars.TRAIN_SPLIT
	all_data = listdir(data_path+'features/')
	num_train_samples = int(train_test_split * len(all_data))

	all_idxs = np.arange(0, len(all_data), dtype='int')
	train_idxs = np.random.choice(all_idxs, num_train_samples, replace=False)

	train_samples = [all_data[i] for i in train_idxs]
	test_samples = [all_data[i] for i in range(len(all_data)) if i not in train_idxs]

	with open('train_test_split.txt', 'w') as f:
		content = 'Train Samples ({})\n\n'.format(num_train_samples)
		content += '\n'.join(train_samples)
		content += '\n\n'
		content += 'Test Samples ({})\n\n'.format(len(all_data)-num_train_samples)
		content += '\n'.join(test_samples)
		f.write(content)

	for i in train_samples:
		shutil.move(data_path+'features/'+i, data_path+'train/'+i)

	for i in test_samples:
		shutil.move(data_path+'features/'+i, data_path+'test/'+i)


def load_target(vars, company, start_date):
	target = None
	target_path = vars.DATA_PATH+'targets/'+company+'/daily_prices.csv'
	try:
		all_data = pd.read_csv(target_path)
	except Exception as e:
		print('Error reading target : ', e)
		return target

	all_data['formatted_date'] = pd.to_datetime(all_data['Date'])
	all_data = all_data.sort_values(by='formatted_date', ascending=True)

	start_date = '{}-{}-{}'.format(start_date[:4], start_date[4:6], start_date[6:])

	all_dates = list(all_data['formatted_date'])
	all_dates = [str(i.date()) for i in all_dates]

	start_idx = all_dates.index(start_date)

	next_thirty_dates = all_data[start_idx:start_idx+30]

	target = np.array(list(next_thirty_dates['Adj Close']))

	if len(target) > 0 and len(target) < vars.NUM_DAYS_PRED:
		target = np.concatenate((target, np.zeros((vars.NUM_DAYS_PRED - len(target),))))

	return target


def get_data_for_volatiles(vars, company, call_date):
	target = None
	target_path = vars.DATA_PATH+'volatiles/'+company+'/daily_prices.csv'
	try:
		all_data = pd.read_csv(target_path)
	except Exception as e:
		print('Error reading target : ', e)
		return target

	all_data['formatted_date'] = pd.to_datetime(all_data['Date'])
	all_data = all_data.sort_values(by='formatted_date', ascending=True)

	start_date = '{}-{}-{}'.format(call_date[:4], call_date[4:6], call_date[6:])

	all_dates = list(all_data['formatted_date'])
	all_dates = [str(i.date()) for i in all_dates]

	start_idx = all_dates.index(start_date)

	volatile_data = {}

	all_periods = [3, 7, 15]

	for period in all_periods:
		past_data = list(all_data[start_idx-period-1:start_idx]['Adj Close'])
		next_data = list(all_data[start_idx:start_idx+period+1]['Adj Close'])
		current = list(all_data[start_idx:start_idx+1]['Adj Close'])

		volatile_data[period] = {
			'past': past_data,
			'next': next_data,
			'curr': current
		}

	return volatile_data


def prepare_data(vars):
	if exists(vars.TOKENIZER_PATH):
		with open(vars.TOKENIZER_PATH, 'rb') as f:
			tokenizer = pickle.load(f)
	else:
		tokenizer = text.Tokenizer(num_words=vars.VOCAB_SIZE)

		sentences = []

		for i in listdir(vars.DATA_PATH+'data/train/'):
			with open(vars.DATA_PATH+'data/train/'+i+'/Text.txt', 'r') as f:
				data = f.read()
				for sentence in data:
					sentences.append(sentence)

		tokenizer.fit_on_texts(sentences)

		with open(vars.TOKENIZER_PATH, 'wb') as f:
			pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

	return tokenizer


def get_coefs(word, *arr):
	return word, np.asarray(arr, dtype='float32')


def get_embeddings(vars, tokenizer, mode='new'):
	# Getting The File
	filePath = vars.EMBEDDING_FILE

	# Creating a Dictionary of format {word : Embedding}
	if mode == 'new':
		embeddings_idx = dict(get_coefs(*i.split(" ")) for i in open(filePath))
		# All Embeddings
		all_embs = np.stack(embeddings_idx.values())

		# Creating The Embedding Matrix with distribution, for if there is a missing word in the embeddings, it'll have
		# the embedding vector with the same distribution
		emb_mean,emb_std = all_embs.mean(), all_embs.std()
		embed_size = all_embs.shape[1]

		word_index = tokenizer.word_index
		nb_words = min(vars.VOCAB_SIZE, len(word_index))
		embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

		# Filling in the given learned embeddings in the embedding matrix
		for word, i in word_index.items():
			if i >= vars.VOCAB_SIZE: continue
			embedding_vector = embeddings_idx.get(word)
			if embedding_vector is not None: embedding_matrix[i] = embedding_vector

	return embeddings_idx, embedding_matrix