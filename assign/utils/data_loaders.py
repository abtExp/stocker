import librosa
import numpy as np
import pandas as pd

import shutil

from os import listdir

import keras.preprocessing.text as text
import keras.preprocessing.sequence as seq
from keras.utils import to_categorical

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split

from audio_module.utils.sound_utils import convert_to_wav, length_normalize


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
	target_path = vars.DATA_PATH+'targets/'+company+'/daily_prices.csv'
	all_data = pd.read_csv(target_path)

	all_data['formatted_date'] = pd.to_datetime(all_data['Date'])
	all_data = all_data.sort_values(by='formatted_date', ascending=True)

	start_date = '{}-{}-{}'.format(start_date[:4], start_date[4:6], start_date[6:])

	all_dates = list(all_data['formatted_date'])
	all_dates = [str(i.date()) for i in all_dates]

	start_idx = all_dates.index(start_date)

	next_thirty_dates = all_data[start_idx:start_idx+30]

	target = np.array(list(next_thirty_dates['Adj Close**']))

	return target


def prepare_data(vars):
	tokenizer = text.Tokenizer(num_words=vars.VOCAB_SIZE)

	sentences = []

	for i in listdir(vars.DATA_PATH+'data/train/'):
		with open(vars.DATA_PATH+'data/train/'+i+'/Text.txt', 'r') as f:
			data = f.read()
			for sentence in data:
				sentences.append(sentence)

	tokenizer.fit_on_texts(sentences)

	return tokenizer


def data_loader(vars, mode='train', comp_list=[]):
	data_folder = ''
	batch_size = 0

	if mode == 'train':
		data_folder = vars.DATA_PATH+'data/train/'
		batch_size = vars.TRAIN_BATCH_SIZE
	else:
		data_folder = vars.DATA_PATH+'data/test/'

	txts = []
	auds = []
	prices = []

	all_datas = listdir(data_folder)

	tokenizer = prepare_data(vars)

	idxs = np.random.choice(np.arange(0, len(all_datas)), batch_size, replace=False)

	for idx in idxs:
		company = all_datas[idx]
		start_date = company[:company.rindex('_')]

		# Loading Text Features
		text_features = []
		aud_features = []

		with open(data_folder+company+'/Text.txt') as f:
			sentences = f.read()
			sentences = sentences.split('\n')
			sentences = tokenizer.texts_to_sequences(sentences)
			features = seq.texts_to_sequences(sentences, maxlen=vars.MAX_SENTENCE_LENGTH)
			text_features.append(features)

		txts.append(text_features)

		# Loading Audio Features
		for i in listdir(data_folder+company+'/Audio/'):
			convert_to_wav(data_folder+company+'/Audio/'+i)
			aud, _ = librosa.load(data_folder+company+'/Audio/'+i[:i.rindex('.')]+'.wav')
			aud, _ = librosa.effects.trim(aud, top_db=20)
			aud = length_normalize(vars, aud)
			aud = np.abs(librosa.stft(aud, n_fft=vars.N_FFT))

			aud_features.append(aud)

		auds.append(aud_features)

		labels = load_target(vars, company, start_date)

		prices.append(labels)

	return np.array(txts), np.array(auds), np.array(prices)


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