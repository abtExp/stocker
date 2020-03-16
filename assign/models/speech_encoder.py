from ..models.base import BASE
from ..models.layers import Attention

from keras.layers import Input, Layer, Dense, Embedding, CuDNNLSTM,\
	Bidirectional, Dropout, LSTM, ConvLSTM2D, GlobalAveragePooling2D

from keras.optimizers import Adam

from keras.models import Model

import keras.backend as K
from keras import initializers

from ..utils.data_feeder import data_loader
from ..utils.data_loader import DATA_LOADER

import tensorflow as tf


class SPEECH_ENCODER(BASE):
	def __init__(self, vars):
		self.load_mode = 'audio'
		self.model_name = 'speech_encoder'
		self.DATA_LOADER = DATA_LOADER
		self.data_feeder = data_loader
		self.graph = tf.get_default_graph()
		super(SPEECH_ENCODER, self).__init__(vars)

	def compose_model(self):
		inp = Input(batch_shape=(1, self.vars.MAX_SENTENCES, self.vars.NUM_SEGMENTS_PER_AUDIO, self.vars.AUDIO_EMBEDDING_SIZE, 1))
		layer = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', return_sequences=True)(inp)
		layer = ConvLSTM2D(filters=128, kernel_size=(3, 3), activation='relu', return_sequences=True)(layer)
		layer = ConvLSTM2D(filters=256, kernel_size=(1, 1), activation='relu')(layer)
		layer = GlobalAveragePooling2D()(layer)
		# layer = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(layer)
		# layer = Attention(self.vars.MAX_SENTENCE_LENGTH)(layer)
		layer = Dense(256, activation='relu')(layer)
		layer = Dropout(0.3)(layer)
		layer = Dense(self.vars.NUM_DAYS_PRED)(layer)

		model = Model(inputs=inp, outputs=layer)
		model.compile(loss='mse', optimizer=Adam())

		return model

	def predict(self, x):
		return self.model.predict(x)