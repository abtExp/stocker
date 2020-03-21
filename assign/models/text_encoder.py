from ..models.base import BASE#, SENTENCE_ENCODER
from ..models.layers import Attention, REMOVE_DIM, EXPAND_DIM

from ..utils.data_loader import DATA_LOADER
from ..utils.data_feeder import data_loader

from keras.layers import Input, Layer, Dense, Embedding, CuDNNLSTM,\
	Bidirectional, Dropout, LSTM, Flatten, GlobalAveragePooling1D

from keras.models import Model

import keras.backend as K
from keras import initializers

from keras.optimizers import Adam

import tensorflow as tf

from transformers import BertModel


class TEXT_ENCODER(BASE):
	def __init__(self, vars):
		self.model_name = 'txt_enc'
		super(TEXT_ENCODER, self).__init__(vars)

		self.graph = tf.get_default_graph()
		self.load_mode = 'text'

		self.DATA_LOADER = DATA_LOADER
		self.data_feeder = data_loader
		#self.encoder = SENTENCE_ENCODER(self.vars)

	def compose_model(self):
		inp = Input(shape=(self.vars.MAX_SENTENCES, self.vars.EMBEDDING_SIZE))
		# layer = REMOVE_DIM()(inp)
		layer = Bidirectional(CuDNNLSTM(units=512, return_sequences=True))(inp)
		layer = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(layer)
		layer = GlobalAveragePooling1D()(layer)
		layer = Dense(256, activation='relu')(layer)
		layer = Dropout(0.3)(layer)
		layer = Dense(self.vars.NUM_DAYS_PRED)(layer)

		model = Model(inputs=inp, outputs=layer)

		model.compile(loss='mse', optimizer=Adam())

		return model

	def predict(self, x):
		return self.model.predict(x)