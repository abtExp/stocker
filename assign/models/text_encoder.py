from ..models.base import BASE
from ..models.layers import Attention, REMOVE_DIM, EXPAND_DIM

from ..utils.data_loader import DATA_LOADER
from ..utils.data_feeder import data_loader

from keras.layers import Input, Layer, Dense, Embedding, CuDNNLSTM,\
	Bidirectional, Dropout, LSTM, Flatten

from keras.models import Model

import keras.backend as K
from keras import initializers


class TEXT_ENCODER(BASE):
	def __init__(self, vars):
		self.model_name = 'txt_enc'
		super(TEXT_ENCODER, self).__init__(vars)

		self.DATA_LOADER = DATA_LOADER
		self.data_feeder = data_loader

	def compose_model(self):
		inp = Input(batch_shape=(1, self.vars.MAX_SENTENCES, self.vars.MAX_SENTENCE_LENGTH, self.vars.EMBEDDING_SIZE))
		layer = REMOVE_DIM()(inp)
		layer = Bidirectional(CuDNNLSTM(units=512, return_sequences=True))(layer)
		layer = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(layer)
		layer = Attention(self.vars.MAX_SENTENCE_LENGTH)(layer)
		layer = EXPAND_DIM()(layer)
		layer = Flatten()(layer)
		layer = Dense(256, activation='relu')(layer)
		layer = Dropout(0.3)(layer)
		layer = Dense(self.vars.NUM_DAYS_PRED)(layer)

		model = Model(inputs=inp, outputs=layer)

		return model

	def predict(self, x):
		return self.model.predict(x)