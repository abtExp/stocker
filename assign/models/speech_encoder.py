from ..models.base import BASE
from ..models.layers import Attention

from keras.layers import Input, Layer, Dense, Embedding, CuDNNLSTM,\
	Bidirectional, Dropout, LSTM, ConvLSTM2D

from keras.models import Model

import keras.backend as K
from keras import initializers


class SPEECH_ENCODER(BASE):
	def __init__(self, vars):
		self.model_name = 'speech_encoder'
		super(SPEECH_ENCODER, self).__init__(vars)

	def compose_model(self):
		inp = Input(shape=(self.vars.MAX_SENTENCES, self.vars.NUM_SEGMENTS_PER_AUDIO, self.vars.AUDIO_EMBEDDING_SIZE, 1))
		layer = Bidirectional(CuDNNLSTM(units=1024, return_sequences=True))(inp)
		layer = Bidirectional(CuDNNLSTM(units=512, return_sequences=True))(layer)
		layer = Attention(self.vars.NUM_SEGMENTS_PER_AUDIO)(layer)
		layer = Dense(768, activation='relu')(layer)

		model = Model(inputs=inp, outputs=layer)

		return model

	def predict(self, x):
		return self.model.predict(x)