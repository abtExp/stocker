from ..models.base import BASE
from ..models.speaker_encoder import SPEAKER_ENCODER
from ..models.text_encoder import TEXT_ENCODER

from ..utils.data_loaders import get_embeddings

from keras.layers import Input, Dense, Dropout, CuDNNLSTM, Bidirectional, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from pase.models.frontend import wf_builder
import torch


class MODEL(BASE):
	def __init__(self, vars, model='assign', inp_shape=()):
		self.inp_shape = inp_shape
		embeddings = get_embeddings(vars)
		self.speaker_encoder = SPEAKER_ENCODER(vars).model
		self.text_encoder = TEXT_ENCODER(vars, embeddings).model

		super(MODEL, self).__init__(vars)

	def compose_model(self):
		speech = Input(shape=())
		text = Input(shape=())

		speech_encoding = self.speaker_encoder(speech)
		text_encoding = self.text_encoder(text)

		features = Concatenate()([speech_encoding, text_encoding])

		layer = Bidirectional(CuDNNLSTM(128, return_sequences=True))(layer)
		layer = Attention(self.vars.MAX_SENTENCE_LENGTH)(layer)
		layer = Dense(1024, activation='relu')(layer)

		layer = Dense(512, activation='relu')(layer)
		layer = Dropout(0.3)(layer)
		layer = Dense(256, activation='relu')(layer)
		layer = Dense(self.vars.NUM_DAYS_PRED)(layer)

		model = Model(inputs=[speech, text], outputs=layer)

		model.compile(loss='mean_squared_error', optimizer=Adam())

		return model