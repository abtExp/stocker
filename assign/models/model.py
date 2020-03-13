from ..models.base import BASE
from ..models.speech_encoder import SPEECH_ENCODER
from ..models.text_encoder import TEXT_ENCODER
from ..models.layers import Attention, MERGE

from ..utils.data_utils import get_embeddings, prepare_data
from ..utils.data_loader import DATA_LOADER
from ..utils.data_feeder import data_loader

from keras.layers import Input, Dense, Dropout, CuDNNLSTM, Bidirectional, Concatenate, Reshape, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam


class MODEL(BASE):
	def __init__(self, vars, model='assign', inp_shape=()):
		self.model_name = model
		self.inp_shape = inp_shape
		self.speech_encoder = SPEECH_ENCODER(vars).model
		self.text_encoder = TEXT_ENCODER(vars).model

		super(MODEL, self).__init__(vars)

		self.DATA_LOADER = DATA_LOADER
		self.data_feeder = data_loader

	# def temporal_encoder_block()


	def compose_model(self):
		speech = Input(batch_shape=(self.inp_shape[0], self.vars.NUM_SEGMENTS_PER_AUDIO, self.vars.AUDIO_EMBEDDING_SIZE))
		text = Input(batch_shape=(self.inp_shape[0], self.vars.MAX_SENTENCE_LENGTH, self.vars.EMBEDDING_SIZE))

		# speech_encoding
		speech_encoding = Bidirectional(CuDNNLSTM(units=1024, return_sequences=True))(speech)
		speech_encoding = Bidirectional(CuDNNLSTM(units=512, return_sequences=True))(speech_encoding)
		speech_encoding = Attention(self.vars.NUM_SEGMENTS_PER_AUDIO)(speech_encoding)
		speech_encoding = Dense(768, activation='relu')(speech_encoding)

		# text_encoding
		text_encoding = Bidirectional(CuDNNLSTM(units=1024, return_sequences=True))(text)
		text_encoding = Bidirectional(CuDNNLSTM(units=512, return_sequences=True))(text_encoding)
		text_encoding = Attention(self.vars.MAX_SENTENCE_LENGTH)(text_encoding)
		text_encoding = Dense(768, activation='relu')(text_encoding)

		features = MERGE()([speech_encoding, text_encoding])

		layer = Bidirectional(CuDNNLSTM(512, return_sequences=True))(features)
		layer = Bidirectional(CuDNNLSTM(512, return_sequences=True))(features)
		layer = GlobalAveragePooling1D()(layer)
		layer = Dense(1024, activation='relu')(layer)

		layer = Dense(512, activation='relu')(layer)
		layer = Dropout(0.3)(layer)
		layer = Dense(256, activation='relu')(layer)
		layer = Dense(self.vars.NUM_DAYS_PRED)(layer)

		model = Model(inputs=[speech, text], outputs=layer)

		model.compile(loss='mean_squared_error', optimizer=Adam())

		return model