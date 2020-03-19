from ..models.base import BASE
from ..models.speech_encoder import SPEECH_ENCODER
from ..models.text_encoder import TEXT_ENCODER
from ..models.layers import Attention, MERGE, REMOVE_DIM

from ..utils.data_utils import get_embeddings, prepare_data
from ..utils.data_loader import DATA_LOADER
from ..utils.data_feeder import data_loader

from tensorflow.keras.layers import Input, Dense, Dropout, CuDNNLSTM, Bidirectional,\
	Concatenate, Reshape, GlobalAveragePooling2D, ConvLSTM2D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import keras

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(sess)

class MODEL(BASE):
	def __init__(self, vars, model='assign'):
		self.model_name = model
		self.inp_shape = (vars.MAX_SENTENCES,)
		# self.speech_encoder = SPEECH_ENCODER(vars).model
		# self.text_encoder = TEXT_ENCODER(vars).model
		self.load_mode = 'both'

		super(MODEL, self).__init__(vars)

		self.DATA_LOADER = DATA_LOADER
		self.data_feeder = data_loader

		self.graph = tf.get_default_graph()

	# def temporal_encoder_block()


	def compose_model(self):
		text = Input(shape=(self.vars.MAX_SENTENCES, self.vars.EMBEDDING_SIZE))
		speech = Input(shape=(self.vars.MAX_SENTENCES, self.vars.NUM_SEGMENTS_PER_AUDIO, self.vars.AUDIO_EMBEDDING_SIZE, 1))

		# text_encoding
		text_encoding = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(text)
		text_encoding = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(text_encoding)
		text_encoding = GlobalAveragePooling1D()(text_encoding)
		text_encoding = Dense(256, activation='relu')(text_encoding)


		# text_encoding = REMOVE_DIM()(text)
		# text_encoding = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(text_encoding)
		# # text_encoding = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(text_encoding)
		# text_encoding = Attention(self.vars.MAX_SENTENCE_LENGTH)(text_encoding)
		# text_encoding = Dense(512, activation='relu')(text_encoding)

		# speech_encoding
		speech_encoding = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', return_sequences=True)(speech)
		speech_encoding = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', return_sequences=True)(speech_encoding)
		speech_encoding = ConvLSTM2D(filters=128, kernel_size=(1, 1), activation='relu')(speech_encoding)
		speech_encoding = GlobalAveragePooling2D()(speech_encoding)
		speech_encoding = Dense(256, activation='relu')(speech_encoding)


		# speech_encoding = REMOVE_DIM()(speech)
		# speech_encoding = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(speech_encoding)
		# # speech_encoding = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(speech_encoding)
		# speech_encoding = Attention(self.vars.NUM_SEGMENTS_PER_AUDIO)(speech_encoding)
		# speech_encoding = Dense(512, activation='relu')(speech_encoding)

		features = MERGE()([speech_encoding, text_encoding])

		layer = Bidirectional(CuDNNLSTM(128, return_sequences=True))(features)
		layer = GlobalAveragePooling1D()(layer)
		layer = Dense(256, activation='relu')(layer)
		# layer = Dropout(0.3)(layer)
		layer = Dense(self.vars.NUM_DAYS_PRED)(layer)

		model = Model(inputs=[text, speech], outputs=layer)

		model.compile(loss='mean_squared_error', optimizer=Adam())

		return model

	def tf_model(self):
		self.model = tf.keras.estimator.model_to_estimator(keras_model=self.model, model_dir=self.vars.CHECK_PATH)

	def tf_init_loaders(self):
		return

	def tf_train_and_eval(self):
		return

	def tf_predict(self, x):
		return