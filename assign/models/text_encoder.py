from ..models.base import BASE

from keras.layers import Input, Layer, Dense, Embedding, CuDNNLSTM,\
	Bidirectional, Dropout

from keras.models import Model

import keras.backend as K
from keras import initializers


class Attention(Layer):
	def __init__(self, step_dim, **kwargs):
		self.supports_masking = True
		self.init = initializers.get('glorot_uniform')
		self.features_dim = 0
		self.step_dim = step_dim
		self.bias = True
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3

		self.W = self.add_weight((input_shape[-1],),
								initializer=self.init,
								name='{}_W'.format(self.name))

		self.features_dim = input_shape[-1]

		if self.bias:
			self.b = self.add_weight((input_shape[1],),
									initializer='zero',
									name='{}_b'.format(self.name),
									)
		else:
			self.b = None

		self.built = True

	def call(self, x, mask=None):
		features_dim = self.features_dim
		step_dim = self.step_dim

		eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
						K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

		if self.bias:
			eij += self.b

		eij = K.tanh(eij)

		a = K.exp(eij)

		if mask is not None:
			a *= K.cast(mask, K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

		a = K.expand_dims(a)
		weighted_input = x * a
		return K.sum(weighted_input, axis=1)

	def compute_output_shape(self, input_shape):
		return input_shape[0],  self.features_dim

class TEXT_ENCODER(BASE):
	def __init__(self, vars, embedding_mtx):
		self.model_name = 'txt_enc'
		self.embedding_mtx = embedding_mtx
		super(TEXT_ENCODER, self).__init__(vars)

	def compose_model(self):
		inp = Input(shape=(self.vars.MAX_SENTENCE_LENGTH,))
		layer = Embedding(self.vars.VOCAB_SIZE, self.vars.EMBEDDING_SIZE, weights=[self.embedding_mtx], trainable=False)(inp)
		layer = Bidirectional(CuDNNLSTM(128, return_sequences=True))(layer)
		layer = Bidirectional(CuDNNLSTM(64, return_sequences=True))(layer)
		layer = Attention(self.vars.MAX_SENTENCE_LENGTH)(layer)
		layer = Dense(256, activation='relu')(layer)

		model = Model(inputs=inp, outputs=layer)

		return model

	def predict(self, x):
		return