import keras.backend as K
from keras import initializers

from keras.layers import Layer
import numpy as np

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


class EXPAND_DIM(Layer):
	def __init__(self):
		super(EXPAND_DIM, self).__init__()

	def call(self, x):
		out = K.expand_dims(x, axis=0)
		out._keras_shape = self.compute_output_shape(out.shape)
		return out

	def compute_output_shape(self, input_shape):
		return input_shape

class REMOVE_DIM(Layer):
	def __init__(self):
		super(REMOVE_DIM, self).__init__()

	def call(self, x):
		out = x[0]
		out._keras_shape = self.compute_output_shape(x.shape)
		return out

	def compute_output_shape(self, input_shape):
		return input_shape[1:]


class MERGE(Layer):
	def __init__(self):
		super(MERGE, self).__init__()

	def call(self, x):
		l1 = x[0]
		l2 = x[1]

		l1 = K.expand_dims(l1, axis=0)
		l2 = K.expand_dims(l2, axis=0)

		final = K.concatenate([l1, l2], axis=0)

		# final._keras_shape = self.compute_output_shape(l1.shape)
		return final

	# def compute_output_shape(self, input_shape):
