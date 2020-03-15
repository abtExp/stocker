from ..models.base import BASE

# from pase.models.frontend import wf_builder
from keras.models import load_model, Model
from keras.layers import Input

class SPEAKER_ENCODER(BASE):
	def __init__(self, vars, graph=None):
		self.model_name = 'speaker_encoder'
		self.graph = graph
		super(SPEAKER_ENCODER, self).__init__(vars)

	def compose_model(self):
		model = load_model(self.vars.PROJECT_PATH+'assign/checkpoints/speaker_encoder.h5')

		with self.graph.as_default():
			model = Model(inputs=model.input, output=model.layers[-3].output)

		return model

	def predict(self, x):
		with self.graph.as_default():
			return self.model.predict(x)