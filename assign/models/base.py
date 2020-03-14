from keras.utils import plot_model
import tensorflow as tf
import keras.backend as K

class BASE():
	def __init__(self, vars):
		self.vars = vars
		self.vars.MODEL_IMAGE_PATH += self.model_name+'.png'
		self.model = self.compose_model()

	def train(self):
		self.init_loaders()
		self.model.fit_generator(self.train_loader, validation_data=self.valid_loader, epochs=self.vars.TRAIN_EPOCHS, callbacks=self.vars.get_callbacks(self.model_name))

	def save(self, path):
		self.model.save(path)

	def load_weights(self, pth):
		if pth:
			self.vars.BEST_WEIGHT_PATH = pth
		self.model.load_weights(self.vars.BEST_WEIGHT_PATH)

	def init_loaders(self):
		self.train_loader = self.DATA_LOADER(self.vars, 'train', loader=self.data_feeder, graph=self.graph)
		self.valid_loader = self.DATA_LOADER(self.vars, 'valid', loader=self.data_feeder, graph=self.graph)

	def summary(self):
		self.model.summary()

	def plot(self):
		plot_model(self.model, self.vars.MODEL_IMAGE_PATH, show_shapes=True)

	def predict(self, data):
		# with tf.Session().as_default():
		with self.graph.as_default():
			self.load_weights(self.vars.BEST_WEIGHT_PATH)
			self.model._make_predict_function()
			return self.model.predict(data)[0]

	def predict_on_batch(self):
		batch = self.vars.TEST_LOADER(self.vars.TEST_BATCH_SIZE, 'test')
		return self.model.predict_on_batch(batch)

	def compose_model(self):
		return

	def process(self, path, *args):
		return

	def train(self):
		self.init_loaders()
		self.model.fit_generator(self.train_loader, validation_data=self.valid_loader, epochs=self.vars.TRAIN_EPOCHS, callbacks=self.vars.get_callbacks(), steps_per_epoch=self.vars.STEPS_PER_EPOCH)