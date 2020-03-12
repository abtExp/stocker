from ..models.base import BASE

from pase.models.frontend import wf_builder
from keras.models import load_model, Model
from keras.layers import Input

class SPEAKER_ENCODER(BASE):
	def __init__(self, vars):
		self.model_name = 'speaker_encoder'
		super(SPEAKER_ENCODER, self).__init__(vars)

	def compose_model(self):
		# model = wf_builder(self.vars.PROJECT_PATH+'cfg/PASE+.cfg').eval()
		# model.load_pretrained(self.vars.PROJECT_PATH+'checkpoints/FE_e199.ckpt', load_last=True, verbose=True)
		# model.cuda()

		# return model

		input = Input(shape=(216, 1))

		model = load_model(self.vars.PROJECT_PATH+'checkpoints/speaker_encoder.h5')
		model = Model(inputs=model.input, output=model.layers[-3].output)


		# inp = Input(shape=(216, 1))
		# x = Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu')(inp)
		# x = Conv1D(filters=128, kernel_size=(5, 5), padding='same', activation='relu')(x)
		# x = Dropout(0.1)(x)
		# x = MaxPooling1D(pool_size=(8, 8))(x)
		# x = Conv1D(filters=128, kernel_size=(5, 5), padding='same', activation='relu')(x)
		# x = Conv1D(filters=128, kernel_size=(5, 5), padding='same', activation='relu')(x)
		# x = Flatten()(x)

		return model

	def predict(self, x):
		return self.model(x)