from transformers import BertModel, BertTokenizer

from keras.utils import Sequence

from ..utils.data_feeder import data_loader
from ..models.speaker_encoder import SPEAKER_ENCODER
from ..models.sentence_encoder import SENTENCE_ENCODER

class DATA_LOADER(Sequence):
	def __init__(self, vars, mode='train', loader=None, graph=None, load_mode='text'):
		self.vars = vars
		self.mode = mode
		self.graph = graph
		self.loader = loader
		self.load_mode = load_mode
		self.tokenizer = None
		self.text_encoder = None
		self.speech_encoder = None

		if self.load_mode == 'text' or self.load_mode == 'both':
			self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
			self.text_encoder = SENTENCE_ENCODER(vars)
		if self.load_mode == 'audio' or self.load_mode == 'both':
			self.speech_encoder = SPEAKER_ENCODER(self.vars, graph=self.graph)

	def __getitem__(self, index):
		x, y, _ = self.__data_generation([])
		return x, y

	def __len__(self):
		return 1

	def __data_generation(self, l):
		return self.loader(self.vars, self.mode, encoder=self.speech_encoder, tokenizer=self.tokenizer, model=self.text_encoder, load_mode=self.load_mode)