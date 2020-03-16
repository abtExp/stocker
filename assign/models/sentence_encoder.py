from transformers import BertForSequenceClassification, BertTokenizer

import torch
from torch import nn
import numpy as np

from ..models import BASE


class ENCODER(nn.Module):
	def __init__(self):
		super(ENCODER, self).__init__()
		self.encoder = BertForSequenceClassification.from_pretrained('bert-base-uncased')
		self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
		self.encoder.eval().cuda()

	def forward(self, x):
		x = torch.from_numpy(x)
		x = x.type(torch.LongTensor)
		out = self.encoder(x.cuda())[1]
		out = out.detach().cpu().numpy()
		return out[0]

class SENTENCE_ENCODER(BASE):
	def __init__(self, vars):
		self.model_name = 'sentence_encoder'
		super(SENTENCE_ENCODER, self).__init__(vars)

	def compose_model(self):
		with torch.no_grad():
			model = ENCODER()

		return model

	def summary(self):
		print(self.model)

	def predict(self, x):
		return self.model.forward(x)