from ..settings import vars
from ..utils.data_feeder import data_loader
from ..models import SENTENCE_ENCODER, SPEAKER_ENCODER

from transformers import BertTokenizer

from os import listdir

import tensorflow as tf

def find_inconsistencies():
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	text_encoder = SENTENCE_ENCODER(vars)
	audio_encoder = SPEAKER_ENCODER(vars, graph=tf.get_default_graph())

	all_errors = ''

	for i in listdir(vars.DATA_PATH+'data/'):
		for j in listdir(vars.DATA_PATH+'data/'+i):
			_, _, err = data_loader(
				vars, mode='test',
				folder='/'+i+'/'+j,
				encoder=audio_encoder,
				tokenizer=tokenizer,
				model=text_encoder, load_mode='both')

			if len(err) > 0:
				all_errors += '{} : {}\n'.format(i+'/'+j, err)

	with open('inconsistencies.txt', 'w') as f:
		f.write(all_errors)