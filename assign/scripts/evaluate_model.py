from .calculate_v_past import calc_volatility, calc_mse
from ..settings import vars
from ..models import TEXT_ENCODER, SENTENCE_ENCODER, SPEECH_ENCODER, SPEAKER_ENCODER

from ..utils.data_feeder import data_loader

import math

from os import listdir

from os.path import exists

import numpy as np

import json

from transformers import BertTokenizer

import tensorflow as tf



def evaluate(vars, eval_model='text_model', weights_path=''):
	tokenizer = None
	text_encoder = None
	audio_encoder = None
	if eval_model == 'text_model' or eval_model == 'both_model':
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		text_encoder = SENTENCE_ENCODER(vars)
		model = TEXT_ENCODER(vars)
	if eval_model == 'audio_model' or eval_model == 'both_model':
		audio_encoder = SPEAKER_ENCODER(vars, graph=tf.get_default_graph())
		model = SPEECH_ENCODER(vars)

	# vars.CHECK_PATH+eval_model+'/'+listdir(vars.CHECK_PATH+eval_model)[-1]
	model.load_weights(weights_path)

	data_folder = vars.DATA_PATH+'data__/test/'

	all_datas = {}

	for folder in listdir(data_folder):
			tx, ty, _ = data_loader(vars, mode='test', folder=folder, encoder=audio_encoder, tokenizer=tokenizer, model=text_encoder, load_mode=eval_model[:eval_model.rindex('_')])

			if len(tx.shape) > 1:
				print('Evaluating...')
				preds = model.predict(tx)[0]
				ty = ty[0]

				periods = [3, 7, 15, 30]

				for period in periods:
					orig = ty[:period]
					pred = preds[:period]
					original_volatility = calc_volatility(orig)
					predicted_volatility = calc_volatility(pred)

					print('Original Volatility : {}, Predicted Volatility : {}'.format(original_volatility, predicted_volatility))

					mse = calc_mse(original_volatility, predicted_volatility)

					print('MSE : {}'.format(mse))

					if not math.isnan(mse) and not math.isinf(mse):
						if period not in all_datas.keys():
							all_datas[period] = [mse]
						else:
							all_datas[period].append(mse)

	for period in all_datas.keys():
		print('Average MSE of Past Volatilities For Period : {} days = {}'.format(period, np.mean(all_datas[period])))