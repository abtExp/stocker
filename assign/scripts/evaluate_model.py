from .calculate_v_past import calc_volatility, calc_mse
from ..settings import vars
from ..models import TEXT_ENCODER, SENTENCE_ENCODER

from ..utils.data_utils import load_target

import math

from os import listdir

from os.path import exists

import numpy as np

import json

from transformers import BertTokenizer

def evaluation_loader(vars, folder, tokenizer, encoder):
	data_folder = vars.PROJECT_PATH+'data/text_data/test/'

	company, start_date = folder.split('_')

	aud_features = []

	auds = []
	txts = []
	prices = []

	# Loading Text Features
	with open(data_folder+folder+'/Text.txt') as f:
		print('READING : {}'.format(data_folder+folder+'/Text.txt'))
		try:
			sentences = f.read()
		except Exception as e:
			print('Can\'t read! : ', e)
			return np.array(txts), np.array(prices)
		sentences = sentences.split('\n')
		sentences = sentences[:vars.MAX_SENTENCES]
		features = []
		for sentence in sentences:
			tokens = tokenizer.encode(sentence, max_length=vars.MAX_SENTENCE_LENGTH, pad_to_max_length=True)
			# tokens = torch.tensor([tokens])
			outputs = encoder.predict(np.array([tokens], dtype=np.long))
			# return
			features.append(outputs)

	features = np.array(features)

	if len(features) < vars.MAX_SENTENCES:
		features = np.concatenate((features, np.zeros((vars.MAX_SENTENCES-len(features), *np.shape(features[0])))))


	# Loading Audio Features
	# cntr = 0
	# for i in listdir(data_folder+folder+'/Audio/'):
	# 	aud = load_audio(vars, data_folder+folder+'/Audio/'+i)
	# 	aud = encoder.predict(aud)
	# 	aud_features.append(aud)
	# 	cntr += 1
	# 	if cntr >= vars.MAX_SENTENCES:
	# 		break

	# aud_features = np.array(aud_features)

	# if len(aud_features) > 0:
	# 	if len(aud_features) < vars.MAX_SENTENCES:
	# 		aud_features = np.concatenate((aud_features, np.zeros((vars.MAX_SENTENCES-len(aud_features), *np.shape(aud_features[0])))))


	labels = load_target(vars, company, start_date)

	if type(labels) == np.ndarray:
		# auds.append(aud_features)
		txts.append(features)
		prices.append(labels)

	# return [np.array(txts), np.array(auds)], np.array(prices)
	return np.array(txts), np.array(prices)


def evaluate(vars):
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	encoder = SENTENCE_ENCODER(vars)

	model = TEXT_ENCODER(vars)
	model.load_weights(vars.PROJECT_PATH+'assign/checkpoints/checkpoints_custom_25/weights.02-1450.79.hdf5')

	data_folder = vars.PROJECT_PATH+'data/text_data/test/'

	all_datas = {}

	for folder in listdir(data_folder):
		if exists(data_folder+folder+'/Text.txt'):
			tx, ty = evaluation_loader(vars, folder, tokenizer, encoder)

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

	print(all_datas)

	with open('./eval.json', 'w') as f:
		json.dump(all_datas, f)

	for period in all_datas.keys():
		print('Average MSE of Past Volatilities For Period : {} days = {}'.format(period, np.mean(all_datas[period])))