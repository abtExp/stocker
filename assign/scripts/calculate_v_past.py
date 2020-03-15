from os import listdir
import pandas as pd
import numpy as np
import json

from ..utils.data_utils import get_data_for_volatiles


def calc_volatility(dates):
	rts = []
	for i in range(1, len(dates)):
		rts.append((dates[i]/dates[i-1])-1)

	mean = np.mean(rts)

	volatility = np.log(np.sqrt((np.sum((rts-mean)**2))/len(rts)))

	return volatility


def calc_mse(x, y):
	return np.mean((x-y)**2)


def calc_v_past(vars):
	vpast_comps = {}
	with open('D:/iiit_assign/comp_with_dates.json', 'r') as f:
		all_comps = json.load(f)

	for comp in all_comps.keys():
		calls = {}
		for call in all_comps[comp]:
			data = get_data_for_volatiles(vars, comp.replace(' ', ''), call)
			if data is not None:
				with open('D:/iiit_assign/datas/{}_{}.json'.format(comp.replace(' ', ''), call), 'w') as f:
					json.dump(data, f)
				days = {}
				for period in data.keys():
					past = data[period]['past']
					nxxt = data[period]['next']
					past_volatility = calc_volatility(past)
					next_volatility = calc_volatility(nxxt)

					vpast = calc_mse(past_volatility, next_volatility)
					days[period] = vpast

				calls[call] = days
			vpast_comps[comp] = calls

	with open('./vpast.json', 'w') as f:
		json.dump(vpast_comps, f)