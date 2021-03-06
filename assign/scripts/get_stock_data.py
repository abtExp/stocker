import numpy as np

import pandas as pd

from datetime import datetime

import time

from os import listdir, mkdir
from os.path import isdir, exists

import json

import requests

import urllib

import re


def get_cookie_crumble(lnk):
	cookie_str = ''
	crumble_str = ''

	cookie_regex = r'Set-Cookie: (.*?); '
	crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'

	response = urllib.request.urlopen(lnk)

	match = re.search(cookie_regex, str(response.info()))
	cookie_str = match.group(1)

	text = str(response.read())
	match = re.search(crumble_regex, text)

	if match is not None:
		crumble_str = match.group(1)

	return cookie_str, crumble_str


def download_csv(vars, comp_code, start_date, end_date):
	vis_url = vars.YAHOO_FINLINK.format(comp_code, comp_code)

	attempts = 0
	while attempts < 5:
		cookie, crumble = get_cookie_crumble(vis_url)
		if len(cookie) > 0 and len(crumble) > 0:
			final_url = vars.YAHOO_DOWNLOAD_FINLINK.format(comp_code, start_date, end_date, crumble)
			req = urllib.request.Request(final_url)
			req.add_header('Cookie', cookie)

			try:
				response = urllib.request.urlopen(req)
				text = response.read()
				print("{} downloaded".format(comp_code))
				return text
			except Exception as e:
				print("{} failed at attempt # {}".format(comp_code, attempts))
				attempts += 1
				time.sleep(2*attempts)
		else:
			attempts += 1

	return b''




def get_companies_dates(vars, comp_code='', start_date='', end_date=''):
	comp_info = pd.read_csv(vars.DATA_PATH+'comp_codes.csv')

	# comp_date_dict
	with open('D:/stocker/comp_with_dates.json', 'r', encoding='utf-8') as f:
		comp_date_dict = json.load(f)

	comps = np.array(comp_info[['Name', 'Ticker']])

	for comp in comp_date_dict.keys():
		name = comp.replace('.', '').replace(',', '').lower()
		comp_code = ''
		for i in comps:
			if type(i) == np.ndarray and type(i[0]) == str:
				i[0] = i[0].replace('.', '').replace(',', '').lower()
				if '"' in i[0]:
					i[0] = i[0].replace('"', "")
				if name in i[0]:
					comp_code = i[1]
					break

		if len(comp_code) > 0:
			dates = comp_date_dict[comp]
			start_date = '{}-{}-{}'.format(dates[0][:4], dates[0][4:6], dates[0][6:])
			end_date = '{}-{}-{}'.format(dates[-1][:4], dates[-1][4:6], dates[-1][6:])

			start_date = str(int(time.mktime(pd.to_datetime(start_date).timetuple())))
			start_date = str(int(start_date) - int(2678400))
			end_date = time.mktime(pd.to_datetime(end_date).timetuple())
			end_date = str(int(end_date) + int(2678400))

			data = download_csv(vars, comp_code, start_date, end_date)

			if not isdir(vars.DATA_PATH+'volatiles/'+comp):
				mkdir(vars.DATA_PATH+'volatiles/'+comp)

			with open(vars.DATA_PATH+'volatiles/'+comp+'/daily_prices.csv', 'w') as f:
				if type(data) != str:
					f.write(data.decode('utf-8'))

		else:
			print('#### Couldn\'t Find {}'.format(comp))


def get_stocks(vars):
	with open('D:/stocker/missing_targets.txt', 'r') as f:
		data = f.read()
		lines = data.split('\n')
		for line in lines:
			comp_code, start_date, end_date = line.split(' ')
			start_date = '{}-{}-{}'.format(start_date[:4], start_date[4:6], start_date[6:])
			end_date = '{}-{}-{}'.format(end_date[:4], end_date[4:6], end_date[6:])

			start_date = str(int(time.mktime(pd.to_datetime(start_date).timetuple())))
			start_date = str(int(start_date) - int(2678400))
			end_date = time.mktime(pd.to_datetime(end_date).timetuple())
			end_date = str(int(end_date) + int(2678400))

			data = download_csv(vars, comp_code, start_date, end_date)

			if not exists('D:/stocker/data/targets/'+comp_code):
				mkdir('D:/stocker/data/targets/'+comp_code)

			with open('D:/stocker/data/targets/'+comp_code+'/daily_prices.csv', 'w') as f:
				f.write(data.decode('utf-8'))