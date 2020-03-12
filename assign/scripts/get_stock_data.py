import numpy as np

import pandas as pd

from datetime import datetime

import time

from os import listdir, mkdir
from os.path import isdir

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

def get_companies_dates(vars):
	all_comps = listdir(vars.DATA_PATH+'ftrs/')

	url = vars.YAHOO_DOWNLOAD_FINLINK

	links = []

	comp_info = pd.read_csv(vars.DATA_PATH+'comp_codes.csv')

	comp_date_dict = {}
	for i in all_comps:
		comp_name = i[:i.rindex('_')]
		date = i[i.rindex('_')+1:]
		if comp_name not in comp_date_dict.keys():
			comp_date_dict[comp_name] = []

		comp_date_dict[comp_name].append(date)

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
					# print('{} -> {} -> {}'.format(comp, i[0], i[1]))
					comp_code = i[1]
					break

		if len(comp_code) > 0:
			dates = comp_date_dict[comp]
			start_date = '{}-{}-{}'.format(dates[0][:4], dates[0][4:6], dates[0][6:])
			end_date = '{}-{}-{}'.format(dates[-1][:4], dates[-1][4:6], dates[-1][6:])

			start_date = str(int(time.mktime(pd.to_datetime(start_date).timetuple())))
			end_date = int(time.mktime(pd.to_datetime(end_date).timetuple()))
			end_date = end_date + int(2678400)
			end_date = str(end_date)

			data = download_csv(vars, comp_code, start_date, end_date)

			if not isdir(vars.DATA_PATH+'targets/'+comp):
				mkdir(vars.DATA_PATH+'targets/'+comp)

			with open(vars.DATA_PATH+'targets/'+comp+'/daily_prices.csv', 'w') as f:
				if type(data) != str:
					f.write(data.decode('utf-8'))

		else:
			print('#### Couldn\'t Find {}'.format(comp))


def get_data(vars):
	target_path = vars.DATA_PATH+'targets/'

	with open(vars.DATA_PATH+'stock_links.txt', 'r') as f:
		links = f.read()
		links = links.split('\n')

		for link in links:
			comp_name, stock_link = link.split(' -> ')
			if isdir(vars.DATA_PATH+'targets/'+comp_name):
				continue
			else:
				mkdir(vars.DATA_PATH+'targets/'+comp_name)
				data = requests.get(stock_link)
