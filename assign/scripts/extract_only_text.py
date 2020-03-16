from os import listdir, mkdir
from os.path import isdir
from shutil import copyfile

def extract():
	for i in listdir('D:/iiit_assign/data/data/'):
		for folder in listdir('D:/iiit_assign/data/data/'+i):
			if not isdir('D:/iiit_assign/data/text_data/'+i+'/'+folder):
				mkdir('D:/iiit_assign/data/text_data/'+i+'/'+folder)
				copyfile('D:/iiit_assign/data/data/'+i+'/'+folder+'/Text.txt', 'D:/iiit_assign/data/text_data/'+i+'/'+folder+'/Text.txt')


extract()