import os
from ..utils.speaker_encoder_preprocess import convert_to_wav

def safe_name(vars):
	data_path = vars.DATA_PATH+'data'
	for dir in os.listdir(data_path):
		for folder in os.listdir(data_path+'/'+dir):
			print('{}/{}'.format(dir, folder))
			for file in os.listdir(data_path+'/'+dir+'/'+folder+'/Audio'):
				# os.rename(data_path+'/'+dir+'/'+folder+'/Audio/'+file, data_path+'/'+dir+'/'+folder+'/Audio/'+file.replace(' ', ''))
				ext = file[file.rindex('.')+1:]
				if ext == 'mp3':
					print(file)
					print('Converting...')
					convert_to_wav(data_path+'/'+dir+'/'+folder+'/Audio/'+file)
					os.remove(data_path+'/'+dir+'/'+folder+'/Audio/'+file[:file.rindex('.')]+'.mp3')

