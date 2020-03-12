import numpy as np
import librosa

import subprocess

def convert_to_wav(input_file):
	output_file = input_file[:input_file.rindex('.')]+'.wav'
	command = 'ffmpeg -y -i {} -acodec pcm_s16le -ac 1 -ar 16000 {} -nostats -hide_banner -loglevel quiet'.format(input_file, output_file)
	subprocess.call(command, shell=True)

def length_normalize(audio, sample_len):
	if len(audio) < sample_len:
		sample = np.zeros(sample_len)
		sample[:len(audio)] = audio
	else:
		sample = audio[:sample_len]

	return sample

def load_audio(vars, aud_path):
	X, _ = librosa.load(aud_path, res_type='kaiser_fast', sr=vars.FRAME_RATE, mono=True)
	features = []

	max_aud_len = int(vars.FRAME_RATE * vars.MAX_AUDIO_DURATION)

	aud = length_normalize(X, max_aud_len)

	num_frames_per_segment = int(vars.MAX_SEGMENT_LENGTH * vars.FRAME_RATE)

	num_samples = int(max_aud_len // num_frames_per_segment)

	for i in range(num_samples):
		sample = aud[int(i*num_frames_per_segment) : int((i+1)*num_frames_per_segment)]
		mfccs = np.mean(librosa.feature.mfcc(y=sample, sr=vars.FRAME_RATE, n_mfcc=13), axis=0)
		features.append(mfccs)

	features = np.array(features)
	features =np.expand_dims(features, axis=2)

	return features