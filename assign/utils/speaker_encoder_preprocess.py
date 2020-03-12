import numpy as np
import librosa


def length_normalize(audio, sample_len):
	if len(audio) < sample_len:
		sample = []
		while len(sample) < sample_len:
			sample += list(audio)

		audio = np.array(sample, dtype=np.float32)

	audio = audio[:sample_len]
	return audio

def load_audio(vars, aud_path):
	X, _ = librosa.load(aud_path, res_type='kaiser_fast', sr=vars.FRAME_RATE, mono=True)
	features = []

	max_aud_len = vars.FRAME_RATE * vars.MAX_AUDIO_DURATION

	aud = length_normalize(X, max_aud_len)

	num_samples = vars.MAX_AUDIO_DURATION // vars.MAX_SEGMENT_LENGTH

	for i in range(num_samples):
		sample = aud[i*vars.MAX_SEGMENT_LENGTH : (i+1)*vars.MAX_SEGMENT_LENGTH]
		mfccs = np.mean(librosa.feature.mfcc(y=sample, sr=vars.FRAME_RATE, n_mfcc=13), axis=0)
		features.append(mfccs)

	features = np.array(features)
	features =np.expand_dims(features, axis=2)