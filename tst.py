from assign import vars
from assign.models import TEXT_ENCODER, SPEAKER_ENCODER, SPEECH_ENCODER
from assign.utils.data_feeder import data_loader

from transformers import BertModel, BertTokenizer

import numpy as np

speaker_encoder = SPEAKER_ENCODER(vars)
txt_encoder = TEXT_ENCODER(vars)
speech_encoder = SPEECH_ENCODER(vars)


model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

txt, voi, ty = data_loader(vars, encoder=speaker_encoder, tokenizer=tokenizer, model=model)

print(txt_encoder.summary())
pred = txt_encoder.predict(txt[0])

print(speech_encoder.summary())
aud_pred = speech_encoder.predict(voi[0])

print(pred.shape)
print(aud_pred.shape)