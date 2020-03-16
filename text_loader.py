from assign import vars
from assign.models import SENTENCE_ENCODER
from assign.utils.data_feeder import data_loader
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = SENTENCE_ENCODER(vars)

tx, ty = data_loader(vars, tokenizer=tokenizer, model=model, load_mode='text')
print(tx.shape)
print(ty.shape)