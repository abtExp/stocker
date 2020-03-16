from assign import vars
from assign.models import TEXT_ENCODER

model = TEXT_ENCODER(vars)
model.train()