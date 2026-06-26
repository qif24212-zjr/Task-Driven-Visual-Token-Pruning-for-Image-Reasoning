from transformers import AutoTokenizer
from transformers import AutoModel

import torch

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

text = "cat"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

embedding = outputs.last_hidden_state.mean(dim=1)

print(embedding.shape)