import torch
import pprint

from transformers import GPT2Tokenizer
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "/content/drive/My Drive/gpt2-trl"

gpt2_model = GPT2HeadWithValueModel.from_pretrained(path)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(path)
_ = gpt2_model.to(device)

input_string = "[negative] The movie"
input_tokens = gpt2_tokenizer.encode(input_string, return_tensors="pt").to(device)
response_tensors = respond_to_batch(gpt2_model, input_tokens, txt_len=256)
response_strings = gpt2_tokenizer.decode(response_tensors[0, :])
pprint.pprint(input_string + response_strings)

input_string = "[positive] The movie"
input_tokens = gpt2_tokenizer.encode(input_string, return_tensors="pt").to(device)
response_tensors = respond_to_batch(gpt2_model, input_tokens, txt_len=256)
response_strings = gpt2_tokenizer.decode(response_tensors[0, :])
pprint.pprint(input_string + response_strings)

input_string = "[neutral] The movie"
input_tokens = gpt2_tokenizer.encode(input_string, return_tensors="pt").to(device)
response_tensors = respond_to_batch(gpt2_model, input_tokens, txt_len=256)
response_strings = gpt2_tokenizer.decode(response_tensors[0, :])
pprint.pprint(input_string + response_strings)
