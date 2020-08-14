import torch
from transformers import BertTokenizer

from tqdm import tqdm
from HypoSelector import HypoSelector
import json

import logging

logging.basicConfig(level=logging.ERROR)


dataset_file = "dataset_hypo_selector.json"
dataset = json.load(open(dataset_file))

val_dataset_file = "val_dataset_hypo_selector.json"
val_dataset = json.load(open(val_dataset_file))

l = len(val_dataset)
print("Dataset Size : ", len(dataset), len(val_dataset))

train_results = {}
val_results = {}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

num_labels = 2
hidden_size = 768
model = HypoSelector(num_labels, hidden_size)

model_save_filepath = "checkpoints/hypo_selector/100.pth"
state = torch.load(model_save_filepath)
model.load_state_dict(state['state_dict'])

device = torch.device(f'cuda:0')
print("Using the device : ", device)
model.to(device)

model.eval()

count = 0
total = 0

with torch.no_grad():
	for d in val_dataset:
		question = d["question"]
		question_encoded = tokenizer.encode(question,max_length=32,pad_to_max_length=True,truncation=False)
		question_encoded_tensor = torch.tensor(question_encoded, dtype=torch.int64).unsqueeze(0)
		print(tokenizer.convert_ids_to_tokens(question_encoded))
		hyponyms = d["detections"]
		hyponym_input_ids = []

		for hypo in hyponyms:
			hypo_encoded = tokenizer.encode(hypo,max_length=8,pad_to_max_length=True,truncation=False)
			hyponym_input_ids.append(hypo_encoded)

		while(len(hyponym_input_ids) < 32):
			hyponym_input_ids.append(torch.zeros(8, dtype=torch.int64))
		
		hyponym_input_ids = torch.tensor(hyponym_input_ids, dtype=torch.int64).unsqueeze(0)

		label_hyper_encoded = tokenizer.encode(d["label_hyper"],pad_to_max_length=False,add_special_tokens=False)	

		def find_sub_list(sl,l):
			sll=len(sl)
			for ind in (i for i,e in enumerate(l) if e==sl[0]):
				if l[ind:ind+sll]==sl:
					return ind,ind+sll-1

		label_i,label_j = 0,0
		
		try:
			label_i,label_j = find_sub_list(label_hyper_encoded, question_encoded)
		except:
			continue


		_,_,(hypo,scores) = model({"question_input_ids" : question_encoded_tensor, "detected_hyponym_input_ids" : hyponym_input_ids, "label_hypo" : torch.tensor([0], dtype=torch.int64), "label_i" : torch.tensor([label_i], dtype=torch.int64), "label_j" : torch.tensor([label_j], dtype=torch.int64)})

		hypo = hypo.item()
		scores = scores[0]

		if hyponyms[hypo] == d["label_hypo"]:
			count += 1

		total += 1

		hyponyms_scores = {}
		i = 0

		for h in hyponyms:
			hyponyms_scores[h] = scores[i].item()
			i += 1

		# print(hyponyms,hypo,d["label_hypo"])
		val_results[d["question_id"]] = {"question" : question, "hypo" : hyponyms[hypo], "label_hypo" : d["label_hypo"], "hyponyms_scores" : hyponyms_scores}

print((count/total)*100)
with open("test_hypo_scorer_mask_results_val.json",'w') as f:
	json.dump(val_results,f)
 



count = 0
total = 0

with torch.no_grad():
	for d in dataset:
		question = d["question"]
		question_encoded = tokenizer.encode(question,max_length=32,pad_to_max_length=True,truncation=False)
		question_encoded_tensor = torch.tensor(question_encoded, dtype=torch.int64).unsqueeze(0)
		hyponyms = d["detections"]
		hyponym_input_ids = []

		for hypo in hyponyms:
			hypo_encoded = tokenizer.encode(hypo,max_length=8,pad_to_max_length=True,truncation=False)
			hyponym_input_ids.append(hypo_encoded)

		while(len(hyponym_input_ids) < 32):
			hyponym_input_ids.append(torch.zeros(8, dtype=torch.int64))
		
		hyponym_input_ids = torch.tensor(hyponym_input_ids, dtype=torch.int64).unsqueeze(0)

		label_hyper_encoded = tokenizer.encode(d["label_hyper"],pad_to_max_length=False,add_special_tokens=False)	

		def find_sub_list(sl,l):
			sll=len(sl)
			for ind in (i for i,e in enumerate(l) if e==sl[0]):
				if l[ind:ind+sll]==sl:
					return ind,ind+sll-1

		label_i,label_j = 0,0
		
		try:
			label_i,label_j = find_sub_list(label_hyper_encoded, question_encoded)
		except:
			continue


		_,_,(hypo,scores) = model({"question_input_ids" : question_encoded_tensor, "detected_hyponym_input_ids" : hyponym_input_ids, "label_hypo" : torch.tensor([0], dtype=torch.int64), "label_i" : torch.tensor([label_i], dtype=torch.int64), "label_j" : torch.tensor([label_j], dtype=torch.int64)})
		hypo = hypo.item()
		scores = scores[0]

		if hyponyms[hypo] == d["label_hypo"]:
			count += 1

		total += 1

		hyponyms_scores = {}
		i = 0

		for h in hyponyms:
			hyponyms_scores[h] = scores[i].item()
			i += 1

		# print(hyponyms,hypo,d["label_hypo"])
		train_results[d["question_id"]] = {"question" : question, "hypo" : hyponyms[hypo], "label_hypo" : d["label_hypo"], "hyponyms_scores" : hyponyms_scores}

print((count/total)*100)
with open("test_hypo_scorer_mask_results.json",'w') as f:
	json.dump(train_results,f)
