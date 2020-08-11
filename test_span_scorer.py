import torch
from transformers import BertTokenizer

from tqdm import tqdm
from SpanSelector import SpanSelector
import json

import logging

logging.basicConfig(level=logging.ERROR)


dataset_file = "dataset.json"
dataset = json.load(open(dataset_file))

l = len(dataset)
print("Dataset Size : ", len(dataset))

results = []

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

num_labels = 2
hidden_size = 768
model = SpanSelector(num_labels, hidden_size)

model_save_filepath = "checkpoints/span_selector/17.pth"
state = torch.load(model_save_filepath)
model.load_state_dict(state['state_dict'])

device = torch.device(f'cuda:0')
print("Using the device : ", device)
model.to(device)

model.eval()

count = 0
total = 0
iou_score = 0
precision_score = 0
recall_score = 0

with torch.no_grad():
	for d in tqdm(dataset):
		question = d["question"]
		question_encoded = tokenizer.encode(question,max_length=32,pad_to_max_length=True,truncation=False)
		question_encoded_tensor = torch.tensor(question_encoded, dtype=torch.int64).unsqueeze(0)
		_,_,(i,j) = model({"question_input_ids" : question_encoded_tensor, "label_i" : torch.tensor([0], dtype=torch.int64), "label_j" : torch.tensor([1], dtype=torch.int64)})
		i = i.item()
		j = j.item()
		span = tokenizer.decode(question_encoded_tensor[0][i:j+1])

		label_hyper_encoded = tokenizer.encode(d["label_hyper"],pad_to_max_length=False,add_special_tokens=False)	

		def find_sub_list(sl,l):
			sll=len(sl)
			for ind in (i for i,e in enumerate(l) if e==sl[0]):
				if l[ind:ind+sll]==sl:
					return ind,ind+sll-1
		try:
			label_i,label_j = find_sub_list(label_hyper_encoded, question_encoded)
		except:
			continue

		span_ids = range(i,j+1)
		label_hyper_ids = range(label_i,label_j+1)

		overlap = len([a for a in span_ids if a in label_hyper_ids])
		if span_ids:
			precision_score += overlap / len(span_ids)

		recall_score += overlap / len(label_hyper_ids)

		iou_score += overlap / (len(span_ids) + len(label_hyper_ids) - overlap)

		if span == d["label_hyper"]:
			count += 1

		total += 1

		results.append({"question_id" : d["question_id"], "question" : question, "span" : span, "label_hyper" : d["label_hyper"]})


print((count/total)*100)
print((iou_score/total)*100)
print((precision_score/total)*100)
print((recall_score/total)*100)
# with open("test_span_scorer_results.json",'w') as f:
# 	json.dump(results,f)
