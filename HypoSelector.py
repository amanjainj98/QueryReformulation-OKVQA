import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import BertModel

class HypoSelector(nn.Module):
	def __init__(self, num_labels=2, hidden_size=768, temperature = torch.tensor([0.0])):
		super(HypoSelector, self).__init__()
		self.num_labels = num_labels
		self.device = None
		self.embedding_size = hidden_size
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.embedding = self.bert.get_input_embeddings()
		self.fc = nn.Linear(hidden_size,512)
		self.fc1 = nn.Linear(hidden_size,512)
		self.bce = nn.BCEWithLogitsLoss(reduction="sum")


	def forward(self, data):
		# data = { k:v.cuda() for k,v in data.items()}
		B,S = data['question_input_ids'].shape
		_,N,SH = data['detected_hyponym_input_ids'].shape

		masked_question = torch.zeros(B,S, dtype=torch.int64).cuda()
		masked_index = torch.zeros(B, dtype=torch.int64).cuda()

		i = data["label_i"].cuda()
		j = data["label_j"].cuda()


		for b in range(B): 
			if j[b] >= i[b]:
				masked_question[b,0:S-j[b]+i[b]] = torch.cat((data['question_input_ids'][b,0:i[b]], torch.tensor([103],dtype=torch.int64), data['question_input_ids'][b,j[b]+1:S]),0)
				masked_index[b] = i[b]

		print(self.tokenizer.convert_ids_to_tokens(masked_question[0]),masked_index[0])

		outputs = self.bert(input_ids=masked_question)

		question_bert_output = outputs[0]  # B X S X 786

		predictions = torch.zeros(B,self.embedding_size).cuda()

		for b in range(B):
			predictions[b] = question_bert_output[b][masked_index[b]]

		embedded_detected_hyponym = torch.sum(self.embedding(data['detected_hyponym_input_ids'].cuda().view(B*N,SH)),dim=1) #[1][0][0] #.view(B,N,self.embedding_size)

		scores = torch.bmm(self.fc1(embedded_detected_hyponym).unsqueeze(1),torch.repeat_interleave(self.fc(predictions), repeats=N, dim=0).unsqueeze(-1)).view(B,N)
		

		hypo = torch.max(scores,dim=1)[1]

		label_hypo = data['label_hypo'].cuda()

		accuracy = torch.sum(hypo == label_hypo)

		label_hypo = nn.functional.one_hot(label_hypo, N)*1.0

		loss = self.bce(scores.view(B*N),label_hypo.view(B*N))
		# loss = torch.sum(loss,dim=0)

		return loss, (accuracy,torch.tensor([0.0]),torch.tensor([0.0])), (hypo,scores)
		
	def to(self, *args, **kwargs):
		self = super().to(*args, **kwargs)
		self.device = args[0] # store device
		self.bert = self.bert.to(*args, **kwargs)
		self.bert.cuda()
		
		return self
