import torch
from transformers import BertTokenizer
from tqdm import tqdm
import json, os, sys
from torch.utils.data import Dataset
import pickle
import math

class OKVQA(Dataset):

	def __init__(self, path, file):
		self.max_sequence_length_question = 32
		self.max_sequence_length_hypernym = 8
		self.max_sequence_length_hyponym = 8
		self.max_hyponyms = 32
		self.max_hypernyms = 32
		self.min_isadb_score = -10000000000.0
		self.default_hypo_score = 0.5
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


		if os.path.exists(path + '.pickle'):
			self.data = pickle.load(open(path + '.pickle','rb'))
		else:
			self.data = self.preprocess(file)
			with open(path + '.pickle','wb') as f:
				pickle.dump(self.data,f)


	def preprocess(self, file):

		list_question_input_ids = []
		list_hypernym_input_ids = []
		list_hyponym_input_ids = []
		list_detected_hyponym_input_ids = []
		list_detected_hyponym_scores = []
		list_score_isadb = []
		list_label_i = []
		list_label_j = []	
		list_label_hypo_input_ids = []
		list_label_hypo = []

	
		with open(file, encoding="utf-8") as f:
			data = json.load(f)

			for q in tqdm(data):
				question = q["question"]
				label_hyper = q["label_hyper"]
				label_hypo = q["label_hypo"]
				hypernymy = q["hypernymy_relations"]
				detected_hyponyms = q["detections"]

				question_encoded = self.tokenizer.encode(question,max_length=self.max_sequence_length_question,pad_to_max_length=True,truncation=False)
				
	
				label_hyper_encoded = self.tokenizer.encode(label_hyper,pad_to_max_length=False,add_special_tokens=False)	

				def find_sub_list(sl,l):
					sll=len(sl)
					for ind in (i for i,e in enumerate(l) if e==sl[0]):
						if l[ind:ind+sll]==sl:
							return ind,ind+sll-1
				try:
					label_i,label_j = find_sub_list(label_hyper_encoded, question_encoded)
				except:
					print(label_hyper,question)
					continue
	
				list_label_i.append(label_i)
				list_label_j.append(label_j)
				list_question_input_ids.append(question_encoded)
				
	
				label_hypo_input_ids = self.tokenizer.encode(label_hypo,max_length=self.max_sequence_length_hyponym,pad_to_max_length=True,truncation=False)
	
				list_label_hypo_input_ids.append(label_hypo_input_ids)
	
				scores = {}

				for hypo,hypers in hypernymy.items():
					score = 0.0

					hypernyms = set()

					for hyper in hypers:
						hn = hyper["hypernym"]

						if label_hyper.count(hn) > 0 and not hn in hypernyms:
							score += math.exp(hyper["score"])

						hypernyms.add(hn)

					scores[hypo] = score

				detected_hyponym_input_ids = []
				detected_hyponym_scores = []
				hypo_id = -1
				index = 0

				for detected_hypo in detected_hyponyms:

					if detected_hypo == label_hypo:
						hypo_id = index

					detected_hypo_encoded = self.tokenizer.encode(detected_hypo,max_length=self.max_sequence_length_hyponym,pad_to_max_length=True,truncation=False)
					detected_hyponym_input_ids.append(detected_hypo_encoded)
	 
					if detected_hypo in scores:
						detected_hyponym_scores.append(scores[detected_hypo])

					else:
						detected_hyponym_scores.append(self.default_hypo_score)

					index += 1


				while(len(detected_hyponym_input_ids) < self.max_hyponyms):
					detected_hyponym_input_ids.append(torch.zeros(self.max_sequence_length_hyponym, dtype=torch.int64))
					detected_hyponym_scores.append(0.0)
					
				
				list_detected_hyponym_input_ids.append(detected_hyponym_input_ids)
				list_label_hypo.append(hypo_id)
				list_detected_hyponym_scores.append(detected_hyponym_scores)


				hypernym_input_ids = []
				hyponym_input_ids = []
				score_isadb = []

				# for hypo,hypers in hypernymy.items():

				# 	hypo_encoded = self.tokenizer.encode(hypo,max_length=self.max_sequence_length_hyponym,pad_to_max_length=True,truncation=False)
				# 	hyponym_input_ids.append(hypo_encoded)
					
				# 	hypernym_input_ids_i = []
				# 	score_isadb_i = []

				# 	for hypers_i in hypers:
				# 		hyper = hypers_i["hypernym"]
				# 		hyper_encoded = self.tokenizer.encode(hyper,max_length=self.max_sequence_length_hypernym,pad_to_max_length=True,truncation=False)
				# 		hypernym_input_ids_i.append(hyper_encoded)
				# 		score_isadb_i.append(hypers_i["score"])

				# 	while(len(hypernym_input_ids_i) < self.max_hypernyms):
				# 		hypernym_input_ids_i.append(torch.zeros(self.max_sequence_length_hypernym, dtype=torch.int64))
				# 		score_isadb_i.append(self.min_isadb_score)

				# 	hypernym_input_ids.append(hypernym_input_ids_i)
				# 	score_isadb.append(score_isadb_i)


				# while(len(hyponym_input_ids) < self.max_hyponyms):
				# 	hyponym_input_ids.append(torch.zeros(self.max_sequence_length_hyponym, dtype=torch.int64))
				# 	hypernym_input_ids.append(torch.zeros(self.max_hypernyms,self.max_sequence_length_hypernym, dtype=torch.int64))
				# 	score_isadb.append([self.min_isadb_score]*self.max_hypernyms)
					
				# list_hypernym_input_ids.append(hypernym_input_ids)
				# list_hyponym_input_ids.append(hyponym_input_ids)
				# list_score_isadb.append(score_isadb)
				


		list_question_input_ids = torch.tensor(list_question_input_ids, dtype=torch.int64)
		# list_hypernym_input_ids = torch.tensor(list_hypernym_input_ids, dtype=torch.int64)
		# list_hyponym_input_ids = torch.tensor(list_hyponym_input_ids, dtype=torch.int64)
		list_detected_hyponym_input_ids = torch.tensor(list_detected_hyponym_input_ids, dtype=torch.int64)
		list_detected_hyponym_scores = torch.tensor(list_detected_hyponym_scores, dtype=torch.float64)
		# list_score_isadb = torch.tensor(list_score_isadb, dtype=torch.float64)
		list_label_i = torch.tensor(list_label_i, dtype=torch.int64)
		list_label_j = torch.tensor(list_label_j, dtype=torch.int64)
		list_label_hypo = torch.tensor(list_label_hypo, dtype=torch.int64)
		list_label_hypo_input_ids = torch.tensor(list_label_hypo_input_ids, dtype=torch.int64)


		dataset = {
			'question_input_ids': list_question_input_ids,
			# 'hypernym_input_ids': list_hypernym_input_ids,
			# 'hyponym_input_ids': list_hyponym_input_ids,
			'detected_hyponym_input_ids' : list_detected_hyponym_input_ids,
			'detected_hyponym_scores' : list_detected_hyponym_scores,
			# 'score_isadb': list_score_isadb,
			'label_i': list_label_i,
			'label_j': list_label_j,
			'label_hypo': list_label_hypo,
			'label_hypo_input_ids' : list_label_hypo_input_ids,
		}

		print("Total obects = ", index)

		return dataset


	def __len__(self):
		return self.data['question_input_ids'].shape[0]

	def __getitem__(self, id):

		return {
			'question_input_ids': self.data['question_input_ids'][id],
			# 'hypernym_input_ids': self.data['hypernym_input_ids'][id],
			# 'hyponym_input_ids': self.data['hyponym_input_ids'][id],
			'detected_hyponym_input_ids': self.data['detected_hyponym_input_ids'][id],
			'detected_hyponym_scores' : self.data['detected_hyponym_scores'][id],
			# 'score_isadb': self.data['score_isadb'][id],
			'label_i': self.data['label_i'][id],
			'label_j': self.data['label_j'][id],
			'label_hypo': self.data['label_hypo'][id],
			'label_hypo_input_ids' :self.data['label_hypo_input_ids'][id],
		}
