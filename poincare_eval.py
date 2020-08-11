from hype.hypernymy_eval import EntailmentConeModel
import argparse
import os
import torch as th
from nltk.corpus import wordnet
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer 
import json

lemmatizer = WordNetLemmatizer()


parser = argparse.ArgumentParser()
parser.add_argument('file')
chkpnt = parser.parse_args()
chkpnt = chkpnt.file

if isinstance(chkpnt, str):
	assert os.path.exists(chkpnt)
	chkpnt = th.load(chkpnt)


model = EntailmentConeModel(chkpnt)

hypernymy_relations = json.load(open("hypernymy_relations_merged.json"))
hypernymy_relations_scores = {}

for k,v in tqdm(hypernymy_relations.items()):
	hypos = [h["instance"].strip().replace(" ","_") for h in v]
	hypers = [h["class"].strip() for h in v]
	
	hypos_synsets = []
	hypos_mask = []
	for hypo in hypos:
		s = wordnet.synsets(hypo.replace(" ","_"))
		if s:
			hypos_synsets.append(s[0].name())
			hypos_mask.append(1)
		else:
			s = wordnet.synsets(lemmatizer.lemmatize(hypo).replace(" ","_"))
			if s:
				hypos_synsets.append(s[0].name())
				hypos_mask.append(1)
			else:
				hypos_synsets.append(hypo)
				hypos_mask.append(0)


	hypers_synsets = []
	hypers_mask = []
	for hyper in hypers:
		s = wordnet.synsets(hyper.replace(" ","_"))
		if s:
			hypers_synsets.append(s[0].name())
			hypers_mask.append(1)
		else:
			s = wordnet.synsets(lemmatizer.lemmatize(hyper).replace(" ","_"))
			if s:
				hypers_synsets.append(s[0].name())
				hypers_mask.append(1)
			else:
				hypers_synsets.append(hyper)
				hypers_mask.append(0)

	dists = []
	with th.no_grad():
		dists = model.predict_many(hypos_synsets, hypers_synsets)

	vv = [{"instance":v[i]["instance"],"class":v[i]["class"],"frequency":v[i]["frequency"],"present":hypos_mask[i]*hypers_mask[i],"score":dists[i].item()} for i in range(len(v))]

	hypernymy_relations_scores[k] = vv


with open("hypernymy_relations_scores.json",'w') as f:
	json.dump(hypernymy_relations_scores,f)


