import json

dataset = json.load(open("val_dataset.json"))
dataset_hypo_selector = []

for d in dataset:
	if d["label_hypo"] in d["detections"]:
		dataset_hypo_selector.append(d)

print(len(dataset_hypo_selector))
with open("val_dataset_hypo_selector.json",'w') as f:
	json.dump(dataset_hypo_selector,f)
