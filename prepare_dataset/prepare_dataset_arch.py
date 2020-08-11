import json

labels = json.load(open("train_annotations_labels_round1.json"))
hypernymy_relations_scores = json.load(open("hypernymy_relations_scores.json"))
detections = json.load(open("detections.json"))
questions_list = json.load(open("OpenEnded_mscoco_train2014_questions.json"))
questions_list = questions_list["questions"]

questions = {}

for q in questions_list:
	questions[str(q["question_id"])] = {"image_id" : q["image_id"], "question" : q["question"]}


dataset = []
for l in labels:
	dataset_i = {}
	qid = l["fields"]["question_id"]
	image_id = str(questions[qid]["image_id"])

	dataset_i["question_id"] = qid
	dataset_i["question"] = questions[qid]["question"].lower()
	dataset_i["detections"] = detections[qid]["detections"]
	dataset_i["label_hyper"] = l["fields"]["hypernym"].lower()
	dataset_i["label_hypo"] = l["fields"]["hyponym"].lower().split("/")[0].replace("_"," ")

	hypernymy_relations = hypernymy_relations_scores[image_id]

	hr = {}
	for h in hypernymy_relations:
		k = h["instance"]
		if k in hr:
			hr[k].append({"hypernym" : h["class"].strip(), "score" : h["score"]})
		else:
			hr[k] = [{"hypernym" : h["class"].strip(), "score" : h["score"]}]


	dataset_i["hypernymy_relations"] = hr
	dataset.append(dataset_i)



with open("dataset.json",'w') as f:
	json.dump(dataset,f)