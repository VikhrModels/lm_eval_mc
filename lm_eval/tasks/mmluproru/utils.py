from datasets import Dataset
import string
import random
import string

def process_docs(dataset: Dataset):
	def _helper(doc,index):
		pref = 10-len(doc['options'])
		if pref!=0:
			add_choices = [dataset[random.randint(1,1900)]['options'][2] for i in range(1,pref)]
			doc['options'].extend(add_choices)
		true_label = doc['options'][doc['answer_index']]
		random.shuffle(doc['options'])
		doc['choices']=doc['options']
		for i,opt in enumerate(doc['options']):
			if opt == true_label:
				doc['answer_index']=i
				doc['answer']=string.ascii_uppercase[i]
				doc['label']=string.ascii_uppercase[i]
				doc['gold']=string.ascii_uppercase[i]
				break
		return doc
	return dataset.map(_helper,with_indices=True)

