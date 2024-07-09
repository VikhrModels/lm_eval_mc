from datasets import Dataset
import string
import random


def process_docs(dataset: Dataset):
    def _helper(doc,index):
        field = doc['correct_answer']
        fi = None
        if field == 'answerA' or field in ['A','А']:
            fi = "A"
        if field == 'answerB' or field in ['B','Б']:
            fi = "B"
        if field == 'answerC' or field in ['C','С']:
            fi = "C"
        if field == 'answerD' or field in ['D','Д']:
            fi = "D"
        if fi:
            doc["choices"] = [doc[f"answer{s}"] for s in list('ABCD')]

        else:
            for idx,(key,val) in enumerate(list(doc.items())[1:-1]):
                if field == val:
                    fi = {0:"A",1:"B",2:"C",3:"D"}.get(idx)


        doc["choices"] = [doc[f"answer{s}"] for s in list('ABCD')]
        if index == 0:
            next_doc1 = dataset[index + 1]
            next_doc2 = dataset[index + 2] if index + 2 < len(dataset) else None
            doc["choices"].extend([next_doc1[f"answer{s}"] for s in list('ABCD')])
            doc["choices"].extend([next_doc2[f"answer{s}"] for s in list('ABCD')])
        elif index == len(dataset) - 1:
            prev_doc1 = dataset[index - 1]
            prev_doc2 = dataset[index - 2]
            doc["choices"].extend([prev_doc1[f"answer{s}"] for s in list('ABCD')])
            doc["choices"].extend([prev_doc2[f"answer{s}"] for s in list('ABCD')])
        else:
            prev_doc = dataset[index - 1]
            next_doc = dataset[index + 1]
            doc["choices"].extend([prev_doc[f"answer{s}"] for s in list('ABCD')])
            doc["choices"].extend([next_doc[f"answer{s}"] for s in list('ABCD')])

        label_map = {label: i for i, label in enumerate(string.ascii_uppercase[:12])}
        inv_label_map = {i: label for i, label in enumerate(string.ascii_uppercase[:12])}
        correct_label = label_map.get(fi)
        if not correct_label:
            return {"label":"failed row w/o answer","gold":""}

        random.shuffle(doc["choices"])
        gold = fi
        shuffled_label = doc["choices"].index(doc[f"answer{gold}"])
        doc["label"] = inv_label_map[shuffled_label]
        doc["gold"] = inv_label_map[shuffled_label]

        return doc
    ds = dataset.map(_helper,with_indices=True)
    return ds.filter(lambda x: len(x['gold'])==1)
