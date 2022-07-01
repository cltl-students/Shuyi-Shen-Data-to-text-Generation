# /usr/bin/env python
# coding=utf-8
from collections import defaultdict
import json
import pandas as pd

def is_soo(triple):
    if (triple[0] in triple[-1]) or (triple[-1] in triple[0]):
        return True
    else:
        return False

def overlapping_search(triples):
    pattern = set()
    entity_pair = [(tri[0], tri[-1]) for tri in triples]
    # if len(set(entity_pair)) != len(triples):
    #     pattern.add('EPO')

    entities = set()
    for idx, triple in enumerate(triples):
        # if is_soo(triple):
        #     pattern.add('SOO')
        entities.add(triple[0])
        entities.add(triple[-1])
    if len(entities) != len(set(entity_pair)) * 2:
        pattern.add('SEO')

    if len(pattern) == 0:
        return ['Normal']
    else:
        return list(pattern)

def process():
    corpus_type = ['Job']
    Saved_corpus_type = 'Job_stats'
    data = ['train', 'val','test']
    for t in corpus_type:
        for dat in data:

            with open(f'../Classifier/PRGC-job/data/job/{dat}_triples.json', 'r', encoding='utf-8') as f:
                data = json.load(f)   
                
            patterns = defaultdict(list)
            for idx, d in enumerate(data):
                pattern = overlapping_search(d['triple_list'])
                for pat in pattern:
                    patterns[pat].append(idx)
            print(f'Patterns of {t} ; data type: {dat} - ', {k: len(v) for k, v in patterns.items()})
            for typ in patterns.keys():
                with open(f'./{ Saved_corpus_type}/{typ}_triples_{dat}.json', 'w', encoding='utf-8') as f:
                    json.dump(pd.Series(data)[patterns[typ]].tolist(), f, indent=4, ensure_ascii=False)

            num_dict = defaultdict(list)
            for d in data:
                if len(d['triple_list']) >= 5:
                    num_dict[5].append(d)
                else:
                    num_dict[len(d['triple_list'])].append(d)
            for key, values in num_dict.items():
                with open(f'./{Saved_corpus_type}/{key}_triples_{dat}.json', 'w', encoding='utf-8') as f:
                    json.dump(values, f, indent=4, ensure_ascii=False)
            print(f'Triples of {dat} dataset: ', {k: len(v) for k, v in num_dict.items()})


if __name__ == '__main__':
    process()
