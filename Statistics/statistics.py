import json
import pandas as pd
import os 
import random
import sklearn.utils as su

input_path = "../Classifier/PRGC-job/data/job/raw_data/merge_data_version6.json"
train_path = "../Classifier/PRGC-job/data/job/train_triples.json"
dev_path = "../Classifier/PRGC-job/data/job/val_triples.json"
test_path = "../Classifier/PRGC-job/data/job/test_triples.json"

def read_json(path):
    with open(path, "r",encoding = 'utf-8') as fr:
        return [json.loads(line.strip()) for line in fr.readlines()]

def split_to_tran_dev_test(source_path):
    """
    split annotation files(after transforming) to train, val, test splits.
    """
    # with open(source_path, "r", encoding = 'utf-8') as fr:
    #     data = json.load(fr)
    data = read_json(source_path)
    shuffle_data = su.shuffle(data, random_state=7)   
    data_size = len(data)
    All_data = shuffle_data[: int(1*data_size)]
    train_data = shuffle_data[: int(0.8*data_size)]
    dev_data = shuffle_data[int(0.8*data_size) : int(0.9*data_size)]
    test_data = shuffle_data[int(0.9*data_size):]
    
    return All_data,  train_data , dev_data, test_data 

def triple_list(source_path):
    """
    split annotation files(after transforming) to train, val, test splits.
    """
    Experience_skills, Knowledge_skills,  Experience_areas, Knowledge_areas, Degree_in = 0,0,0,0,0
    with open(source_path, "r", encoding = 'utf-8') as fr:
        data = json.load(fr)
    for item in data:
        triples = item["triple_list"]
        Experience_skills+= len([triple[1] for triple in triples if triple[1] == "Experience_skills" ])
        Knowledge_skills+= len([triple[1] for triple in triples if triple[1] == "knowledge_skills" ])
        Experience_areas += len([triple[1] for triple in triples if triple[1] == "Experience_areas" ])
        Knowledge_areas+= len([triple[1] for triple in triples if triple[1] == "knowledge_areas" ])
        Degree_in+= len([triple[1] for triple in triples if triple[1] == "degree_in" ])
    
    Relations =   [Experience_skills, Knowledge_skills, Experience_areas, 
                   Knowledge_areas, Degree_in]
    return Relations
        
def transform_format(data):
    """
    transform raw data to this code repository.
    source path: raw data path
    source path: transformed data path
    """
    # data points in total
    length_dataset = len(data)
    # print('There are {} data points in {}'.format( length_dataset,path), '\n')
    Experience, Knowledge, Skills, Areas, Diploma, Major = 0,0,0,0,0,0
    Experience_skills, Knowledge_skills,  Experience_areas, Knowledge_areas, Degree_in = 0,0,0,0,0
    length_texts = 0
    length_words = 0
    length_triple_list = 0
    for item in data:
        tokens = item["spans"]
        words = item["tokens"]
        length_words  += len([word["text"] for word in words])
        # The number of sentences per data point
        length_texts += len(item['text'].split('\n'))
        # calculate length of each entity per data point
        Experience+= len([token["label"] for token in tokens if token["label"] == "experience" ])
        Knowledge+= len([token["label"] for token in tokens if token["label"] == "knowledge" ])
        Skills += len([token["label"] for token in tokens if token["label"] == "skills" ])
        Areas+= len([token["label"] for token in tokens if token["label"] == "areas" ])
        Diploma+= len([token["label"] for token in tokens if token["label"] == "diploma" ])
        Major+= len([token["label"] for token in tokens if token["label"] == "major" ])
        # calculate length of each relation per data point
        relations = item["relations"]
       
        Experience_skills+= len([relation["label"] for relation in relations if relation["label"] == "Experience_skills" ])
        Knowledge_skills+= len([relation["label"] for relation in relations if relation["label"] == "knowledge_skills" ])
        Experience_areas += len([relation["label"] for relation in relations if relation["label"] == "Experience_areas" ])
        Knowledge_areas+= len([relation["label"] for relation in relations if relation["label"] == "knowledge_areas" ])
        Degree_in+= len([relation["label"] for relation in relations if relation["label"] == "degree_in" ])
        
        triple_list = []
        tokens = item["tokens"]
        tokens = [token["text"] for token in tokens]
        
        for r in item["relations"]:
            head_text = " ".join(
                tokens[r["head_span"]["token_start"]: r["head_span"]["token_end"]+1]
            )
            child_text = " ".join(
                tokens[r["child_span"]["token_start"]: r["child_span"]["token_end"]+1]
            )
            triple_list.append(
                [head_text, r["label"], child_text]
            )
        length_triple_list +=len(triple_list)
    
    Average_length_word = round((length_words/ length_dataset), 2) 
    Average_length_sent = round((length_texts/ length_dataset), 2) 
    Entities =   [Experience, Knowledge, Skills, Areas, Diploma, Major]
    Relations =   [Experience_skills, Knowledge_skills, Experience_areas, 
                   Knowledge_areas, Degree_in]

    return Entities, Relations, Average_length_sent,  Average_length_word, length_triple_list

All_data,  train_data , dev_data, test_data  = split_to_tran_dev_test(input_path)
Entities_1, Relations_1,  Average_length_sent_1,  Average_length_word_1, length_triple_list_1 = transform_format(All_data)
Entities_2, Relations_2,  Average_length_sent_2,  Average_length_word_2, length_triple_list_2 = transform_format(train_data)
Entities_3, Relations_3,  Average_length_sent_3,  Average_length_word_3, length_triple_list_3 = transform_format(dev_data)
Entities_4, Relations_4,  Average_length_sent_4,  Average_length_word_4, length_triple_list_4 = transform_format(test_data)

EN = ["Experience", "Knowledge", "Skills" , "Areas" , "Diploma" ,  "Major"]
RE = ["Experience_skills", "Knowledge_skills", "Experience_areas", "Knowledge_areas", "Degree_in"]

# Create dataframe for statistics
data = {'All_data':Entities_1+Relations_1 +[str(Average_length_sent_1)] +[str(Average_length_word_1)] +[str(length_triple_list_1)],
        'Train':Entities_2+Relations_2+[str(Average_length_sent_2)] +[str(Average_length_word_2)]+[str(length_triple_list_2)], 
        'Validate':Entities_3+Relations_3+[str(Average_length_sent_3)] +[str(Average_length_word_3)]+[str(length_triple_list_3)],
        'Test':Entities_4+Relations_4+[str(Average_length_sent_4)] +[str(Average_length_word_4)]+[str(length_triple_list_4)],
       }

# Creates pandas DataFrame.
df = pd.DataFrame(data, index =EN+RE+['Average_sentence', "Average_word", "Triple"])
df = df.transpose()
print(df.to_markdown())

#define data
train_relation = triple_list(train_path)
val_relation = triple_list(dev_path)
test_relation = triple_list(test_path)
# Create dataframe for statistics
data = {'Train':train_relation, 
        'Validate':val_relation,
        'Test':test_relation}

# Creates pandas DataFrame.
df = pd.DataFrame(data, index =RE)
df = df.transpose()
print(df.to_markdown())