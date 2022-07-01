# coding=utf-8
import argparse
import time
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
import pandas as pd
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.filterwarnings('ignore')


def generate(text):
    model.eval()
    input_ids = tokenizer.encode("WebNLG:{} </s>".format(text), return_tensors="pt")  # Batch size 1
    # input_ids.to(dev)
    outputs = model.generate(input_ids)
    gen_text=tokenizer.decode(outputs[0]).replace('<pad>','').replace('</s>','')

    print('generated text:', gen_text)
    return gen_text

def predict_one(rdf):
    # res = generate(' Russia | leader | Putin')
    s = time.time()
    res = generate(rdf)
    elapsed = time.time() - s
    print('Generated in {} seconds'.format(str(elapsed)[:4]))
    # print(res)
    return res

def predict_test_csv(csv_file):
    pred = []
    df=pd.read_csv(csv_file, index_col=[0])
    input_list = df['input_text'].tolist()
    label = df['target_text'].tolist()
    
    for input in input_list:
        gen_text = generate(input)
        pred.append(gen_text)
    with open('result/{}_pred_final.txt'.format(args.mode)  ,'w', encoding='utf8') as f:
        for _pred in pred:
          f.write(_pred)
          f.write('\n')
    with open('result/{}_label_final.txt'.format(args.mode) ,'w', encoding='utf8') as f:
        for _label in label:
          f.write(_label)
          f.write('\n')


if __name__ == '__main__':
    # parameter
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',  default='t5', type=str)
    parser.add_argument('--data',  default='data/test_result.csv', type=str)
    args = parser.parse_args()

    mode = args.mode
    data_file = args.data


    # load model
    if 't5' in mode.lower():
        model = T5ForConditionalGeneration.from_pretrained('result/nlg_{}_100.bin'.format(mode), return_dict=True,config='nlg_t5/config.json')

        """uncomment to test the ablation study"""
        # model = T5ForConditionalGeneration.from_pretrained('result/nlg_{}_70.bin'.format(mode), return_dict=True,config='nlg_t5/config.json')
        # model = T5ForConditionalGeneration.from_pretrained('result/nlg_{}_50.bin'.format(mode), return_dict=True,config='nlg_t5/config.json')
        # model = T5ForConditionalGeneration.from_pretrained('result/nlg_{}_10.bin'.format(mode), return_dict=True,config='nlg_t5/config.json')
        # model = T5ForConditionalGeneration.from_pretrained('result/nlg_{}_1.bin'.format(mode), return_dict=True,config='nlg_t5/config.json')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
 
    predict_test_csv(data_file)
   