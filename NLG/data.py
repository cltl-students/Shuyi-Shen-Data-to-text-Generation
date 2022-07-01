# coding=utf-8
import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from transformers import T5Tokenizer
from torch.nn import functional as F

class NLGDataset(Dataset):
    def __init__(self, df, mode='t5'):
        self.data = df
        if 't5' in mode.lower():
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base')


    def __getitem__(self,index):
        row = self.data.iloc[index]
        # print(row['input_text'])
        input = 'WebNLG: '+row['input_text']+'</s>'
        labels = row['target_text']+'</s>'

        inputbatch=self.tokenizer.encode_plus(input,
                                              # padding='max_length',
                                              truncation=True,
                                              max_length=512,return_tensors='pt')["input_ids"]
        labelbatch=self.tokenizer.encode_plus(labels,
                                              # padding='max_length',
                                              truncation=True,
                                              max_length=512,return_tensors="pt") ["input_ids"]
       
        return inputbatch, labelbatch

    def __len__(self):
        return len(self.data)


    def collate_fn(self, l):
        inputbatch, labelbatch = zip(*l)
        max_len = 0
        for x in inputbatch:
            max_len = max(x.shape[-1], max_len)
        for x in labelbatch:
            max_len = max(x.shape[-1], max_len)

        l1 = []
        for x in inputbatch:
            if x.shape[-1] < max_len:
                padding = ( 0, max_len-x.shape[-1] )
                x = F.pad(x, padding)
            l1.append(x)

        # max_len = 0
        # for x in labelbatch:
        #     max_len = max(x.shape[-1], max_len)
        l2 = []
        # print(max_len)
        for x in labelbatch:
            if x.shape[-1] < max_len:
                padding = ( 0, max_len-x.shape[-1] )
                x = F.pad(x, padding)
            l2.append(x)
            # print(x.shape)
        l1 = torch.stack(l1, dim=0)
        l2 = torch.stack(l2, dim=0)
        return  l1,  l2

if __name__ == '__main__':

    # load data
    train_df=pd.read_csv('data/train.csv', index_col=[0])
    val_df=pd.read_csv('data/validate.csv', index_col=[0])
    # train_df=train_df.iloc[  :500,:]
    train_df=train_df.sample(frac = 1, replace=True, random_state=1)
    val_df=val_df.sample(frac = 1, replace=True, random_state=1)
    
###################################### train 

    print("train data:--", len(train_df))
    print("val data:--", len(val_df))
    
    for index in range(len(train_df)):
        row =train_df.iloc[index]
        print(row)
        input = 'WebNLG: '+row['input_text']+'</s>'
        labels = row['target_text']+'</s>'
        print(input)
        print(labels)
    exit()
    
###################################### validate   

    for index in range(len(val_df)):
        row =val_df.iloc[index]
        print(row)
        input = 'WebNLG: '+row['input_text']+'</s>'
        labels = row['target_text']+'</s>'
        print(input)
        print(labels)
    exit()
    
    train_d = NLGDataset(train_df, mode='t5')
    val_d = NLGDataset(val_df, mode='t5')
    
    train_mydataloader = DataLoader(dataset=train_d,
                              batch_size=3,
                              # num_workers=4,
                              collate_fn= train_d.collate_fn,
                              shuffle=True)
    
    val_mydataloader = DataLoader(dataset=val_d,
                              batch_size=3,
                              # num_workers=4,
                              collate_fn= val_d.collate_fn,
                              shuffle=True)

    for data, label in train_mydataloader:
        print(data.shape, label.shape)
        pass
    
    for data, label in val_mydataloader:
        print(data.shape, label.shape)
        pass
        # exit()