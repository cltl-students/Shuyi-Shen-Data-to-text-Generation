# coding=utf-8
import argparse
import os
import pandas as pd
import numpy as np
import random
import os
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers.optimization import Adafactor, AdamW
import time
from torch.utils.data import Dataset,DataLoader
from data import NLGDataset
import logging
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

def train(train_df, mode, optimizer, model, model_name):
    train_d = NLGDataset(train_df, mode=mode)
    train_mydataloader = DataLoader(dataset=train_d,
                              batch_size=batch_size,
                              num_workers=2,
                              collate_fn= train_d.collate_fn,
                              shuffle=True)
    
    # model.to(device)
    # train network
    #Sets the module in training mode
    model.train()
    best_loss = 100000.
    print('start training')
    loss_per_10_steps=[]
    # for epoch in tqdm(range(1,num_of_epochs+1), total=num_of_epochs):
    #     print('Running epoch: {}'.format(epoch))
    running_loss=[]

    # for i, (inputbatch, labelbatch) in tqdm(enumerate(mydataloader), total=len(mydataloader)):
    for i, (inputbatch, labelbatch) in enumerate(train_mydataloader) :

        inputbatch = torch.squeeze(inputbatch, dim=1)  
        labelbatch = torch.squeeze(labelbatch, dim=1)  
        inputbatch=inputbatch.to(device)
        labelbatch=labelbatch.to(device)
        # print(inputbatch.shape,labelbatch.shape)
        # print(labelbatch.shape)
        # clear out the gradients of all Variables
        optimizer.zero_grad()

        # Forward propogation
        # outputs = model(input_ids=inputbatch, labels=labelbatch)
        outputs = model(inputbatch, labels=labelbatch)

        loss = outputs.loss
        loss_num=loss.item()
        logits = outputs.logits
        running_loss.append(loss_num)
        if i%10 ==0:
            loss_per_10_steps.append(loss_num)
            print("current progress: {} steps - loss: {}".format(i,loss_num))
        # calculating the gradients
        loss.backward()

        #updating the params
        optimizer.step()

    _loss= np.mean(np.array(running_loss))
    if _loss < best_loss:
        torch.save(model.state_dict(), os.path.join('result',model_name) )
    best_loss = min(best_loss, _loss)
    print('Epoch: {} , Running loss: {}'.format(epoch,_loss))
   
    return  _loss
    
    
def val(val_df, mode, model, model_name):
    val_d = NLGDataset(val_df, mode=mode)
    val_mydataloader = DataLoader(dataset=val_d,
                              batch_size=batch_size,
                              num_workers=2,
                              collate_fn= val_d.collate_fn,
                              shuffle=True)

        
    #Sets the module in training mode
    model.eval()
    best_loss = 100000.
    print('start validating')
    loss_per_10_steps=[]
    # for epoch in tqdm(range(1,num_of_epochs+1), total=num_of_epochs):
    #     print('Running epoch: {}'.format(epoch))
    running_loss=[]

    # for i, (inputbatch, labelbatch) in tqdm(enumerate(mydataloader), total=len(mydataloader)):
    for i, (inputbatch, labelbatch) in enumerate(val_mydataloader) :

        inputbatch = torch.squeeze(inputbatch, dim=1)  
        labelbatch = torch.squeeze(labelbatch, dim=1)  
        inputbatch=inputbatch.to(device)
        labelbatch=labelbatch.to(device)
        # print(inputbatch.shape,labelbatch.shape)
        # print(labelbatch.shape)
        # clear out the gradients of all Variables
        # optimizer.zero_grad()

        # Forward propogation
        # outputs = model(input_ids=inputbatch, labels=labelbatch)
        outputs = model(inputbatch, labels=labelbatch)

        loss = outputs.loss
        loss_num=loss.item()
        logits = outputs.logits
        running_loss.append(loss_num)
        if i%10 ==0:
            loss_per_10_steps.append(loss_num)
            print("current progress: {} steps - val_loss: {}".format(i,loss_num))
#         # calculating the gradients
#         loss.backward()

#         #updating the params
#         optimizer.step()

    val_loss= np.mean(np.array(running_loss))
    if val_loss < best_loss:
        torch.save(model.state_dict(), os.path.join('result',model_name) )
    best_loss = min(best_loss, val_loss)
    print('Epoch: {} , Running loss: {}'.format(epoch,val_loss))

    return  val_loss



if __name__ == '__main__':
    # parameter
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',  default='t5', type=str)
    parser.add_argument('--data_train',  default='data/train.csv', type=str)
    parser.add_argument('--data_dev',  default='data/validate.csv', type=str)
    args = parser.parse_args()

    mode = args.mode
    train_data_file = args.data_train
    dev_data_file = args.data_dev
    # parameter
    batch_size=4
    num_of_epochs=2
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu'  )

    
    print('load train model_{}'.format(mode))
    if 't5' in mode.lower():
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
        optimizer = Adafactor(
            model.parameters(),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

   
    #moving the model to device(GPU/CPU)
    model.to(device)
    model_name = 'nlg_{}_70.bin'.format(mode)

    # laod data
    print('load data')
    train_df=pd.read_csv(train_data_file, index_col=[0])
    # train_df=train_df.iloc[  :50,:]
    train_df=train_df.sample(frac = 0.7, replace=True, random_state=1)
    
    val_df=pd.read_csv(dev_data_file , index_col=[0])
    # train_df=train_df.iloc[  :50,:]
    val_df=val_df.sample(frac = 0.7, replace=True, random_state=1)
    
    print('the data we"re training is：', len(train_df))
    print('the data we"re validating is：', len(val_df))
    
    epoch_train_val_loss = {'train_loss':[], 'val_loss':[]}
    
    # training
    # epoch_train_loss =  []
    for epoch in tqdm(range(1,num_of_epochs+1), total=num_of_epochs):
        print(' Running epoch: {}'.format(epoch))
        
        train_loss = train(train_df, mode, optimizer, model, model_name)
        val_loss = val(val_df, mode, model, model_name)
        
        print("  train_loss: %.5f - val_loss: %.5f-"%(train_loss, val_loss))
        print()
        
        epoch_train_val_loss['train_loss'].append(train_loss)
        epoch_train_val_loss['val_loss'].append(val_loss)

    
    epochs = [i+1 for i in range(len( epoch_train_val_loss['train_loss']))]
    plt.plot(epochs, epoch_train_val_loss['train_loss'], label= "train_loss")
    plt.plot(epochs, epoch_train_val_loss['val_loss'], label= "val_loss")
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Average loss per epoch')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('./result/loss_function_70.png', dpi=100)