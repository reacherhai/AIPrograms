import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
from collections import Counter
from argparse import Namespace

param = Namespace(
    train_file_path="语料/train_seg/train",
    checkpoint_path='checkpoint2',
    seq_size=32,
    batch_size=64,
    embedding_size=128, # embedding dimension
    lstm_size=128, # hidden dimension
    gradients_norm=5, # gradient clipping
    top_k=5,
    num_epochs=50,
    learning_rate=0.001
)


corpus_path = "语料/train_text/"  # 未分词分类预料库路径
seg_path = "语料/train_seg/train"  # 分词后分类语料库路径

catelist = os.listdir(seg_path)  # 获取未分词目录下所有子目录

word_to_int = {}

with open("word_dict.txt",'r',encoding = "utf-8") as fdict:
    for line in fdict.readlines():
        linelist = line.split()
        num = int(linelist[0])
        word = linelist[1]
        word_to_int[word] = num

int_to_word = {k: w for w,k in word_to_int.items()}

from torch.utils import data

class MyDataset(data.Dataset):
    def __init__(self,filepath, batch_size,seq_size):
        text = []
        file_list = os.listdir(filepath)
        for file_path in file_list:
            full_name = filepath + "/" + file_path
            #print("当前处理的文件是:",full_name)
            with open(full_name, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    linelist = line.split()
                    newlist = []
                    for word in linelist:
                        if(len(word)>1):
                            newlist.append(word)
                    text += newlist

        int_data =[word_to_int[word] for word in text]
        num_bacthes = int(len(int_data) / (seq_size * batch_size) )
        x_data = int_data[:num_bacthes*batch_size *seq_size]
        y_data = np.zeros_like(x_data)
        y_data[:-1] = x_data[1:]
        y_data[-1] = x_data[0]

        self.x_data = np.reshape(x_data,(-1,seq_size))
        self.y_data = np.reshape(y_data,(-1,seq_size))

    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, id):
        x = self.x_data[id]
        y = self.y_data[id]
        return x,y




class LSTMModule(nn.Module):
    def __init__(self,numWord,seq_size,embedding_size,lstm_size):
        super(LSTMModule,self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding =  nn.Embedding(numWord,embedding_size)

        self.lstm = nn.LSTM(embedding_size,lstm_size,batch_first = True)
        self.linear = nn.Linear(lstm_size,numWord)
    def forward(self,x,prestate):

        embedding = self.embedding(x)
        output,state = self.lstm(embedding,prestate)
        logits = self.linear(output)
        return logits, state
    #set zero state, used for setting up
    def zero_state(self,batch_size):
        return (torch.zeros(1,batch_size,self.lstm_size), torch.zeros(1,batch_size,self.lstm_size))








import time,sys

def train():
    if(torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    train_data = MyDataset(param.train_file_path,param.batch_size,param.seq_size)
    train_loader = data.DataLoader(dataset = train_data,batch_size = param.batch_size,shuffle = False)
    network = LSTMModule(len(word_to_int), param.seq_size, param.embedding_size, param.lstm_size )
    network = network.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr = param.learning_rate)
    iter = 0
    losses = []
    start_time = time.time()
    minloss = float("inf")
    for epoch in range(param.num_epochs):
        state_h, state_c = network.zero_state(param.batch_size)
        state_h, state_c = state_h.to(device),state_c.to(device)

        for i,(x,y) in enumerate(train_loader):
            iter += 1
            network.train()     #use train mode
            optimizer.zero_grad()    # 梯度清零
            x = x.long()
            y = y.long()
            x = torch.LongTensor(x).to(device)
            y = torch.LongTensor(y).to(device)      #转化模型输入为longtensor
            logits, (state_h,state_c) = network(x,(state_h,state_c))
            #print(logits.size(),y.size())

            loss = criterion(logits.transpose(1,2), y)

            loss_value = loss.item()

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(network.parameters(),param.gradients_norm)

            optimizer.step()
            losses.append(loss_value)

            if iter %1000 == 0:
                torch.save(network.state_dict(),'{}/model-{}.pth'.format(param.checkpoint_path, iter))
            if loss_value < minloss:
                minloss = loss_value
                torch.save(network.state_dict(),'{}/model-final.pth'.format(param.checkpoint_path))
if __name__ == "__main__":
    train()




