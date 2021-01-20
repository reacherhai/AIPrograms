from __future__ import unicode_literals, print_function, division
from io import open
import  unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_token = 0
BOS_token = 3
EOS_token = 4
batch_size = 32
MAX_LENGTH = 150
'''
SOS_token = 0
EOS_token = 1

word2index = {}
word2count = {}
index2word = {0:"SOS",1:"EOS"}
n_words = 2
'''
#class lang: word->index and index->word
class Lang:
    def __init__(self,name):
        self.name =name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"<PAD>", 1:"<BOS>",2:"<EOS>"}
        self.n_words = 3
    def get_dicts(self,filename):
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                list = line.split()
                index = int(list[0])
                word = list[1]
                count = int(list[2])
                self.word2index[word] = index
                self.word2count[word] = count
                self.index2word = {self.word2index[word]: word for word in self.word2index.keys()}
                self.n_words = len(self.word2index)



train_cn = "train_source_8000.txt"
train_en = "train_target_8000.txt"

def maxlen(list):
    max = 0
    for i in list:
        if(len(i)>max):
            max =  len(i)
    return  max


def readlangs(lang1,lang2,reverse = False):
    sentences_seg = []
    with open(train_cn, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for sentence in lines:
        list = sentence.split(" ")
        sentences_seg.append(list)

    maxl = maxlen(sentences_seg)
    for i, sentence in enumerate(sentences_seg):
        from copy import deepcopy

        newsen = deepcopy(sentence)
        if (len(sentence) < maxl):
            for j in range(maxl - len(sentence)):
                newsen.append("<PAD>")
        sentences_seg[i] = newsen
    lines = []
    for list in sentences_seg:
        str = ""
        for word in list:
            str += word + " "
        lines.append(str)

    with open(train_en,'r',encoding = "utf-8") as f2:
        lines2 = f2.readlines()
    pairs = [[]for i in range(len(lines))]
    for i,line in enumerate(lines):
        content = line.replace("\n", "")
        pairs[i].append(content)
    for i,line in enumerate(lines2):
        content = line.replace("\n", "")
        #content = normalizeString(content)
        pairs[i].append(content)
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    if (reverse):
        pairs = [list(reversed(p)) for p in pairs]
    print(pairs[:10])
    return input_lang, output_lang, pairs


def prepareData(lang1,lang2,reverse = False):
    input_lang, output_lang, pairs = readlangs(lang1,lang2,reverse)
    print("read %s sentence pairs" % len(pairs))

    file_CN = "word_dict.txt"
    file_EN = "word_dict_en.txt"
    input_lang.get_dicts(file_CN)
    output_lang.get_dicts(file_EN)

    print("counted words:")
    print(input_lang.name,input_lang.n_words)
    print (output_lang.name,output_lang.n_words)

    #print(pairs[:10])
    return input_lang,output_lang,pairs

input_lang,output_lang,pairs = prepareData("Chinese","eng")


def langtoSentence(lang,sentence):
    list = []
    for word in sentence.split(' '):
        if(word != ''):
            list.append(lang.word2index[word])
    return list
    #return [lang.word2index[word] for word in sentence.split(' ')]

def tensorfromSentence(lang,sentence):
    index = langtoSentence(lang,sentence)
    return torch.tensor(index,dtype = torch.long,device = device).view(-1,1)

def tensorsFromPair(pair):
    input_tensor = tensorfromSentence(input_lang,pair[0])
    target_tensor = tensorfromSentence(output_lang,pair[1])
    return (input_tensor,target_tensor)

tensorpairs = [tensorsFromPair(pairs[i]) for i in range(len(pairs))]

from torch.utils import data

class mydataset(data.Dataset):
    def __init__(self,pairs):
        self.x_data = [pairs[i][0] for i in range(len(pairs))]
        self.x_lens = [len(pairs[i][0]) for i in range(len(pairs))]
        self.y_data = [pairs[i][1] for i in range(len(pairs))]
        self.y_lens = [len(pairs[i][1]) for i in range(len(pairs))]

    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index],self.x_lens[index],self.y_lens[index]

train_set = mydataset(tensorpairs)

train_loader = data.DataLoader(dataset=train_set, batch_size=32, shuffle=True,drop_last = False)
#print(train_set[:2])


#print(train_set[:10])
#print(tensorsFromPair(random.choice(pairs)))


#print(input_lang.word2index["poll"])
#pair  =random.choice(pairs)



#print ("total pairs",len(pairs))

#print(random.choice(pairs))


