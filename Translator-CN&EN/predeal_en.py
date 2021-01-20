# -*- coding: utf-8 -*-
#本程序主要用于jieba分词，以及去除停用词

import os
import re
import jieba
from collections import Counter
from nltk.tokenize import word_tokenize,sent_tokenize
import unicodedata
vocabulary_size = 500000

# 保存文件的函数
def savefile(savepath,content):
    fp = open(savepath,'w',encoding='utf-8',errors='ignore')
    fp.write(content)
    fp.close()

# 读取文件的函数
def readfile(path):
    fp = open(path, "r", encoding='utf-8', errors='ignore')
    content = fp.read()
    fp.close()
    return content

corpus_path = "语料/EnglishDataset/"  # 未分词分类预料库路径
seg_path = "语料/train_seg_dest/"  # 分词后分类语料库路径


catelist = os.listdir(corpus_path)  # 获取未分词目录下所有子目录

worddict = {}

all_words = []

def maxlen(list):
    max = 0
    for i in list:
        if(len(i)>max):
            max =  len(i)
    return  max

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1", s) #add a space before the token
    s = re.sub(r"[^a-zA-Z,.!?\d]+", r" ", s) #remove most of the punctuations
    s = re.sub(r"\s+",r" ",s)
    return s


for mydir in catelist:
        class_path = corpus_path + mydir + "/"  # 拼出分类子目录的路径
        seg_dir = seg_path + mydir  + "/"  # 拼出分词后预料分类目录
        if not os.path.exists(seg_dir):  # 是否存在，不存在则创建
            os.makedirs(seg_dir)

        file_list = os.listdir(class_path) # 列举当前目录所有文件
        for file_path in file_list:
            fullname = class_path + file_path # 路径+文件名
            print("当前处理的文件是： ",fullname)
            sentences = []
            with open(fullname,'r',encoding = 'utf-8') as f:
                for line in f.readlines():
                    sentences.append(normalizeString(line).encode("utf-8").decode("utf-8"))
                    #sentences.append(line)
            sentences_seg = [word_tokenize(sentence) for sentence in sentences]
            sentences_seg = [["<BOS>"] + sentence for sentence in sentences_seg ] #添加token
            maxl =  maxlen(sentences_seg)
            for i,sentence in enumerate(sentences_seg):
                from copy import deepcopy
                newsen = deepcopy(sentence)
                if(len(sentence)<maxl):
                    for j in range(maxl - len(sentence)):
                        newsen.append("<PAD>")
                newsen.append("<EOS>")
                sentences_seg[i] = newsen
            list_content = ''
            for content_seg in sentences_seg:
                for word in content_seg:
                    list_content += word
                    list_content += " "
                    if(word in worddict.keys()):
                        worddict[word] += 1
                    else:
                        worddict[word] = 1
                words = list(content_seg)
                list_content += "\n"
            #print(list_content)
            list_content = list_content.replace("   ", " ").replace("  ", " ")
            all_words +=  list_content.split()
            savefile(seg_dir+file_path, "".join(list_content))


#计数器，此处将所有标点符号和长度小于等于1的无意义字符删去。
c=Counter()
for x in all_words:
    if x != '\r\n':
        c[x] += 1

#获取词典
vocab = c.most_common(vocabulary_size)
index_to_word = [x[0] for x in vocab]
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)]) #{'hello',100}{word,'index'}

#保存到文件
with open("word_dict_en.txt",'w',encoding = "utf-8") as text:
    for i in range(len(vocab)):
        string = str(i) + " " + vocab[i][0] + " " + str(vocab[i][1]) + "\n"
        text.write(string)