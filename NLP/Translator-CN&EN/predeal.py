# -*- coding: utf-8 -*-
#本程序主要用于jieba分词，以及去除停用词

import os
import re
import jieba
from collections import Counter

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


corpus_path = "语料/ChineseDataset/"  # 未分词分类预料库路径
seg_path = "语料/train_seg_source/"  # 分词后分类语料库路径

catelist = os.listdir(corpus_path)  # 获取未分词目录下所有子目录


worddict = {}

all_words = []

def maxlen(list):
    max = 0
    for i in list:
        if(len(i)>max):
            max =  len(i)
    return  max


def normalizeString(s):
    s = s.replace("。",'.')
    s = s.replace("？",'?')
    s = s.replace("！",'!')
    s = s.replace("，",',')
    s = re.sub(r"([,.!?])", r" \1", s) #add a space before the token
    s = re.sub(r"[^a-zA-Z,.!?\d\u4e00-\u9fa5]+", r" ", s) #remove most of the punctuations
    s = re.sub(r"\s+",r" ",s)
    s = s.lower().strip()
    return s

for mydir in catelist:
    class_path = corpus_path + mydir + "/"
    seg_dir = seg_path + mydir + "/"
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    file_list = os.listdir(class_path)
    for file_path in file_list:
        fullname = class_path +file_path
        print("当前处理的文件是： ",fullname)
        sentences = []
        with open(fullname,"r",encoding = "utf-8") as f:
            for line in f.readlines():
                content = line.replace("\n", "").strip()  # 删除换行和多余的空格
                content = content.replace("   ", " ").replace("  ", " ")
                sentences.append(normalizeString(content).encode("utf-8").decode("utf-8"))
            sentences_seg = [list(jieba.cut(sentence)) for sentence in sentences]
            sentences_seg = [["<BOS>"] + sentence for sentence in sentences_seg]
            for s in sentences_seg:
                while ' '  in s:
                    s.remove(' ')
                while 'nbsp' in s:
                    s.remove('nbsp')
            maxl = maxlen(sentences_seg)
            for i, sentence in enumerate(sentences_seg):
                from copy import deepcopy

                newsen = deepcopy(sentence)
                if (len(sentence) < maxl):
                    for j in range(maxl - len(sentence)):
                        newsen.append("<PAD>")
                newsen.append("<EOS>")
                sentences_seg[i] = newsen
            #for sentence in sentences_seg:
            #    print(len(sentence))
            #    print (sentence)
            list_content = ''
            for content_seg in sentences_seg:
                for word in content_seg:
                    list_content += word
                    list_content += " "
                    if (word in worddict.keys()):
                        worddict[word] += 1
                    else:
                        worddict[word] = 1
                list_content += "\n"
            # print(list_content)
            all_words += list_content.split()
            savefile(seg_dir + file_path, "".join(list_content))

#计数器，此处将无意义字符删去。
c=Counter()
for x in all_words:
    if x != '\r\n':
        c[x] += 1

#获取词典
vocab = c.most_common(vocabulary_size)
index_to_word = [x[0] for x in vocab]
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)]) #{'hello',100}{word,'index'}

#保存到文件
with open("word_dict.txt",'w',encoding = "utf-8") as text:
    for i in range(len(vocab)):
        string = str(i) + " " + vocab[i][0] + " " + str(vocab[i][1]) + "\n"
        text.write(string)