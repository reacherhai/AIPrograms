import jieba
import os
from collections import Counter

train_file_path="语料/train_seg/train_n_gram"
vocab_size = 5000
#由于样本较小，因此出现次数过少的单词不予预测。词表大小选择5000。


## 去除停用词的2个函数
# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 对句子去除停用词
def movestopwords(sentence):
    stopwords = stopwordslist('语料/bd_stop_words.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence:
        if word not in stopwords:
            if word != '\t'and'\n':
                outstr += word
                # outstr += " "
    return outstr

def movestopwordslist(list):
    stopwords = stopwordslist('语料/bd_stop_words.txt')
    outlist = []
    for word in list:
        #print (word)
        if word not in stopwords:
            outlist.append(word)
    return outlist

## use n-gram model to predict text,n-gram is based on previous n-gram

import jieba
from _overlapped import NULL


#将句子变为"BOSxxxxxEOS"这种形式
def reform(sentence):
    #如果是以“。”结束的则将“。”删掉
    if sentence.endswith("。"):
        sentence=sentence[:-1]
    #添加起始符BOS和终止符EOS
    sentence_modify1=sentence.replace("。", "EOSBOS")
    sentence_modify2="BOS"+sentence_modify1+"EOS"
    return sentence_modify2


#分词并统计词频
def segmentation(sentence,lists,dicts=NULL):
    jieba.suggest_freq("BOS", True)
    jieba.suggest_freq("EOS", True)
    sentence = jieba.cut(sentence,HMM=False)
    format_sentence=",".join(sentence)
    #将词按","分割后依次填入数组word_list[]
    lists=format_sentence.split(",")
    #统计词频，如果词在字典word_dir{}中出现过则+1，未出现则=1
    if dicts!=NULL:
        for word in lists:
            if word not in dicts:
                dicts[word]=1
            else:
                dicts[word]+=1
    return lists


#比较两个数列，二元语法
def compareList(ori_list,test_list):
    #申请空间
    count_list=[0]*(len(test_list))
    #遍历测试的字符串
    for i in range(0,len(test_list)-1):
        #遍历语料字符串，且因为是二元语法，不用比较语料字符串的最后一个字符
        for j in range(0,len(ori_list)-2):
            #如果测试的第一个词和语料的第一个词相等则比较第二个词
            if test_list[i]==ori_list[j]:
                if test_list[i+1]==ori_list[j+1]:
                    count_list[i]+=1
    return count_list

def count_tuple(word1,word2,ori_list):
    count = 0
    for i,word in enumerate(ori_list):
        if( word == word1):
            if(ori_list[i+1] ==word2):
                count +=1
    return count

#计算概率
def probability(test_list,count_list,ori_dict):
    flag=0
    #概率值为p
    p=1
    for key in test_list:
        #数据平滑处理：加1法
        if(key not in ori_dict):
            continue
        p*=(float(count_list[flag]+1)/float(ori_dict[key]+1))
        flag+=1
    return p

def load_data(filepath):
    text = []
    file_list = os.listdir(filepath)
    for file_path in file_list:
        full_name = filepath + "/" + file_path
        # print("当前处理的文件是:",full_name)
        with open(full_name, "r", encoding="utf-8") as f:
            for line in f.readlines():
                linelist = line.split()
                newlist = []
                for word in linelist:
                    if (len(word) > 1):
                        newlist.append(word)
                text += newlist
    return text


if __name__ == "__main__":
    ori_list = []
    test_list = []
    ori_list = load_data(train_file_path)
    ori_dict = {}

    with open("word_dict_n-gram.txt", 'r', encoding="utf-8") as fdict:
        for line in fdict.readlines():
            linelist = line.split()
            num = int(linelist[0])
            word = linelist[1]
            count = int(linelist[2])
            ori_dict[word] = count

    #ori_list = segmentation(sentence_ori_temp,ori_list,ori_dict)

    #test_sentence.replace("[MASK]",'MM')

    ground_true = []
    with open("test_datas/answer.txt", 'r', encoding="utf-8")as ans:
        for line in ans.readlines():
            word = line.replace("\n", "")
            ground_true.append(word)
    acc = 0

    quests = []

    ##quests need to delete stopwords
    with open("test_datas/questions.txt", 'r', encoding="utf-8")as quest:
        for line in quest.readlines():
            # sentence = movestopwords(line)
            sentence = line.replace("[MASK]", " MM ")
            seg_list = list(jieba.cut(sentence))
            seg_list = movestopwordslist(seg_list)
            seg_list = ["<BOS>"] + seg_list + ["<EOS>"]
            temp_list = seg_list.copy()
            for i, word in enumerate(temp_list):
                if (len(word)) <= 1:
                    seg_list.remove(word)
            quests.append(seg_list)

    #print(quests[:10])


    acc = 0
    most_frequent_dict = {k: ori_dict[k] for k in list(ori_dict.keys())[:vocab_size]}

    myanswers = open("prediction_n_gram.txt", 'w', encoding='utf-8')

    for i in range(len(quests)):
        maxword = ""
        maxprob = 0
        predlist = []
        test_list = quests[i]
        iter = 0
        for key,count in most_frequent_dict.items():
            import copy
            templist = copy.deepcopy(test_list)
            if(key not in ["<BOS>","<EOS>"]):

                #2-gram:
                index2 = templist.index("MM")
                templist[index2] = key
                index1 = index2 -1
                pre_word = templist[index1]
                pred_word = key
                c = count_tuple(pre_word,pred_word, ori_list)
                #print(pre_word,pred_word,c)

                if(c>maxprob):
                    maxprob = c
                    maxword = pred_word
                predlist.append((c, key))
                '''
                #multi-gram
                index = templist.index("MM")
                templist[index] = key
                #test_list = test_list.replace("MM",key)
                count_list = compareList(ori_list, templist)
                
                pred = probability(templist,count_list,ori_dict)

                predlist.append((pred,key))
                #pred = predict(sentence_ori,testdata,ori_dict)
                if(pred>maxprob):
                    maxprob = pred
                    maxword = key
                '''
            iter += 1

        predlist = sorted(predlist)
        predlist.reverse()
        wordlist = [word for (_,word) in predlist[:5]]

        ans = ground_true[i]
        if(ans in wordlist):
            print("{}/ {} - {} √".format(i,wordlist,ans))
            acc +=1
        else:
            print("{}/ {} - {} ×".format(i,wordlist, ans))

        myanswers.write("{}\n".format(' '.join(wordlist)))

    print("Acc: {}".format(acc))
