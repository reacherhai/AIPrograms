
ground_true = []
with open("test_datas/answer.txt", 'r', encoding="utf-8")as ans:
    for line in ans.readlines():
        word = line.replace("\n", "")
        ground_true.append(word)

quests = []
with open("test_datas/questions.txt", 'r', encoding="utf-8")as ans:
    for line in ans.readlines():
        #print(line)
        quests.append(line)

top1Acc_n = 0
top3Acc_n = 0
top5Acc_n = 0
words_n = [[]for i in range(100)]
with open("prediction_n_gram.txt","r", encoding ='utf-8') as pred_l:
    for i,line in enumerate(pred_l.readlines()):
        words = line.split()
        if(ground_true[i] == words[0]):
            top1Acc_n += 1
        if(ground_true[i] in words[:3]):
            top3Acc_n += 1
        if(ground_true[i] in words):
            top5Acc_n += 1
        else:
            words_n[i] = words

top1Acc = 0
top3Acc = 0
top5Acc = 0


with open("prediction_lstm.txt","r", encoding ='utf-8') as pred_l:
    for i,line in enumerate(pred_l.readlines()):
        words = line.split()
        if(ground_true[i] == words[0]):
            top1Acc += 1
        if(ground_true[i] in words[:3]):
            top3Acc += 1
        if(ground_true[i] in words):
            top5Acc += 1
            if(len(words_n[i])>0):
                print(i,quests[i])
                print(i,words,ground_true[i],words_n[i])




top1Acc,top3Acc,top5Acc = top1Acc/100,top3Acc/100,top5Acc/100
top1Acc_n,top3Acc_n,top5Acc_n = top1Acc_n/100,top3Acc_n/100,top5Acc_n/100
print(top1Acc,top3Acc,top5Acc)
print(top1Acc_n,top3Acc_n,top5Acc_n)

list1 = [top1Acc,top3Acc,top5Acc ]
list2 = [top1Acc_n,top3Acc_n,top5Acc_n]
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

name_list = ['top1_Acc', 'top3_Acc', 'top5_Acc']
num_list = list1
num_list1 = list2
x = list(range(len(num_list)))
total_width, n = 0.8, 2
width = total_width / n

plt.bar(x, num_list, width=width, label='LSTM', fc='y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='2-gram', tick_label=name_list, fc='r')
plt.legend()
plt.show()