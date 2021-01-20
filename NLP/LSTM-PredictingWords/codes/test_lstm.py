from Mylstm import *
import jieba
###test model on test data

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

##now use lstm model to predict:

def predict(device,network, quest, numWord, topk = 5):
    network.eval()
    q_index = quest.index("MM")
    pre_q , post_q = quest[:q_index],quest[q_index+len("MM"):]
    state_h, state_c = network.zero_state(1)
    state_h,state_c = state_h.to(device), state_c.to(device)

    for word in pre_q:
        index = word_to_int.get(word)
        if(index ==None):
            continue
        ix = torch.tensor([[index]]).to(device)
        #label chaoguoyuzhi
        out, (state_h,state_c) = network(ix,(state_h,state_c))
    _,topix = torch.topk(out[0],k = topk)
    choices = topix.tolist()
    return [int_to_word[x] for x in choices[0]]

if __name__ =="__main__":
    if (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    network = LSTMModule(len(word_to_int), param.seq_size, param.embedding_size, param.lstm_size)
    network.load_state_dict(torch.load("checkpoint2/model-final.pth"))
    network.to(device)

    ground_true = []
    with open("test_datas/answer.txt", 'r', encoding="utf-8")as ans:
        for line in ans.readlines():
            word = line.replace("\n", "")
            ground_true.append(word)
    acc = 0

    quests = []

    with open("test_datas/questions.txt", 'r', encoding="utf-8")as quest:
        for line in quest.readlines():
            #sentence = movestopwords(line)
            sentence = line.replace("[MASK]", " MM ")
            seg_list = list(jieba.cut(sentence))
            seg_list = movestopwordslist(seg_list)
            temp_list = seg_list.copy()
            for i, word in enumerate(temp_list):
                if (len(word)) <= 1:
                    seg_list.remove(word)
            quests.append(seg_list)

    print(quests[:10])

    myanswers = open("prediction_lstm.txt", 'w',encoding='utf-8')
    for i in range(len(quests)):
        pred = predict(device,network,quests[i],len(word_to_int),5)
        ans = ground_true[i]
        if (ans in pred):
            acc += 1
            print("{} - pred: {} ans: {} √ ".format(i,pred,ans) )
        else:
            print("{} - pred: {} ans: {} ❌".format(i,pred,ans) )

        myanswers.write("{}\n".format(' '.join(pred)))
    acc = acc / len(quests)
    print("Acc:{}".format(acc))


