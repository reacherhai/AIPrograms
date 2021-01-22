# Data Aggregate
import numpy as np
import pandas as pd
import sklearn
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

#扩充停用词以进行去噪
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append("think")
stopwords.append("people")
stopwords.append("thing")
stopwords.append("http")
stopwords.append("https")
stopwords.append("youtube")

labeldict = {}

#创建数据集，同时将labels离散化
def createDataset(path):
    data = pd.read_csv(path, header=None)
    data = data.values.tolist()
    labels = []
    dataset = []
    i = 0
    for line in data[1:]:
        if(line[0] not in labeldict.keys()):
            labeldict[line[0]] = i
            labels.append(i)
            i += 1
        else:
            labels.append(labeldict[line[0]])
        dataset.append(line[1])
    return dataset,labels

mydataset, labels = createDataset("mbti_1.csv")
#X = vectorizer.fit_transform(mydataset)

#使用tfidf进行去噪并提取关键特征
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words=stopwords, use_idf=True)
X = tfidf_vectorizer.fit_transform(mydataset)

#使用SVD将特征降维到10维
svd = TruncatedSVD(10)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
mydataset = lsa.fit_transform(X)

#得到最终的dataframe形式的dataset，其最后一列为labels
mydataset = pd.DataFrame(mydataset)
mydataset['labels'] = labels


