from __future__ import division
import pandas as pd
import numpy as np
import copy
import random
import math
from sklearn import svm
from preDeal import mydataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def main():
    #dataset 的形式是特征 + labels
    df = mydataset
    df_train, df_test = train_test_split(df, test_size=0.2)
    labels = df.columns.values.tolist()

    #获取训练和测试数据
    x_train = np.array(df_train.ix[:,0:df.shape[1]-2].values.tolist() )
    y_train = np.array(df_train.ix[:,'labels'].values.tolist() )
    x_test = np.array(df_test.ix[:, 0:df.shape[1]-2].values.tolist())
    y_test = np.array(df_test.ix[:, 'labels'].values.tolist())
    #训练SVM 模型
    clf = svm.SVC(C=0.8, kernel='linear', gamma=15, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    #验证模型，取得分数
    print ("训练集精度：",clf.score(x_train, y_train))  # 精度
    y_pred = clf.predict(x_train)
    print ("测试集精度：",clf.score(x_test, y_test))
    y_pred = clf.predict(x_test)
    print("F1 score:",f1_score(y_test, y_pred, average='macro'))

    #线性回归
    from sklearn import datasets, linear_model
    regr = linear_model.LogisticRegression()
    regr.fit(x_train, y_train)
    print("测试集精度(linear)：",regr.score(x_test,y_test))
    y_pred = regr.predict(x_test)
    print("F1 score:", f1_score(y_test, y_pred, average='macro'))

    #随机森林
    from sklearn.ensemble import RandomForestClassifier
    rdf = RandomForestClassifier(max_depth=2, random_state=0)
    rdf.fit(x_train, y_train)
    print("测试集精度(random forest)：",rdf.score(x_test,y_test))
    y_pred = rdf.predict(x_test)
    print("F1 score:", f1_score(y_test, y_pred, average='macro'))


main()
