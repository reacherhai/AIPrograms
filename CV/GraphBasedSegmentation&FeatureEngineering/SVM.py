from __future__ import division
import pandas as pd
import numpy as np
import copy
import random
import math
from sklearn import svm
from RGB import dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from graph import calc_IOU

def generate_image(forest, width, height,colors):
    #random_color = lambda: (int(random()*255), int(random()*255), int(random()*255))
    #colors = [random_color() for i in range(width*height)]

    img = Image.new('RGB', (width, height))
    im = img.load()
    dict = forest.dict_parent2node()
    for y in range(height):
        for x in range(width):
            comp = forest.find(y * width + x)
            im[x, y] = colors[comp]

    #return img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
    return img

def test_acc():
    pass

def main():
    #dataset 的形式是特征 + labels
    df_train, df_test = dataset()
    #labels = df.columns.values.tolist()

    #获取训练和测试数据
    x_train = np.array(df_train.ix[:,0:df_train.shape[1]-2].values.tolist() )
    y_train = np.array(df_train.ix[:,'labels'].values.tolist() )
    x_test = np.array(df_test.ix[:, 0:df_test.shape[1]-2].values.tolist())
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

main()
