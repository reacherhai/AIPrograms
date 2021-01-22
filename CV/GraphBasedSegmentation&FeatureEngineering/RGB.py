import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from sklearn import preprocessing
from graph import get_forest
import os
import pandas as pd
from sklearn.decomposition import PCA

min_max_scaler = preprocessing.MinMaxScaler()

def get_feature(Rhist,Ghist,Bhist):
    list = []
    for i in Rhist:
        for j in Ghist:
            for k in Bhist:
                list.append(i[0]+j[0]+k[0])
    return list


def calculate_area_feature(image,area_dict,width,Bins = 8):
    area_feature = {}
    areahist = {}
    #calculate area hists and get RGB features
    for parent in area_dict.keys():
        area = area_dict[parent]
        areahist[parent] = [[ [0.] for i in range(Bins)] for j in range(3)]
        area_feature [parent] = []
        for pixel in area:
            i = pixel % width
            j = (pixel - i) / width
            r,g,b = image.getpixel((i,j))
            areahist[parent][0][r // (256 // Bins)][0] += 1
            areahist[parent][1][g // (256 // Bins)][0] += 1
            areahist[parent][2][b // (256 // Bins)][0] += 1
        #print(areahist[parent])
        normal_r = min_max_scaler.fit_transform(np.array(areahist[parent][0]))
        normal_g = min_max_scaler.fit_transform(np.array(areahist[parent][1]))
        normal_b = min_max_scaler.fit_transform(np.array(areahist[parent][2]))
        areahist[parent][0] = normal_r
        areahist[parent][1] = normal_g
        areahist[parent][2] = normal_b

        area_feature[parent] = get_feature(normal_r,normal_g,normal_b)
        #arr = np.array(area_feature[parent])
    return area_feature

gt_folder = "./data/gt/"
img_folder = './data/imgs/'


def gain_dataset(imagename,sigma = 0.5 ,K = 200, min_size = 20):
    img_path = os.path.join(img_folder, imagename)
    original_img = cv2.imread(img_path)
    img = cv2.resize(original_img, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    b, g, r = cv2.split(img)

    # get the global hist
    min_max_scaler = preprocessing.MinMaxScaler()
    histb = cv2.calcHist([b], [0], None, [8], [0.0, 255.0])
    normal_b = min_max_scaler.fit_transform(histb)

    histg = cv2.calcHist([g], [0], None, [8], [0.0, 255.0])
    normal_g = min_max_scaler.fit_transform(histg)

    histr = cv2.calcHist([r], [0], None, [8], [0.0, 255.0])
    normal_r = min_max_scaler.fit_transform(histr)

    global_feature = get_feature(normal_r, normal_g, normal_b)

    #get area forest by algorithm 1
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    size = image.size
    width = size[0]
    height = size[1]
    forest = get_forest(image,sigma,K,min_size)
    area_dict = forest.dict_parent2node()

    area_feature = calculate_area_feature(image,area_dict,width)

    #combind global and area feature
    combind_feature = {}
    for parent in area_feature:
        combind_feature[parent] = []
        combind_feature[parent].append(area_feature[parent])
        combind_feature[parent].append(global_feature)
        combind_feature[parent] = np.array(combind_feature[parent]).reshape(1,1024)

    gt_path = os.path.join(gt_folder, imagename)
    gt_file = Image.open(gt_path)

    #get labels of areas
    counter = {}
    labels = {}
    for key in area_dict.keys():
        counter[key] = 0
        labels[key] = 0
    for parent in area_dict.keys():
        area = area_dict[parent]
        for number in area:
            i = number % width
            j = (number - i) / width
            gr, gg, gb = gt_file.getpixel((i, j))
            if(gr<=128 and gg<=128 and gb<=128):
                counter[parent] += 1
    for parent in area_dict.keys():
        if(counter[parent] >= len(area_dict[parent])/2):
            labels[parent] = 0
        else:
            labels[parent] = 1

    #acquire dataset
    mydataset = []
    for parent in area_dict.keys():
        row = combind_feature[parent][0].tolist()
        mydataset.append(row)

    labels = list(labels.values())
    mydataset = pd.DataFrame(mydataset)
    #mydataset['labels'] = labels

    return mydataset,labels

    #print(combind_feature[1004])

def dataset():
    testlist = [str(11 + (i * 100)) + ".png" for i in range(1, 10)]
    trainlist = [str(i) + ".png" for i in range(2,10) if i % 100 != 11 ]

    pca = PCA(n_components=50)

    #train dataset
    train_dataset,train_labels = gain_dataset("1.png")

    for train_pic in trainlist:
        dataset,pic_labels = gain_dataset(train_pic)
        train_dataset = train_dataset.append(dataset)
        train_labels += pic_labels
    train_dataset = pca.fit_transform(train_dataset)
    train_dataset = pd.DataFrame(train_dataset)
    train_dataset['labels'] = train_labels

    #test dataset
    test_dataset,test_labels = gain_dataset("11.png")

    for test_pic in testlist:
        dataset,pic_labels = gain_dataset(test_pic)
        test_dataset = test_dataset.append(dataset)
        test_labels += pic_labels
    test_dataset = pca.fit_transform(test_dataset)
    test_dataset = pd.DataFrame(test_dataset)
    test_dataset['labels'] = test_labels

    #print(train_dataset)
    #print(test_dataset)
    return train_dataset,test_dataset





