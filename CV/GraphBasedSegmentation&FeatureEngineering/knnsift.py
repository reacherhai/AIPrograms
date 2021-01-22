import cv2
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image, ImageFilter
from RGB import calculate_area_feature
from sklearn.cluster import KMeans

def generate_image(width, height,colors):
    img = Image.new('RGB', (width, height))
    im = img.load()
    for y in range(height):
        for x in range(width):
            comp = y * width + x
            if(comp in colors.keys()):
                im[x, y] = colors[comp]
            else:
                im[x,y] = (0,0,0)
    return img

def generate_source_image(width, height):
    img = Image.new('RGB', (width, height))
    im = img.load()
    for y in range(height):
        for x in range(width):
            im[x,y] = (0,0,0)
    return img

def knn_sift(pic_num):
    pic_name = str(pic_num) +'.png'
    img = cv2.imread('./data/imgs/'+pic_name)
    img1 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

    '''
    cv2.drawKeypoints(gray, keypoints_1, img)
    cv2.drawKeypoints(gray, keypoints_1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.subplot(121), plt.imshow(img),
    plt.title('Dstination'), plt.axis('off')
    plt.subplot(122), plt.imshow(img1),
    plt.title('Dstination'), plt.axis('off')
    plt.show()
    '''

    pca = PCA(n_components=10)

    sift_feature = pca.fit_transform(descriptors_1)

    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    size = image.size
    width = size[0]
    height = size[1]
    #get patch
    area_dict = {}

    point_num = 0
    for keypoint in keypoints_1:
        x,y = keypoint.pt
        x = int(x+0.5)
        y = int(y+0.5)
        keypoint_num = x*width + y
        patch_nums = []
        for j in range(0,17):
            for i in range(0,17):
                a = (x-8+i)
                b = (y-8+j)
                if( a>=0 and a<=width-1 and b>=0 and b<=height-1):
                    patch_nums.append(a + b*width)
        area_dict[point_num] = patch_nums
        point_num += 1


    #gain area feature
    area_feature = calculate_area_feature(image,area_dict,width,4)

    #gain combinded feature
    combind_feature = [[] for i in range(len(keypoints_1))]
    sift_feature = sift_feature.tolist()
    i = 0
    for keypoint in area_feature.keys():
        combind_feature[i] += area_feature[keypoint]
        combind_feature[i] += sift_feature[i]
        i += 1
    combind_feature = np.array(combind_feature)

    #k-means
    data = pd.DataFrame(combind_feature)
    model = KMeans(n_clusters=3, init='k-means++')
    model.fit(data)

    r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
    r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
    r.columns = list(data.columns) + ['class num']  # 重命名表头
    #print(r)

    r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)  # 详细输出每个样本对应的类别
    r.columns = list(data.columns) + ['category']  # 重命名表头

    if(not os.path.exists("patch_pic"+str(pic_num))):
        os.mkdir("patch_pic"+str(pic_num))
    for center_num in area_dict.keys():
        category = r.at[center_num,'category']
        area = area_dict[center_num]
        colors = {}
        for pixel in area:
            color = [0,0,0]
            color[category] = 255
            color = tuple(color)
            colors[pixel] = color
        img = generate_image(width,height,colors)
        img.save('./patch_pic' + str(pic_num)+'/patch'+str(center_num)+".png")

    img = generate_source_image(width,height)
    img.save('./result'+str(pic_num)+'.png')

if __name__ == '__main__':
    test_pic = [ 11 + 100*i for i in range(0,10)]
    for pic_num in test_pic:
        knn_sift(pic_num)