from PIL import Image
import os
import random
import math

def blend_image(width, height,img1,img2):
    img = Image.new('RGB', (width, height))
    im = img.load()
    for y in range(height):
        for x in range(width):
            r,g,b = img1.getpixel((x,y))
            r2,g2,b2 = img2.getpixel((x,y))
            if(r>0 or g>0 or b>0):
                im[x,y] = (r,g,b)
            else:
                im[x,y] = (r+r2,g+g2,b+b2)
    return img


def handle_img(pic_num):
    img_folder = './patch_pic' + str(pic_num)
    imgs = os.listdir(img_folder)
    imgNum = len(imgs)
    print(imgNum)

    for i in range(imgNum):
        img1 = Image.open(img_folder + '/' + imgs[i])
        #img = img1.resize((102, 102))  # 将图片调整到合适大小

        oriImg = Image.open("./result"+str(pic_num)+".png")  # 打开图片
        size = oriImg.size  # 获取图片大小尺寸
        width = size[0]
        height = size[1]
        # oriImg.paste(img, (image[0]-102, image[1]-102))

        oriImg = blend_image(width,height,oriImg,img1)

        oriImg1 = oriImg.convert('RGB')
        oriImg1.save("./result"+str(pic_num)+".png")

if __name__ == '__main__':
    test_pic = [ 11 + 100*i for i in range(0,10)]
    for pic_num in test_pic:
        handle_img(pic_num)