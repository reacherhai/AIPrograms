
import numpy
import pandas

with open('human.txt','w',encoding="utf-8") as f:
    for i in range(400,651):
        string = "fight_0_" + str(i) + ".jpg"
        f.write(string+" "+str(1)+"\n")
    for  i in range(900,1501):
        string = "fight_0_" + str(i) + ".jpg"
        f.write(string+" "+str(0)+"\n")
    for i in range(2600,2801):
        string = "fight_0_" + str(i) + ".jpg"
        f.write(string+" "+str(1)+"\n")
    for i in range(3820,3921):
        string = "fight_0_" + str(i) + ".jpg"
        f.write(string+" "+str(1)+"\n")
    for i in range(4200,5031):
        string = "fight_0_" + str(i) + ".jpg"
        f.write(string+" "+str(0)+"\n")
    for i in range(1,801):
        string = "fight_1_" + str(i) + ".jpg"
        f.write(string+" "+str(0)+"\n")
    for i in range(1000,1601):
        string = "fight_1_" + str(i) + ".jpg"
        f.write(string+" "+str(1)+"\n")
    for i in range(1920,2041):
        string = "fight_1_" + str(i) + ".jpg"
        f.write(string+" "+str(1)+"\n")
    for i in range(2180,2621):
        string = "fight_1_" + str(i) + ".jpg"
        f.write(string+" "+str(1)+"\n")
    for i in range(3000,3580):
        string = "fight_1_" + str(i) + ".jpg"
        f.write(string+" "+str(0)+"\n")
    print("labels writing complete")



from PIL import Image

label = []
imgs = []
root ="./data/"
with open("human.txt",'r',encoding="utf-8") as f:
    for line in f:
        line = line.rstrip()
        words = line.split(" ")
        imgs.append( words[0])
        label.append(int(words[1]))
        fn = words[0]
        img = Image.open(root+fn).convert('RGB')
        #print(img)

print("test finished")
#print(imgs)
#print(label)

