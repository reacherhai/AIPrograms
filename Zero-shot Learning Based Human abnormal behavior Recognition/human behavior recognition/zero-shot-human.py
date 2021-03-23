from __future__ import print_function
# 导入相应的包
import torch
from torch import nn, optim
import torch.nn.functional as f
import torch.utils.data as data

#implement here
import loadData
import readAttribute
import numpy as np
# 搭建网络
class FModule(nn.Module):
    def __init__(self, L1in, current, L2out):
        super(FModule, self).__init__()
        self.l1 = nn.Linear(L1in, current)
        self.tan = nn.Tanh()
        self.weight = torch.randn(current, L2out, requires_grad=True)

    def forward(self, xin, pin):
        f = self.l1(xin)
        f = self.tan(f)
        m = torch.mm(f, self.weight)
        pm = torch.mm(m, pin)
        return pm


# 自定义损失函数

class lossmoudle(nn.Module):
    def forward(self, pm, inI, R, batch, classes):
        t = torch.zeros(batch, classes)
        for i in range(batch):
            for j in range(classes):
                l = torch.zeros(2)
                l[1] = R - inI[i, j] * pm[i, j]
                t[i, j] = torch.max(l)
        return t


# 本地数据批次导入

class MyDataset(data.Dataset):
    def __init__(self, dataMat, labels):
        self.dataMat = dataMat
        self.labels = labels

    def __getitem__(self, index):
        dataArr, label = self.dataMat[index], self.labels[index]
        return dataArr, label

    def __len__(self):
        return len(self.dataMat)


# 超参数定义
train_batch = 50
epoch = 300
learning_rate = 0.01
selectList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 24]

# 获取对应的判断矩阵（判断标签和数据是否匹配。匹配为1否则为-1）

def getinI(la):
    reArr = torch.ones(train_batch, 18, dtype=torch.float32)
    reArr = reArr * (-1)
    label = la.detach().numpy()
    for i in range(train_batch):
        y = label[i]
        index = selectList.index(y)
        reArr[i, index] = 1.0
    return reArr

module1 = FModule(60, 80, 42)  # 实例化模型
#implement here
dataList, labels = loadData.getData('./subject102.dat')  # 导入数据

# print(type(data))
dataArr = torch.from_numpy(dataList)  # 转化为pytorch能执行处理的tensor类型
labelArr = torch.from_numpy(labels)
train_loader = torch.utils.data.DataLoader(MyDataset(dataArr, labelArr), batch_size=train_batch, shuffle=True,
                                           drop_last=True)  # 构建批次
testdata, testlabels = loadData.getData('./subject106.dat')
testArr = torch.from_numpy(testdata)
testLArr = torch.from_numpy(testlabels)

test_loader = torch.utils.data.DataLoader(MyDataset(testArr, testLArr), batch_size=train_batch, shuffle=True,
                                          drop_last=True)
f = readAttribute.read_excel()  # 导入语义矩阵
#implement here
f = torch.from_numpy(f)
f = f.float()
print("f type is :", type(f))
print("f len is :", len(f))

optimizer = optim.SGD(module1.parameters(), lr=learning_rate)  # 实例化优化函数
lossmodel = lossmoudle()  # 实例化损失函数

for j in range(epoch):
    loss_runn = 0.0
    Acc_rate = 0.0
    for i, data in enumerate(train_loader, 0):
        da, la = data
        inI = getinI(la)
        out = module1(da.float(), f)
        Kin = out.detach().numpy()
        # print(Kin)

        for x in range(50):
            print("Kin is :", Kin[x])
            result = np.max(Kin[x])
            index = np.where(Kin[x] == result)
            TrueIndex = selectList[np.int(index[0])]
            print(result, "+", TrueIndex)
            if (TrueIndex == la[x].numpy()):
                Acc_rate += 1.0
        print('roch={},Acc_rate={}'.format((i + 1), Acc_rate))
        print('acc_rate:{:.6f}'.format(Acc_rate / ((i + 1) * train_batch)))

        err = lossmodel(out, inI, 10, train_batch, 18)
        optimizer.zero_grad()
        loss = err.mean()
        loss.backward()
        optimizer.step()
        loss_runn += loss.data * da.size(0)

        print('[{}/{}] Loss: {:.6f}'.format(i + 1, 52, loss_runn / (train_batch * (i + 1))))

for i, data in enumerate(test_loader, 0):
    da, la = data
    inI = getinI(la)
    out = module1(da.float(), f)
    Kin = out.detach().numpy()
    # print(Kin)

    for x in range(50):
        # print("Kin is :",Kin[x])
        result = np.max(Kin[x])
        index = np.where(Kin[x] == result)
        TrueIndex = selectList[np.int(index[0])]
        print(result, "+", TrueIndex)
        if (TrueIndex == la[x].numpy()):
            Acc_rate += 1.0
    print('text roch={},Acc_rate={}'.format((i + 1), Acc_rate))
    print('text acc_rate:{:.6f}'.format(Acc_rate / ((i + 1) * train_batch)))
