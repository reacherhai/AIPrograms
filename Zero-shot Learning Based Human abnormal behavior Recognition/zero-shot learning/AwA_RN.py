import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
import math
import argparse
import random
import os
from sklearn.metrics import accuracy_score
import pdb

parser = argparse.ArgumentParser(description = "Zero Shot Learning")
parser.add_argument("-b","--batch_size",type = int, default = 32)
parser.add_argument("-e","--episode",type = int, default = 50000)
parser.add_argument("-t","--test_episode",type =int, default = 1000)
parser.add_argument("-l","--learning_rate",type = float, default = "1e-5")
parser.add_argument("-g",'--gpu',type = int, default = 0)
args = parser.parse_args()

#Hyper parameters

BATCH_SIZE = args.batch_size
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

class AttributeNetwork(nn.Module):
    def __init__(self,input_size, hidden_size,output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class RelationNetwork(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def main():
    print("init dataset")

    dataroot = './data'
    dataset = 'AWA2_data'
    image_embedding = 'res101'
    class_embedding = 'att'

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + '.mat')
    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1
    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + '_splits.mat')
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

    attribute = matcontent['original_att'].T

    x = feature[trainval_loc] # train_features
    train_label = label[trainval_loc].astype(int) #train_label
    att = attribute[train_label] #train attributes

    x_test = feature[test_unseen_loc]
    test_label = label[test_unseen_loc].astype(int)
    x_test_seen = feature[test_unseen_loc]
    test_label_seen = label[test_seen_loc].astype(int)
    test_id = np.unique(test_label)
    att_pro = attribute[test_id]

    train_features = torch.from_numpy(x)
    print(train_features.shape)

    train_label = torch.from_numpy(train_label).unsqueeze(1)
    print(train_label.shape)

    all_attributes = np.array(attribute)
    print(all_attributes)

    attributes = torch.from_numpy(attribute)

    test_featuers = torch.from_numpy(x_test)
    print(test_featuers.shape)

    test_label = torch.from_numpy(test_label).unsqueeze(1)
    print(test_label.shape)

    testclasses_id = np.array(test_id)
    print(testclasses_id.shape)

    test_attributes = torch.from_numpy(att_pro).float()
    print(test_attributes.shape)

    test_seen_features = torch.from_numpy(x_test_seen)
    print(test_seen_features.shape)

    test_seen_label = torch.from_numpy(test_label_seen)

    train_data = TensorDataset(train_features,train_label)

    print("init networks")
    attribute_network = AttributeNetwork(85,1024,2048)
    relation_network  = RelationNetwork(4096,400)

    attribute_network.cuda(GPU)
    relation_network.cuda(GPU)

    attribute_network_optim  = torch.optim.Adam(attribute_network.parameters(),lr = LEARNING_RATE,weight_decay = 1e-5)
    attribute_network_scheduler = StepLR(attribute_network_optim, step_size = 20000,gamma = 0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr = LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size = 20000, gamma  = 0.5)

    print("training...")
    last_accuracy = 0.0

    for episode in range(EPISODE):
        attribute_network_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)

        batch_features, batch_labels = train_loader.__iter__().next()
        sample_labels = []
        for label in batch_labels.numpy():
            if label not in sample_labels:
                sample_labels.append(label)

        #print(sample_labels)
        sample_attributes = torch.Tensor([all_attributes[i] for i in sample_labels]).squeeze(1)
        class_num = sample_attributes.shape[0]




if __name__ == '__main__':
    main()




