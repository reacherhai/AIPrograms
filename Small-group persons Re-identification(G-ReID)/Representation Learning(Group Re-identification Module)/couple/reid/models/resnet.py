from __future__ import absolute_import
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
import numpy as np


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,trainer=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.trainer=trainer

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        #if self.trainer==1:
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                #self.model1= nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.BatchNorm1d(self.num_features),nn.ReLU())
                #init.kaiming_normal(self.model[0].weight.data, mode='fan_out')
                #init.constant(self.model[0].bias.data ,0)
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)
            #if self.num_classes > 0 and self.trainer==1:
            #    self.pairclassifier = nn.Linear(self.num_features, self.num_classes)
            #    init.normal(self.pairclassifier.weight, std=0.001)
            #    init.constant(self.pairclassifier.bias, 0)


        if not self.pretrained:
            self.reset_params()

    def forward(self, x1, x2, output_feature=None):
    #def forward(self, x, output_feature=None):
        name1=['conv1']
        name2=['bn1']#,'relu','maxpool','layer1','layer2','layer3','layer4']
        for name, module in self.base._modules.items():
            if name=='avgpool':
                break
            if name in name1:
               x1=module(x1)
               x2=module(x2)
            elif name in name2:
                x=x1-x2
                #x=torch.cat((x1,x2),1)
                x=module(x)
            else:
                x=module(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        if output_feature == 'pool5':
            x = F.normalize(x)
            return x
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
