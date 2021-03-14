from __future__ import print_function, absolute_import
import time

import torch
from torch.nn import init
from torch.autograd import Variable
from torch import nn
from .evaluation_metrics import accuracy
from .loss import TripletLoss
from .utils.meters import AverageMeter
import pdb
import torch.nn.functional as F
import numpy as np


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec

class PairTrainer(object):
    def __init__(self, model, criterion, pair_loader, num_classes):
        super(PairTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.pair_loader = pair_loader
        self.num_classes = num_classes
        self.pair_loader_iter = iter(self.pair_loader)

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train() 

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            try:
                pair_inputs = next(self.pair_loader_iter)
            except:
                self.pair_loader_iter = iter(self.pair_loader)
                pair_inputs = next(self.pair_loader_iter)
            inputs1, inputs2, targets = self._parse_data(pair_inputs)
            loss, prec1, _ = self._forward(inputs1, inputs2, targets)

            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs1, imgs2, pids, _ = inputs
        inputs1 = [Variable(imgs1).cuda()]
        inputs2 = [Variable(imgs2).cuda()]
        targets = Variable(pids.cuda())
        return inputs1, inputs2, targets

    def _forward(self, inputs1, inputs2, targets):
        ###mehotd1########
        #inputs=np.array(inputs1)-np.array(inputs2)
        outputs=self.model(*inputs1,*inputs2)
        ####method2########
        #outputs1 = self.model(*inputs1)
        #outputs2 = self.model(*inputs2)
        #outputs=outputs1-outputs2
        #outputs= torch.cat((outputs1,outputs2),1)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec, outputs

class TripletTrainer(object):
    def __init__(self, model, criterion, triplet_loader, num_classes):
        super(TripletTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.triplet_loader = triplet_loader
        self.num_classes = num_classes
        self.triplet_loader_iter = iter(self.triplet_loader)

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train() 

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            try:
                triplet_inputs = next(self.triplet_loader_iter)
            except:
                self.triplet_loader_iter = iter(self.triplet_loader)
                triplet_inputs = next(self.triplet_loader_iter)
            ###
            inputsA1, inputsA2, inputsA3, inputsB1, inputsB2, inputsC1, num = self._parse_data(triplet_inputs)
            loss = self._forward(inputsA1, inputsA2, inputsA3, inputsB1, inputsB2, inputsC1)
            losses.update(loss.item(), num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        A1, A2, A3, B1, B2, C1 = inputs
        inputsA1 = [Variable(A1).cuda()]
        inputsA2 = [Variable(A2).cuda()]
        inputsA3 = [Variable(A3).cuda()]
        inputsB1 = [Variable(B1).cuda()]
        inputsB2 = [Variable(B2).cuda()]
        inputsC1 = [Variable(C1).cuda()]
        num=len(A1)
        return inputsA1, inputsA2, inputsA3, inputsB1, inputsB2, inputsC1, num

    def _forward(self,inputsA1, inputsA2, inputsA3, inputsB1, inputsB2, inputsC1):
        #####method1############
        inputsA1B1=np.array(inputsA1)-np.array(inputsB1)
        inputsA2B2=np.array(inputsA2)-np.array(inputsB2)
        inputsA3C1=np.array(inputsA3)-np.array(inputsC1)

        outputsA1B1=self.model(*inputsA1B1)
        outputsA2B2=self.model(*inputsA2B2)
        outputsA3C1=self.model(*inputsA3C1)
        ####method2#########
        #outputsA1B1=abs(self.model(*inputsA1)-self.model(*inputsB1))
        #outputsA2B2=abs(self.model(*inputsA2)-self.model(*inputsB2))
        #outputsA3C1=abs(self.model(*inputsA3)-self.model(*inputsC1))
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, torch.nn.TripletMarginLoss):
            loss= self.criterion(outputsA1B1, outputsA2B2, outputsA3C1)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss

class CamStyleTrainer(object):
    def __init__(self, model, criterion, camstyle_loader):
        super(CamStyleTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.camstyle_loader = camstyle_loader
        self.camstyle_loader_iter = iter(self.camstyle_loader)

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            try:
                camstyle_inputs = next(self.camstyle_loader_iter)
            except:
                self.camstyle_loader_iter = iter(self.camstyle_loader)
                camstyle_inputs = next(self.camstyle_loader_iter)
            inputs, targets = self._parse_data(inputs)
            camstyle_inputs, camstyle_targets = self._parse_data(camstyle_inputs)
            loss, prec1 = self._forward(inputs, targets, camstyle_inputs, camstyle_targets)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = Variable(imgs.cuda())
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets, camstyle_inputs, camstyle_targets):
        outputs = self.model(inputs)
        camstyle_outputs = self.model(camstyle_inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        camstyle_loss = self._lsr_loss(camstyle_outputs, camstyle_targets)
        loss += camstyle_loss
        return loss, prec

    def _lsr_loss(self, outputs, targets):
        num_class = outputs.size()[1]
        targets = self._class_to_one_hot(targets.data.cpu(), num_class)
        targets = Variable(targets.cuda())
        outputs = torch.nn.LogSoftmax()(outputs)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def _class_to_one_hot(self, targets, num_class):
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.FloatTensor(targets.size()[0], num_class)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets, 0.9)
        targets_onehot.add_(0.1 / num_class)
        return targets_onehot

