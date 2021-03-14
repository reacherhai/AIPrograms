from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.trainers import Trainer, PairTrainer, TripletTrainer
from reid.evaluators import Evaluator, EvaluatorRelations
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.preprocessor import Preprocessor_Pair
from reid.utils.data.preprocessor import Preprocessor_Triplet
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
import pdb


def get_data(dataname, data_dir, height, width, batch_size, trainer, re=0, workers=8):
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root, trainer)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=re),
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])


    
    if trainer <= 0:
        num_classes = dataset.num_train_ids
        train_loader = DataLoader(
            Preprocessor(dataset.train, root=osp.join(dataset.images_dir, dataset.train_path),transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)
        query_loader = DataLoader(
            Preprocessor(dataset.query,root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=False, pin_memory=True)

        gallery_loader = DataLoader(
            Preprocessor(dataset.gallery,root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=False, pin_memory=True)
        return dataset, num_classes, train_loader, query_loader, gallery_loader

    elif trainer==1:
        num_classes = dataset.num_pair_ids
        pair_loader = DataLoader(
            Preprocessor_Pair(dataset.pair, root=osp.join(dataset.images_dir, dataset.pair_path),
                         transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)
        query_loader = DataLoader(
            Preprocessor_Pair(dataset.query,root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=False, pin_memory=True)
        gallery_loader = DataLoader(
            Preprocessor_Pair(dataset.gallery,root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=False, pin_memory=True)
        return dataset, num_classes, pair_loader, query_loader, gallery_loader

    elif trainer==2:
        num_classes = dataset.num_triplet_ids
        triplet_loader = DataLoader(
            Preprocessor_Triplet(dataset.triplet, root=osp.join(dataset.images_dir, dataset.triplet_path),
                         transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)
        query_loader = DataLoader(
            Preprocessor_Pair(dataset.query,root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=False, pin_memory=True)
        gallery_loader = DataLoader(
            Preprocessor_Pair(dataset.gallery,root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=False, pin_memory=True)
        return dataset, num_classes, triplet_loader, query_loader, gallery_loader




def main(args):
    cudnn.benchmark = True
    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    dataset, num_classes, train_loader, query_loader, gallery_loader = \
        get_data(args.dataset, args.data_dir, args.height,
                 args.width, args.batch_size, args.trainer, args.re, args.workers)
    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)

    # Load from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))
    model = nn.DataParallel(model).cuda()

    #Evaluator
    if args.trainer==0:
        evaluator = Evaluator(model)
        if args.evaluate:
            print("Test:")
            evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)
            return
    if args.trainer==1:
        evaluator = EvaluatorRelations(model)
        if args.evaluate:
            print("Test:")
            evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)
            #return
    if args.trainer==2:
        evaluator = EvaluatorRelations(model)
        if args.evaluate:
            print("Test:")
            evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)
            return
    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    base_param_ids = set(map(id, model.module.base.parameters()))
    new_params = [p for p in model.parameters() if
                    id(p) not in base_param_ids]
    param_groups = [
        {'params': model.module.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]

    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    #optimizer = torch.optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.99))

    # Trainer
    if args.trainer == 0:
        trainer = Trainer(model, criterion)
    if args.trainer == 1:
        #checkpoint = load_checkpoint('/home/nvlab/groupre-id/CamStyle/logs/market1501duke20/checkpoint.pth.tar')
        #model_dict=model.state_dict()
        #checkpoint = {k:v for k,v in checkpoint.items() if k in model_dict}
        #model_dict.update(checkpoint)
        #model.load_state_dict(model_dict)
        trainer = PairTrainer(model, criterion, train_loader,num_classes=num_classes)
    if args.trainer == 2:
        #checkpoint = load_checkpoint('/home/nvlab/groupre-id/CamStyle/logs/market1501duke20/checkpoint.pth.tar')
        #model_dict=model.state_dict()
        #checkpoint = {k:v for k,v in checkpoint.items() if k in model_dict}
        #model_dict.update(checkpoint)
        #model.load_state_dict(model_dict)
        criterion = nn.TripletMarginLoss(margin=100.0, p=2).cuda()
        trainer = TripletTrainer(model, criterion, train_loader,num_classes=num_classes)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 40
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        if args.trainer == 0:
            trainer.train(epoch, train_loader, optimizer)
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
            }, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d} \n'.
                format(epoch))
            if epoch==args.epochs-1:
                evaluator = Evaluator(model)
                print('Test with best model:')
                evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)

        if args.trainer == 1:
            trainer.train(epoch, train_loader, optimizer)
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
            }, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d} \n'.
                format(epoch))
            if epoch%1==0:
                evaluator = EvaluatorRelations(model)
                print('Test with best model:')
                evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)

        if args.trainer == 2:
            trainer.train(epoch, train_loader, optimizer)
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
            }, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d} \n'.
                format(epoch))
            if epoch==args.epochs-1:
                evaluator = EvaluatorRelations(model)
                print('Test with best model:')
                evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Group Re-id")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=24)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=27)#27 and 3
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    #random erasing
    parser.add_argument('--re', type=float, default=0)
    # method for training batchsize
    parser.add_argument('--trainer', type=int, default=0,help="0-none,1-concate,2-triplet")
    #  perform re-ranking
    parser.add_argument('--rerank', action='store_true', help="perform re-ranking")

    main(parser.parse_args())
