from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import pdb

import torch
import numpy as np

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter

from torch.autograd import Variable
from .utils import to_torch
from .utils import to_numpy
import pdb
import scipy.io as scio
import operator
from functools import reduce
import json

def extract_cnn_feature(model, inputs, output_feature=None):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, volatile=True)
    outputs = model(inputs, output_feature)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=1, output_feature=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
        data_time.update(time.time() - end)
        outputs = extract_cnn_feature(model, imgs, output_feature)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))
    return features, labels


def extract_features_pair(model, data_loader, print_freq=1, output_feature=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    features = {}
    end = time.time()
    for k, (imgs1, imgs2, ids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)
        ######method1############
        imgs=imgs1-imgs2
        outputs = extract_cnn_feature(model, imgs, output_feature)
        #######method2#############
        #outputs1 = extract_cnn_feature(model, imgs1, output_feature)
        #outputs2 = extract_cnn_feature(model, imgs2, output_feature)
        #outputs = outputs1-outputs2

        ids=ids.numpy()
        for i in range(len(outputs)):
            if ids[i] in features:
                features[ids[i]].append(outputs[i])
            else:
                features[ids[i]] = [outputs[i]]
        batch_time.update(time.time() - end)
        end = time.time()
        if (k + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(k + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                             data_time.val, data_time.avg))
    return features

def group_features_pair(query_features,gallery_features,query=None,gallery=None,method=None):
    dismat=np.zeros((len(query_features),len(gallery_features)))
    for key1 in query_features:
        for key2 in gallery_features:
            average_dist=0
            dist=pairwise_distance(query_features[key1],gallery_features[key2])
            m,n=dist.shape
            if method=='min':
                dist=reduce(operator.add, dist)
                average_dist=min(dist)
            if method=='min2':
                dist=reduce(operator.add, dist)
                dist=sorted(dist)
                average_dist=sum(dist[0:2])/2
            if method=='mean':
                if m<=n:
                    indices=np.argsort(dist, axis=1)
                    for i in range(m):
                        id=indices[i][0] ### query id(i)-->gallery id(indices[i][0])
                        average_dist=average_dist+dist[i][indices[i][0]]
                    average_dist=average_dist/m
                if m>n:
                    indices=np.argsort(dist, axis=0)
                    for i in range(n):
                        id=indices[0][i]
                        average_dist=average_dist+dist[id][i]
                    average_dist=average_dist/n
            dismat[key1][key2]=average_dist
    return dismat

def group_features(query_features,gallery_features,query=None,gallery=None,method=None):
    query_d={}
    gallery_d={}
    coor={}
    for fname, _, _, pic in query:
        if pic not in query_d:
            query_d[pic]=[query_features[fname]]
        else:
            query_d[pic].append(query_features[fname])
    for fname, _, _, pic in gallery:
        if pic not in gallery_d:
            gallery_d[pic]=[gallery_features[fname]]
        else:
            gallery_d[pic].append(gallery_features[fname])
    dismat=np.zeros((len(query_d),len(gallery_d)))
    for key1 in query_d:
        for key2 in gallery_d:
            average_dist=0
            dist=pairwise_distance(query_d[key1],gallery_d[key2])
            m,n=dist.shape
            if method=='min':
                dist=reduce(operator.add, dist)
                average_dist=min(dist)
            if method=='min2':
                dist=reduce(operator.add, dist)
                dist=sorted(dist)
                average_dist=sum(dist[0:2])/2
            if method=='mean':
                if m<=n:
                    indices=np.argsort(dist, axis=1)
                    coor[str(key1)+'-'+str(key2)+'A->B']=np.zeros(m)
                    for i in range(m):
                        id=indices[i][0] ### query id(i)-->gallery id(indices[i][0])
                        coor[str(key1)+'-'+str(key2)+'A->B'][i]=id
                        average_dist=average_dist+dist[i][indices[i][0]]
                    average_dist=average_dist/m
                if m>n:
                    indices=np.argsort(dist, axis=0)
                    coor[str(key1)+'-'+str(key2)+'B->A']=np.zeros(n)
                    for i in range(n):
                        id=indices[0][i]
                        coor[str(key1)+'-'+str(key2)+'B->A'][i]=id
                        average_dist=average_dist+dist[id][i]
                    average_dist=average_dist/n
            dismat[key1][key2]=average_dist
    return dismat,coor


def pairwise_distance(query, gallery):
    x = torch.cat([query_features.unsqueeze(0) for query_features in query], 0)
    y = torch.cat([gallery_features.unsqueeze(0) for gallery_features in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all_pair(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):

    if query is not None and gallery is not None:
        query_pics = [pic for _, _, pic, _ in query]
        gallery_pics = [pic for _, _, pic, _ in gallery]
        query_cams = [cam for _, _, _, cam in query]
        gallery_cams = [cam for _, _, _, cam in gallery]
    else:
        assert (query_pics is not None and gallery_pics is not None
                and query_cams is not None and gallery_cams is not None)
    query_pics=list(set(query_pics))
    gallery_pics=list(set(gallery_pics))
    # Compute mean AP
    mAP = mean_ap(distmat, query_pics, gallery_pics, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_pics, gallery_pics,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores['market1501'][k - 1]))
    return cmc_scores['market1501'][0]


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):

    if query is not None and gallery is not None:
        query_pics = [pic for _, _, _, pic in query]
        gallery_pics = [pic for _, _, _, pic in gallery]
        query_ids = [ids for _, _, ids, _ in query]
        gallery_ids = [ids for _, _, ids, _ in gallery]
        query_cams = [cam for _, _, cam, _ in query]
        gallery_cams = [cam for _, _, cam,_ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)
    query_pics=list(set(query_pics))
    gallery_pics=list(set(gallery_pics))

    # Compute mean AP
    mAP = mean_ap(distmat, query_pics, gallery_pics, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_pics, gallery_pics,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores['market1501'][k - 1]))
    return cmc_scores['market1501'][0]


def reranking(query_features, gallery_features, query=None, gallery=None, k1=20, k2=6, lamda_value=0.3):
        x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
        feat = torch.cat((x, y))
        query_num, all_num = x.size(0), feat.size(0)
        feat = feat.view(all_num, -1)

        dist = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
        dist = dist + dist.t()
        dist.addmm_(1, -2, feat, feat.t())

        original_dist = dist.numpy()
        all_num = original_dist.shape[0]
        original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
        V = np.zeros_like(original_dist).astype(np.float16)
        initial_rank = np.argsort(original_dist).astype(np.int32)

        print('starting re_ranking')
        for i in range(all_num):
            # k-reciprocal neighbors
            forward_k_neigh_index = initial_rank[i, :k1 + 1]
            backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
            fi = np.where(backward_k_neigh_index == i)[0]
            k_reciprocal_index = forward_k_neigh_index[fi]
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
                candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                   :int(np.around(k1 / 2)) + 1]
                fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
                candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
        original_dist = original_dist[:query_num, ]
        if k2 != 1:
            V_qe = np.zeros_like(V, dtype=np.float16)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
            V = V_qe
            del V_qe
        del initial_rank
        invIndex = []
        for i in range(all_num):
            invIndex.append(np.where(V[:, i] != 0)[0])

        jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

        for i in range(query_num):
            temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = []
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                                   V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

        final_dist = jaccard_dist * (1 - lamda_value) + original_dist * lamda_value
        del original_dist
        del V
        del jaccard_dist
        final_dist = final_dist[:query_num, query_num:]
        return final_dist

        
class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery, output_feature=None, rerank=False):
        query_features, query_labels = extract_features(self.model, query_loader, 1, output_feature)
        gallery_features, gallery_labels = extract_features(self.model, gallery_loader, 1, output_feature)
        np.save("duke_gallery_f.npy",gallery_features)
        np.save("duke_gallery_l.npy",gallery_labels)
        np.save("duke_query_f.npy",query_features)
        np.save("duke_query_l.npy",query_labels)
        if rerank:
            distmat = reranking(query_features, gallery_features, query, gallery)
        else:
            distmat,coor = group_features(query_features, gallery_features, query, gallery,method='mean')
            indices = np.argsort(distmat, axis=1)
        return evaluate_all(distmat, query=query, gallery=gallery)

class EvaluatorRelations(object):
    def __init__(self, model):
        super(EvaluatorRelations, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery, output_feature=None, rerank=False):
        save1='pair_data/ranking3.mat'
        save2='pair_data/dist3.mat'
        query_features = extract_features_pair(self.model, query_loader, 1, output_feature)
        gallery_features = extract_features_pair(self.model, gallery_loader, 1, output_feature)
        if rerank:
            distmat = reranking(query_features, gallery_features, query, gallery)
        else:
            distmat = group_features_pair(query_features, gallery_features, query, gallery,method='mean')
            indices = np.argsort(distmat, axis=1)           
            scio.savemat(save1,{'ranking':indices})
            scio.savemat(save2,{'dist':distmat})
        return evaluate_all_pair(distmat, query=query, gallery=gallery)
