from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re
import itertools
import random

class market1501(object):

    def __init__(self, root, trainer):
        self.trainer = trainer
        self.images_dir = osp.join(root)
        self.train_path = 'train'
        self.gallery_path = 'gallery'
        self.query_path = 'query'
        self.pair_path = 'train'
        self.triplet_path = 'train'
        self.train,self.query, self.gallery, self.pair, self.triplet = [], [], [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids, self.num_pair_ids, self.num_triplet_ids = 0, 0, 0, 0, 0
        self.load()

    def preprocess(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fname, pid, cam,0))
        return ret, int(len(all_pids))
    
    def preprocess_pair(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        imgs_dict = {}
        combine={}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, _ = map(int, pattern.search(fname).groups())
            if pid not in imgs_dict:
                imgs_dict[pid] = [fname]
            else:
                imgs_dict[pid].append(fname)

        s=[]
        for key in imgs_dict:
            if len(imgs_dict[key])>=14 and len(imgs_dict[key])<=16:
                s.append(key)
        ids=0
        #f=open('/home/nvlab/groupre-id/id.txt','r')
        #id1s=[]
        #id2s=[]
        #pattern=re.compile(r'(\d{1,4})\+(\d{1,4})')
        #datas=f.readlines()
        #for data in datas:
        #    id1,id2=map(int, pattern.search(data).groups())
        #    id1s.append(id1)
        #    id2s.append(id2)
        for key1 in s:
            for key2 in s:
        #for key1,key2 in zip(id1s,id2s):
                if key1<key2:
                    combine[ids]=list(itertools.product(imgs_dict[key1],imgs_dict[key2]))
                    ids=ids+1

        for key in combine:
            for item in combine[key]:
                ret.append((item[0],item[1],key, 0))
        random.shuffle(ret)
        return ret,int(len(combine))

    def preprocess_triplet(self, path, num_classes):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        imgs_dict = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, _ = map(int, pattern.search(fname).groups())
            if pid not in imgs_dict:
                imgs_dict[pid] = [fname]
            else:
                imgs_dict[pid].append(fname)
        s=[]
        for key in imgs_dict:
            if len(imgs_dict[key])>=14 and len(imgs_dict[key])<=16:
                s.append(key)

        #########sample select#######
        #ids=0
        #for key1 in s[:50]:
        #    for key2 in s[:50]:
        #        if key1<key2:
        #            combine[ids]=list(itertools.product(imgs_dict[key1],imgs_dict[key2]))
        #            ids=ids+1
        for i in range(0,num_classes):
            for j in range(0,num_classes):
                if i!=j:
                    for k in range(0,num_classes):
                        if k!=i and k!=j:
                            for t in range(0,len(imgs_dict[s[k]])):
                                A=random.sample(range(0,len(imgs_dict[s[i]])-1),3)
                                B=random.sample(range(0,len(imgs_dict[s[j]])-1),2)
                                #A1=random.randint(0,len(imgs_dict[s[i]])-1)
                                #A2=random.randint(0,len(imgs_dict[s[i]])-1)
                                #A3=random.randint(0,len(imgs_dict[s[i]])-1)
                                #B1=random.randint(0,len(imgs_dict[s[j]])-1)
                                #B2=random.randint(0,len(imgs_dict[s[j]])-1)
                                C1=t
                                ret.append((imgs_dict[s[i]][A[0]],imgs_dict[s[i]][A[1]],imgs_dict[s[i]][A[2]],imgs_dict[s[j]][B[0]],imgs_dict[s[j]][B[1]],imgs_dict[s[k]][C1]))
        random.shuffle(ret)
        return ret, num_classes


    def preprocess_pair_group(self, path, cam):
        pattern = re.compile(r'pic([-\d]+)cam([-\d]+)_(\d+)')
        imgs_dict = {}
        pair_dict = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.bmp')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pic, _, _= map(int, pattern.search(fname).groups())
            if pic not in imgs_dict:
                imgs_dict[pic] = [fname]
            else:
                imgs_dict[pic].append(fname)

        for key in imgs_dict:
            pair_dict[key] = list(itertools.product(imgs_dict[key],imgs_dict[key]))
            for i in range(0,len(pair_dict[key])):
                if pair_dict[key][i][0]!=pair_dict[key][i][1]:
                    ret.append((pair_dict[key][i][0], pair_dict[key][i][1], key, cam))
        return ret,len(imgs_dict)

                        
    def preprocess_group(self, path, relabel=True):
        pattern = re.compile(r'pic([-\d]+)cam([-\d]+)_(\d+)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.bmp')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pic, cam, pid = map(int, pattern.search(fname).groups())
            pid=int(str(pic)+str(pid))
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            #pid = all_pids[pid]
            #cam -= 1
            ret.append((fname, pid, cam,pic))
        return ret, int(len(all_pids))


    def load(self):
        if self.trainer==0:
            self.train, self.num_train_ids = self.preprocess(self.train_path)
            self.gallery, self.num_gallery_ids = self.preprocess_group(self.gallery_path, 1)
            self.query, self.num_query_ids = self.preprocess_group(self.query_path, 2)
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                .format(self.num_train_ids, len(self.train)))
            print("  query    | {:5d} | {:8d}"
                .format(self.num_query_ids, len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                .format(self.num_gallery_ids, len(self.gallery)))
        if self.trainer==1:
            self.pair, self.num_pair_ids = self.preprocess_pair(self.pair_path)
            self.gallery, self.num_gallery_ids = self.preprocess_pair_group(self.gallery_path, 1)
            self.query, self.num_query_ids = self.preprocess_pair_group(self.query_path, 2)
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                .format(self.num_pair_ids, len(self.pair)))
            print("  query    | {:5d} | {:8d}"
                .format(self.num_query_ids, len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                .format(self.num_gallery_ids, len(self.gallery)))
        if self.trainer==2:
            self.triplet, self.num_triplet_ids = self.preprocess_triplet(self.triplet_path,20 )
            self.gallery, self.num_gallery_ids = self.preprocess_pair_group(self.gallery_path, 1)
            self.query, self.num_query_ids = self.preprocess_pair_group(self.query_path, 2)
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                .format(self.num_triplet_ids, len(self.triplet)))
            print("  query    | {:5d} | {:8d}"
                .format(self.num_query_ids, len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                .format(self.num_gallery_ids, len(self.gallery)))

