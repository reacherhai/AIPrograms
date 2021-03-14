from __future__ import absolute_import
import os.path as osp

from PIL import Image


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid, pic = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid, pic

class Preprocessor_Pair(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor_Pair, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname1, fname2, ids, cam = self.dataset[index]
        fpath1 = fname1
        fpath2 = fname2
        if self.root is not None:
            fpath1 = osp.join(self.root, fname1)
            fpath2 = osp.join(self.root, fname2)
        img1 = Image.open(fpath1).convert('RGB')
        img2 = Image.open(fpath2).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, ids, cam

class Preprocessor_Triplet(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor_Triplet, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        A1, A2, A3, B1, B2, C1 = self.dataset[index]
        if self.root is not None:
            A1 = osp.join(self.root, A1)
            A2 = osp.join(self.root, A2)
            A3 = osp.join(self.root, A3)
            B1 = osp.join(self.root, B1)
            B2 = osp.join(self.root, B2)
            C1 = osp.join(self.root, C1)
        imgA1 = Image.open(A1).convert('RGB')
        imgA2 = Image.open(A2).convert('RGB')
        imgA3 = Image.open(A3).convert('RGB')
        imgB1 = Image.open(B1).convert('RGB')
        imgB2 = Image.open(B2).convert('RGB')
        imgC1 = Image.open(C1).convert('RGB')
        if self.transform is not None:
            imgA1 = self.transform(imgA1)
            imgA2 = self.transform(imgA2)
            imgA3 = self.transform(imgA3)
            imgB1 = self.transform(imgB1)
            imgB2 = self.transform(imgB2)
            imgC1 = self.transform(imgC1)
        return imgA1, imgA2, imgA3, imgB1, imgB2, imgC1
