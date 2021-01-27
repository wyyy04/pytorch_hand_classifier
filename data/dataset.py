# -*- coding:utf-8 -*-
'''
COCO motivation dataset loader
loading images and labels
'''

import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# dir of data:
DataDir = 'D:\data'

# file name
Datasetfile = 'dataset.txt'
Embeddingfile = 'skipthoughts.npz'
Clusterfile = 'clusters.npz'
Clustersnamefile = 'clustersname.txt'
Clusterfile_256 = 'clusters_256.npz'
# image file dir
ImageDir = 'COCO_motivations_clean'


class COCO_motivations_Dataset(data.Dataset):

    def __init__(self, root=None, transforms=None, train=True):
        '''
        Get images, divide into train/test set
        '''

        self.train = train
        if root is not None:
            self.images_root = os.path.join(root, ImageDir)
        else:
            self.images_root = os.path.join(DataDir, ImageDir)

        self.Loader = COCO_Dataloader(root=root)

        self._get_dataset()

        if transforms is None:
            # normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
            #                         std=[0.229, 0.224, 0.225])
            normalize = T.Normalize(mean=[0.395, 0.393, 0.389],
                                    std=[0.272, 0.271, 0.272])

            if not train:
                self.transforms = T.Compose([
                    T.Resize((224, 224)),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize((256, 256)),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def _get_dataset(self):
        self.images_path = []
        self.images_labels = []

        if self.train:
            self.images_path, self.images_labels = self.Loader.get_trainset()
        else:
            self.images_path, self.images_labels = self.Loader.get_testset()

    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        img_path = os.path.join(self.images_root, self.images_path[index])
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        label = self.images_labels[index]
        return data, int(label)

    def __len__(self):
        return len(self.images_path)


class COCO_Dataloader():
    def __init__(self, root=None):
        if root is not None:
            self.root = root
        else:
            self.root = DataDir
        self.textloader = Read_text(root=root)
        self.npzloader = Read_npz(root=root)
        self.trainimage, self.testimage, self.trainlabel, self.testlabel = self.load_data()

    def get_dataset(self):
        return self.trainimage, self.testimage, self.trainlabel, self.testlabel

    def get_trainset(self):
        return self.trainimage, self.trainlabel

    def get_testset(self):
        return self.testimage, self.testlabel

    def get_trainlabels(self):
        return self.testlabel

    def load_data(self):
        trainIndex, testIndex = self.textloader.loadDataset_TrainTest_Index()

        img_name = self.textloader.loadDataset_IMGname().squeeze(-1)
        train_img_name = img_name[trainIndex]
        test_img_name = img_name[testIndex]

        labels = self.npzloader.loadMotivationClusters()
        trainlabel = labels[trainIndex]
        testlabel = labels[testIndex]

        return train_img_name, test_img_name, trainlabel, testlabel


class Read_npz():
    def __init__(self, root=None):
        if root is not None:
            self.root = root
        else:
            self.root = DataDir

    # 读取npz
    def loadnpzData(self, file):
        data = np.load(file)
        return data

    # 读取skipthoughts.npz
    def loadMotivationEmbedding(self):
        file = os.path.join(self.root, Embeddingfile)
        embedding = self.loadnpzData(file)
        data = np.array(embedding['m'])
        # (10191, 4800)
        return data

    def loadMotivationClusters(self):  # labels
        file = os.path.join(self.root, Clusterfile)
        embedding = self.loadnpzData(file)
        data = np.array(embedding['result'])
        # (10191,)
        return data


class Read_text():
    def __init__(self, root=None):
        if root is not None:
            self.root = root
        else:
            self.root = DataDir

    # 读取txt到list
    def loadtxt(self, file):
        f = open(file, 'r')
        sourceInLine = f.readlines()
        dataset = []
        for line in sourceInLine:
            temp1 = line.strip('\n')
            temp2 = temp1.split('\t')
            dataset.append(temp2)
        f.close()
        return dataset

    # 读取dataset.txt
    def loadDataset(self):
        file = os.path.join(self.root, Datasetfile)
        dataset = self.loadtxt(file)
        dataset = np.array(dataset)
        return dataset

    # 读取clustersname.txt
    def loadClustersname(self):
        file = os.path.join(self.root, Clustersnamefile)
        dataset = self.loadtxt(file)
        dataset = np.array(dataset)
        return dataset

    def loadDataset_i_m(self):
        data = self.loadDataset()
        data = pd.DataFrame(data)
        data.columns = ['image', 'action', 'motivation', 'scene', 'traintest']
        data = np.array(data)
        return data

    def loadDataset_TrainTest_Index(self):
        data = self.loadDataset()
        data = pd.DataFrame(data)
        data.columns = ['image', 'action', 'motivation', 'scene', 'traintest']

        trainIndex = [i for i, x in enumerate(list(data['traintest'])) if x == 'train']
        testIndex = [i for i, x in enumerate(list(data['traintest'])) if x == 'test']

        return trainIndex, testIndex

    def loadDataset_IMGname(self):
        data = self.loadDataset()
        data = pd.DataFrame(data)
        data.columns = ['image', 'action', 'motivation', 'scene', 'traintest']

        img = data.loc[:, ['image']]
        img = np.array(img)

        return img


if __name__ == '__main__':
    # 功能测试
    # D = COCO_Dataloader()
    cocom_train_data = COCO_motivations_Dataset(root='D:\data', train=True)
    cocom_test_data = COCO_motivations_Dataset(train=False)
    cocom_train = DataLoader(cocom_train_data, batch_size=32, shuffle=True)
    cocom_test = DataLoader(cocom_test_data, batch_size=32, shuffle=True)

    x, label = iter(cocom_train).next()  # 迭代器，一条一条的获取数据
    print("x, label", x, x.shape, x.type(), label, label.shape, label.type())

    for i, (x, label) in enumerate(cocom_test):
        print(i, ":  ", x.shape, x.type(), label.shape, label.type())
