from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms
import torch
from dataUtils import *
import numpy as np

class MyDataSet(Dataset):
    def __init__(self, images_path='./', label_path='./',Type='train'):
        self.images = os.listdir(images_path)
        self.label = os.listdir(label_path)
        self.images_path = images_path
        self.label_path = label_path
        self.Type=Type
        if Type=="train":
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=(80,80),scale=(0.2, 0.4)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(15)
            ])
        else :
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=(80,80), scale=(0.2,0.4)),
                transforms.RandomHorizontalFlip(0.5),
            ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    @staticmethod
    def Binaryzation(x):
        mask = x > 0.5
        x[mask] = 1
        mask = x <= 0.5
        x[mask] = 0
        return x

    def __len__(self):
        return len(self.images)


    def __getitem__(self, item):
        # 添加数据增强方法
        image=LOG_pro(self.images_path + self.images[item])
        image_row=Image.open(self.images_path + self.images[item])
        x = self.test_transforms(image)
        image_row=self.test_transforms(image_row)
        label = Image.open(self.label_path + self.label[item])
        y = self.test_transforms(label)
        y= self.Binaryzation(y)
        if self.Type=="test":
            return torch.cat([image_row,x], dim=0), y
        temp =torch.cat([image_row,x,y], dim=0)
        temp = self.transforms(temp)
        y = temp[x.shape[0]+image_row.shape[0]:, :, :]
        x = temp[:x.shape[0]+image_row.shape[0], :, :]

        if torch.sum(y) == 0:
             return self.__getitem__(item)
        return x, y


def get_loader(batch_size=20,train_path="./",val_path="./",Type="train"):
    set = MyDataSet(train_path,val_path,Type)
    loader = DataLoader(set, batch_size=batch_size, shuffle=True)
    return loader
