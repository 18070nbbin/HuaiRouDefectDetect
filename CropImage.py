import os
import random

from PIL import Image
from torchvision import transforms
import torch
import numpy as np


def Binaryzation(x):
    mask = x > 0.5
    x[mask] = 1
    mask = x <= 0.5
    x[mask] = 0
    return x


def CropImage(image_Path='./',label_Path='./'):

    randomCrop=transforms.RandomResizedCrop(size=(112,112),scale=(0.02,0.03))
    toTensor=transforms.ToTensor()

    image=Image.open(image_Path)
    label=Image.open(label_Path)
    image=toTensor(image)
    label=toTensor(label)

    label=Binaryzation(label)

    temp=torch.cat([image,label],dim=0)
    temp=randomCrop(temp)
    label=temp[image.shape[0]:,:,:]
    image=temp[:image.shape[0],:,:]

    label=label*255
    image=image*255
    image=image.permute(1,2,0)

    label =label.squeeze()
    label=label.numpy()
    image=image.numpy()
    label=label.astype('uint8')
    image=image.astype('uint8')
    if np.sum(label)<5:
        return CropImage(image_Path,label_Path)
    else :

        return Image.fromarray(image,'RGB'),Image.fromarray(label)



def Run(in_image_Path='./',in_label_Path='./',out_Path='./'):
    image_dir = os.listdir(in_image_Path)
    label_dir = os.listdir(in_label_Path)
    for round in range(1):
        for idx in range(len(image_dir)):
            im,la= CropImage(in_image_Path+image_dir[idx],in_label_Path+label_dir[idx])
            im.save(out_Path+'image/'+str(round)+"_"+str(idx)+'.png')
            la.save(out_Path+'label/'+str(round)+"_"+str(idx)+'.png')

if __name__=="__main__":
    Run('./WholeData/train/image/','./WholeData/train/label/','CropImage/valid/')