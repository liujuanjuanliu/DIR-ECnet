"""
Aum Sri Sai Ram
By DG on 06-06-2020

Dataset class for RAFDB : 7 Basic emotions

Purpose: To return images from RAFDB dataset

Output:  bs x c x w x h        
            
"""

import torch.utils.data as data
from PIL import Image, ImageFile
import os
import pickle
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.utils import make_grid
ImageFile.LOAD_TRUNCATED_IAMGES = True
import random as rd 

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def default_reader(fileList):
    #print(fileList)
    counter_loaded_images_per_label = [0 for _ in range(7)]

    num_per_cls_dict = dict()
    for i in range(0, 7):
        num_per_cls_dict[i] = 0

    imgList = []
    if fileList.find('occlusion_list.txt') > -1:
       fp = open(fileList,'r')
       for names in fp.readlines():
           image_path, target, _  = names.split(' ')
           image_path = image_path.strip()+'.jpg'
           target = int(target)
           num_per_cls_dict[target] = num_per_cls_dict[target] + 1
           imgList.append((image_path, target))
       return imgList ,    num_per_cls_dict

    elif fileList.find('pose') > -1:
       fp = open(fileList,'r')
       for names in fp.readlines():
           target, image_path = names.split('/')  #Eg. for each entry before underscore lable and afterwards name in 1/fer0034656.jpg
           image_path = image_path.strip()
           #print(target,image_path)
           target = int(target)
           # target = change_emotion_label_same_as_affectnet(target)
           num_per_cls_dict[target] = num_per_cls_dict[target] + 1
           imgList.append((image_path, target))
       return imgList,    num_per_cls_dict
    else:#val/train/validation.csv

       fp = open(fileList,'r')

       for names in fp.readlines():

           image_path, target = names.split(' ')
           name,ext = image_path.strip().split('.')
           image_path = name + '_aligned.' + ext
           target = int(target) -1
           counter_loaded_images_per_label[target] += 1

           num_per_cls_dict[target] = num_per_cls_dict[target] + 1

           imgList.append((image_path, int(target)))

       fp.close()
       return imgList, num_per_cls_dict


# RAF-DB Labels : 1-7 made 0-6
def get_class(idx):  #class expression label
        classes = {
           0: 'Surprise',
           1: 'Fear',
           2: 'Disgust',
           3: 'Happiness',
           4: 'Sadness',
           5: 'Anger',
           6: 'Neutral'
        }

        return classes[idx]

class ImageList(data.Dataset):
    def __init__(self, root, fileList,  transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.cls_num = 7
        self.imgList, self.num_per_cls_dict = list_reader(fileList)
        self.transform = transform
        self.loader = loader
        self.is_save = True
        self.totensor = transforms.ToTensor()


    def __getitem__(self, index):
        imgPath, target_expression = self.imgList[index]
        # print(imgPath, target_expression)
        img = self.loader(os.path.join(self.root, imgPath))
        if self.transform is not None:
            img = self.transform(img)
        return img, target_expression

    def __len__(self):
        return len(self.imgList)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

if __name__=='__main__':
   testlist = default_reader('./rafdb_data/EmoLabel/all_of_label.txt')  # ../data/RAFDB/EmoLabel/val_raf_db_list_pose_45.txt
   imagesize =  224
   transform = transforms.Compose([transforms.Resize((imagesize,imagesize)), transforms.ToTensor()])
   # for i in range(20):
   #     print(testlist[i])
   
   dataset = ImageList(root='./rafdb_data/aligned/', fileList ='./rafdb_data/EmoLabel/pose_raf_db_list_45.txt', transform = transform)
#    dataset = ImageList(root='../data/RAFDB/Image/aligned/', fileList ='../data/RAFDB/EmoLabel/val_raf_db_list_pose_45.txt', transform = transform)
   fdi = iter(dataset)
   for i, data in enumerate(fdi):
        if i < 1:
           print(' ', data[0].size(), data[1] )
        else:
           break
   
