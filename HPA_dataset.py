import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import transform

import torch
from torch.utils.data import Dataset, DataLoader

class HPAdata_train(Dataset):
    ''' Dataset for Human Protein Atlas pics, 
        read in all 4 channels:
        protein (green),
        nucleus (blue), 
        microtubules (red), 
        endoplasmic reticulum (yellow)'''
    
    def __init__(self, ids_labels, path, transform=None):
        '''
        Args:
            ids_labels(object): DataFrame object containing ids and labels
            path(str): root path from which read in images
            transform(object, optional): tranforms applied on input
        '''        
        self.ids_labels = ids_labels
        self.path = path 
        self.channels = ['_red.png', '_green.png', '_blue.png', '_yellow.png']
        self.transform = transform
    
    def __len__(self):
        return self.ids_labels.shape[0]
    
    
    def __getitem__(self, idx):
        ''' return a tuple including:
            an image of 4 channels and a tuple indecating labels'''
        
        row = self.ids_labels.loc[idx]
        
        # img processing
        img_id = row.Id
        img_dirs = [os.path.join(self.path, img_id+self.channels[i]) for i in range(4)]
        img = np.zeros((512, 512, 4), dtype=np.float32)
        for i in range(4):
            img[:,:,i] = plt.imread(img_dirs[i])
        if self.transform:
            img = self.transform(img)
       
        # labels processing
        labels = row[2:].values.astype(np.float32)
        labels = torch.from_numpy(labels)
        
        result = (img, labels)
        return result

    
class HPAdata_test(Dataset):
    ''' Dataset for Human Protein Atlas pics, 
        read in all 4 channels:
        protein (green),
        nucleus (blue), 
        microtubules (red), 
        endoplasmic reticulum (yellow)'''
    
    def __init__(self, path, transform=None):
        '''
        Args:
            path(str): root path from which read in images
            transform(object, optional): tranforms applied on input
        '''        
        self.path = path
        self.imgs_ids = [i[:i.find('_')] for i in os.listdir(self.path)][::4]
        self.channels = ['_red.png', '_green.png', '_blue.png', '_yellow.png']
        self.transform = transform    
    
    def __len__(self):
        return len(self.imgs_ids)
       
    def __getitem__(self, idx):
        ''' return a tuple containing:
            an image of 4 channels and its id'''
        
        img_id = self.imgs_ids[idx]
        img_dirs = [os.path.join(self.path, img_id+self.channels[i]) for i in range(4)]
        img = np.zeros((512, 512, 4), dtype=np.float32)
        for i in range(4):
            img[:,:,i] = plt.imread(img_dirs[i])
        if self.transform:
            img = self.transform(img)
        
        result = (img_id, img)
        return result


class RandomRotate(object):
    ''' Random rotation'''
    
    def __init__(self, angle):
        if angle>30 or angle<-30:
            raise Exception('Rotation Angle should not be larger than 30 degree !')
        else:
            self.angle = angle
            
    def __call__(self, img):
        ran_angle = (np.random.rand()-0.5)*2*self.angle
        return transform.rotate(img, ran_angle)
    

class Scale(object):
    ''' Scale the input to a desired size '''
    
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            raise Exception('Size must be an integer, output will be squared !')
            
    def __call__(self, img):
        return transform.resize(img, self.size)

    
class RandomCrop(object):
    ''' Crop the input to the specified size randomly '''
    
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            raise Exception('Size must be an integer, output will be squared !')
            
    def __call__(self, img):
        img_h, img_w, _ = img.shape
        crop_h, crop_w = self.size
        upperleft_y, upperleft_x = np.random.choice(img_h-crop_h), np.random.choice(img_w-crop_w)
        img_cropped = img[upperleft_x:upperleft_x+crop_w, upperleft_y:upperleft_y+crop_h, :]
        return img_cropped

        
class RandomColor(object):
    ''' Randomly add small vibrations on four channles r,g,b,y'''
    
    def __init__(self, r_ratio, g_ratio, b_ratio, y_ratio):
        if r_ratio > 1 or g_ratio > 1 or b_ratio > 1 or y_ratio > 1:
            raise Exception('Ratio could NOT be larger than 1 !')
        else:
            self.ratios = np.array([r_ratio, g_ratio, b_ratio, y_ratio])
            
    def __call__(self, img):
        ran_ratios = np.ones((4,)) - self.ratios * ((np.random.rand(4,)-0.5)*2)
        return np.clip((img * ran_ratios), 0, 1)

    
class totensor(object):
    ''' Convert to tensor'''
    
    def __call__(self, img):
        img_tensor = torch.from_numpy(img.transpose(2,0,1)).to(torch.float32)
        return img_tensor