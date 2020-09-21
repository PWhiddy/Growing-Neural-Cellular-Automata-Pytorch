import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import imageio
import random

class CifarImg2ImgDataset(Dataset):

    def __init__(self, label_img_dir, train=True, transform=None):

        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                        download=True, transform=None)
        if transform == None:
           self.transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transform = transform

        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'red', 'star')
        self.class_imgs = [
            self.transform(imageio.imread(label_img_dir+'/'+cls+'/'+cls+'1.png')[:,:,0:3]) 
            for cls in self.classes]
        self.blank = self.transform(imageio.imread(label_img_dir+'/../c/c1.png')[:,:,0:3])
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
    
        img, label = self.dataset.__getitem__(idx)
        img = self.transform(img)
        orig = torch.ones([img.shape[0], img.shape[1]+10, img.shape[2]])
        orig[:,0:32,:] = img
        orig[:,32:42,:] = self.blank

        # randomly make some very easy to classify
        if random.random() < 0.12:
            label = 10
            orig[:,0:32,:] = -1.0
            orig[0,0:32,:] = 1.0
            if random.random() < 0.5:
                label = 11
                sz = random.randint(5,30)
                x_off = random.randint(1,32-sz)
                y_off = random.randint(1,32-sz)
                
                #orig[0,0:32,:] = 0.5 #############
                orig[:,y_off:y_off+sz,x_off:x_off+sz] = -1.0
                orig[:,y_off:y_off+sz,x_off:x_off+sz] = -1.0
            noise = torch.rand_like(orig[:,0:32,:])
            orig[:,0:32,:] *= torch.pow(noise,0.2)

        labeled = orig.clone()
        labeled[:,32:42,:] = self.class_imgs[label]

        # randomly do not request classification
        if random.random() < 0.05:
            orig[:,32:42,:] = 1.0
            labeled[:,32:42,:] = 1.0

        return orig, labeled