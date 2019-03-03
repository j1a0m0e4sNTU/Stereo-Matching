import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutil
from matplotlib import pyplot as plt
from util import *

class Dataset_mine(Dataset):
    def __init__(self, root, mode= 'train',transform = None):
        super(Dataset_mine, self).__init__()
        self.transform = transform
        data_dir = os.path.join(root,'training')
        left_dir = os.path.join(data_dir,'image_2')
        right_dir = os.path.join(data_dir, 'image_3')
        disp_dir = os.path.join(data_dir, 'disp_occ_0')

        self.left_list  = [os.path.join(left_dir, name) for name in os.listdir(left_dir) if name.endswith('_10.png')]
        self.right_list = [os.path.join(right_dir, name)  for name in os.listdir(right_dir) if name.endswith('_10.png')]
        self.disp_list  = [os.path.join(disp_dir, name) for name in os.listdir(disp_dir)]
        
        self.left_list.sort()
        self.right_list.sort()
        self.disp_list.sort()

        validation_num = int(len(self.disp_list) * 0.2)
        if mode == 'train':
            self.left_list = self.left_list[validation_num:]
            self.right_list= self.right_list[validation_num:]
            self.disp_list = self.disp_list[validation_num:]
        else:
            self.left_list = self.left_list[:validation_num]
            self.right_list= self.right_list[:validation_num]
            self.disp_list = self.disp_list[:validation_num]

    def __len__(self):
        return len(self.disp_list)

    def __getitem__(self, idx):
        sample = {}
        img_left = plt.imread(self.left_list[idx])
        img_right= plt.imread(self.right_list[idx])
        img_disp = plt.imread(self.disp_list[idx])

        if self.transform:
            img_left = self.transform(img_left)
            img_right= self.transform(img_right)
        img_disp = torch.tensor(img_disp)
        
        sample['left'] = img_left
        sample['right']= img_right
        sample['disp'] = img_disp

        return sample

def unit_test():
    data = Dataset_mine('../data_scene_flow', 'train',
                        transform= transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
                        ]))
    
    loader = DataLoader(data, batch_size= 8)
    
    print('data set size:', len(data))
    for id, sample in enumerate(loader):
        if id == 5:
            break
        print(sample['disp'][0].size())
        print(torch.sum(sample['disp'][0]))
    

if __name__ == '__main__':
    unit_test()