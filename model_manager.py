import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from util import *

class Manager():
    def __init__(self, model, args):
        
        if  args.load:
            load_name = os.path.join('../weights/', args.load)
            model.load_state_dict(torch.load(load_name))
        
        self.id = args.id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr)
        self.criteria = SmoothL1Loss()
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        
        self.save_name = os.path.join('../weights/', self.id + '.pkl')
        self.log_file = open(os.path.join("results/", self.id + '.txt'), 'w')
        self.info = args.info
    
    def load_data(self, data_loader_train, data_loader_valid):
        self.data_train = data_loader_train
        self.data_valid = data_loader_valid
        
    def record(self, message):
        self.log_file.write(message)
        print(message, end='')

    def get_info(self):
        info = get_string('\n ID:', self.id, '\n')
        info = get_string(info, 'parameter number:', parameter_number(self.model), '\n')
        info = get_string(info, 'infomation:', self.info, '\n')
        info = get_string(info, 'Epoch number:', self.epoch_num, '\n')
        info = get_string(info, 'Batch size:', self.batch_size, '\n')
        info = get_string(info, '=======================\n\n')
        return info

    def train(self):
        self.record(self.get_info())
        self.record(self.model.__str__() + "\n")

        for epoch in range(self.epoch_num):
            train_loss = self.forward('train')
            # valid_loss = self.forward('valid')
            valid_loss = "no"
            info = get_string('Epoch', epoch, '|Train loss:', train_loss, '|Validation loss:', valid_loss, '\n')
            self.record(info)
            torch.save(self.model.state_dict(), self.save_name)

    def forward(self, mode):
        if mode == 'train':
            dataloader = self.data_train
        elif mode == 'valid':
            dataloader = self.data_valid

        total_loss = 0
        for batch_id, sample in enumerate(dataloader):
            left_img = sample['left'].to(self.device)
            right_img = sample['right'].to(self.device)
            target_disp = sample['disp'].to(self.device)   
                
            # disp1, disp2, disp3 = self.model(left_img, right_img)
            # loss1, loss2, loss3 = self.criteria(disp1, disp2, disp3, target_disp)
            # loss = loss1 * 0.5 + loss2 * 0.7 + loss3 * 1.0
            # total_loss += loss3.item()
            # self.record(get_string('batch loss:', loss3.item(), '\n'))

            disp = self.model(left_img, right_img)
            loss = F.smooth_l1_loss(disp, target_disp)
            total_loss += loss.item()
            print(loss.item())
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return total_loss / (batch_id + 1)

    def predict(self, img_left, img_right, out):
        transform = transforms.Compose([ToTensor(),
                            Normalize(mean= (0.5, 0.5, 0.5), std= (0.5, 0.5, 0.5))])

        
        left_cut = plt.imread(img_left)[:320, :1216]
        right_cut= plt.imread(img_right)[:320, :1216]
        sample = {}
        sample['left'], sample['right'] = left_cut, right_cut
        
        sample = transform(sample)
        img_left  = sample['left'].unsqueeze(0).to(self.device) 
        img_right = sample['right'].unsqueeze(0).to(self.device)
        
        disp = self.model(img_left, img_right)
        disp = disp.squeeze(0).detach().cpu().numpy()
        
        f, axarr = plt.subplots(2)
        axarr[0].imshow(left_cut)
        axarr[1].imshow(disp)
        plt.savefig(out)
