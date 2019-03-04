import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from util import *

class Manager():
    def __init__(self, model, args):
        
        if  args.load:
            load_name = os.path.join('../weights/', args.load)
            model.load_state_dict(torch.load(load_name + '.pkl'))
        
        self.id = args.id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr)
        self.criteria = F.smooth_l1_loss()
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        
        self.save_name = os.path.join('../weights/', self.id + '.pkl')
        self.folder = os.path.join('results',self.id)
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_file = open(os.path.join(self.folder, 'record.txt'), 'w')
        self.info = args.info
    
    def load_data(self, data_loader_train, data_loader_valid):
        self.data_train = data_loader_train
        self.data_valid = data_loader_valid
        
    def record(self, message):
        self.log_file.write(message)
        print(message, end='')

    def get_info(self):
        info = get_string('\n ID:', self.id, '\n')
        info = get_string(info, 'infomation:', self.info, '\n')
        info = get_string(info, 'Epoch number:', self.epoch_num, '\n')
        info = get_string(info, 'Batch size:', self.batch_size, '\n')
        info = get_string(info, '=======================\n\n')
        return info

    def train(self):
        self.record(self.get_info())

        for epoch in range(self.epoch_num):
            train_loss = self.forward('train')
            valid_loss = self.forward('valid')
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
            # self.record(get_string('batch loss:', loss3, '\n'))

            disp = self.model(left_img, right_img)
            loss = self.criteria(disp, target_disp)
            total_loss += loss.item()
            self.record(get_string('batch loss:', loss, '\n'))

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return total_loss / (batch_id + 1)

    def predict(self, img_left, img_right, out):
        pass
