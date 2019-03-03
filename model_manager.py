import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from util import *

class Manaeger():
    def __init__(self, model, args):
        
        if  args.load:
            load_name = os.path.join('../weights/', args.load)
            model.load_state_dict(torch.load(load_name + '.pkl'))
        
        self.id = args.id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr)
        self.criteria = None
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.check_batch_num = args.check_batch_num
        
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
        info = get_string('\nID:', self.id, '\n')
        info = get_string(info, 'infomation:', self.info, '\n')
        info = get_string(info, 'Model:', self.model.name(), '\n')
        info = get_string(info, 'Learning rate:', self.lr, '\n')
        info = get_string(info, 'Epoch number:', self.epoch_num, '\n')
        info = get_string(info, 'Batch size:', self.batch_size, '\n')
        info = get_string(info, '=======================\n\n')
        return info

    def train(self):
        self.record(self.get_info())

        for epoch in range(self.epoch_num):
            self.model.train()
            
            for batch_id, imgs in enumerate(self.data_loader):
                self.model.zero_grad()
                

                # Record some information
                if (batch_id + 1) % self.check_batch_num == 0:
                    info = ""
                    self.record(info + '\n')

            torch.save(self.model.state_dict(), self.save_name)
           
    def predict(self, img_left, img_right, out):
        pass
