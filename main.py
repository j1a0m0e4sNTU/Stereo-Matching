import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model_manager import Manager
from dataset import Dataset_mine
from matplotlib import pyplot as plt
from util import *
import sys
sys.path.append('models')
from PSMnet import PSMnet
from components import *

parser = argparse.ArgumentParser()
parser.add_argument('mode', help='Train/Generate', choices=['train', 'predict'])
parser.add_argument('-maxdisp', help= 'max disparity', type= int, default= 192)
parser.add_argument('-id', help= 'Experiment ID, used for saving files', default= '0')
parser.add_argument('-lr', help= 'Learning rate',type=float, default= 0.001)
parser.add_argument('-batch_size', type= int, default= 2)
parser.add_argument('-epoch_num', type = int, default= 10)
parser.add_argument('-load', help='Weights to be load', default= None)
parser.add_argument('-info', help= 'information for records', default= None)
parser.add_argument('-left', help= 'Path to left image', default= None)
parser.add_argument('-right', help= 'Path to right image', default= None)
parser.add_argument('-output', help= 'Path to output image', default= None)
args = parser.parse_args()

transform = transforms.Compose([RandomCrop([256, 512]),
                                    ToTensor(),
                                    Normalize(mean=(0.5,0.5,0.5), std= (0.5,0.5,0.5))])
# Prepare datasets, data loader
data_train = Dataset_mine('../data_scene_flow', mode='train', transform= transform)
data_loader_train = DataLoader(data_train, batch_size= args.batch_size, shuffle= True)
data_valid = Dataset_mine('../data_scene_flow', mode='valid', transform= transform)
data_loader_valid = DataLoader(data_valid, batch_size= args.batch_size, shuffle= False)

model = Net_1(args.maxdisp, 320)

def main():
    print('main function is running ...')
    manager = Manager(model, args)
    manager.load_data(data_loader_train, data_loader_valid)
    if args.mode == 'train':
        manager.train()
    elif args.mode == 'predict':
        img_left  = transform(plt.imread(args.left))
        img_right = transform(plt.imread(args.right))
        manager.predict(img_left, img_right, args.out)

if __name__ == '__main__':
    main()
