import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model_manager import Manaeger
from dataset import Dataset_mine
from matplotlib import pyplot as plt
import sys
sys.path.append('models')


parser = argparse.ArgumentParser()
parser.add_argument('mode', help='Train/Generate', choices=['train', 'predict'])
parser.add_argument('model', help= 'Model')
parser.add_argument('-id', help= 'Experiment ID, used for saving files', default= '0')
parser.add_argument('-lr', help= 'Learning rate',type=float, default= 0.001)
parser.add_argument('-batch_size', type= int, default= 64)
parser.add_argument('-epoch_num', type = int, default= 10)
parser.add_argument('-load', help='Weights to be load', default= None)
parser.add_argument('-check_batch_num', help= 'How many batches to show result once', type= int, default=10)
parser.add_argument('-info', help= 'information for records', default= None)
parser.add_argument('-left', help= 'Path to left image', default= None)
parser.add_argument('-right', help= 'Path to right image', default= None)
parser.add_argument('-output', help= 'Path to output image', defualt= None)
args = parser.parse_args()

transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

# Prepare datasets, data loader
data_set = Dataset_mine('../data_scene_flow',
                        transform= transform)
data_loader = DataLoader(data_set, batch_size= args.batch_size, shuffle= True, drop_last = True)

def get_model(model_name):
    file = __import__(model_name)
    model = file.Model()
    return model

def main():
    print('main function is running ...')
    model = get_model(args.model)
    manager = Manaeger(model, args)
    manager.load_data(data_loader)
    if args.mode == 'train':
        manager.train()
    elif args.mode == 'predict':
        img_left  = transform(plt.imread(args.left))
        img_right = transform(plt.imread(args.right))
        manager.predict(img_left, img_right, args.out)

if __name__ == '__main__':
    main()
