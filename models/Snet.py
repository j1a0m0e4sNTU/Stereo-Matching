import torch
import torch.nn as nn
import torch.nn.functional as F
from components import *

class SNet_1(nn.Module):
    def __init__(self, max_disp):
        super().__init__()
    
    def forward(self, img_left, img_right):
        return 

class SNet_0(nn.Module):
    def __init__(self, max_disp):
        super().__init__()
        self.spp = SPP(in_c= 6, out_c = 32)
        self.block0 = StackedResidual(3, in_c = 32*5, out_c = 256, kernel_size= 3, stride= 1, padding= 1)
        self.block1 = StackedResidual(2, in_c = 256, out_c = max_disp, kernel_size = 3, stride= 1, padding= 1)
        self.regression = DisparityRegression(max_disp)

    def forward(self, img_left, img_right):
        img_stack = torch.cat([img_left, img_right], dim= 1)
        out = self.spp(img_stack)
        out = self.block0(out)
        out = self.block1(out)
        disp_prob = F.softmax(out, dim=1)
        disp = self.regression(disp_prob)
        return disp

class Net_bf(nn.Module):
    '''Brute Force -- Fit left image to its disparity'''
    def __init__(self, max_disp, hidden_depth= 256, depth_unit= 32):
        super().__init__()
        self.feature_extract = FeatureNet(in_c= 3, out_c= hidden_depth, depth_unit= depth_unit)
        self.disp_extract = DispNet(in_c= hidden_depth, out_c= max_disp, depth_unit= depth_unit)
        self.regression = DisparityRegression(max_disp)

    def forward(self, img_left, img_right):
        feature = self.feature_extract(img_left)
        disp_prob = self.disp_extract(feature)
        disp = self.regression(disp_prob)
        return disp

class Net_0(nn.Module):
    ''' Ectract featre then stack'''
    def __init__(self, max_disp, hidden_depth= 256):
        super().__init__()
        self.feature_extract = FeatureNet(in_c= 3, out_c= hidden_depth, depth_unit= 32)
        self.disp_extract = DispNet(in_c= hidden_depth * 2, out_c= max_disp, depth_unit= 32)
        self.regression = DisparityRegression(max_disp)

    def forward(self, img_left, img_right):
        feature_left = self.feature_extract(img_left)
        feature_right = self.feature_extract(img_right)
        feature_stacked = torch.cat([feature_left, feature_right], dim= 1)
        disp_prob = self.disp_extract(feature_stacked)
        disp = self.regression(disp_prob)
        return disp

class Net_1(nn.Module):
    '''Stack image first then extract feature'''
    def __init__(self, max_disp, hidden_depth= 256):
        super().__init__()
        self.feature_extract = FeatureNet(in_c= 6, out_c= hidden_depth, depth_unit= 32)
        self.disp_extracct = DispNet(in_c= hidden_depth, out_c= max_disp, depth_unit= 32)
        self.regression = DisparityRegression(max_disp)

    def forward(self, img_left, img_right):
        img_stack = torch.cat([img_left, img_right], dim= 1)
        feature = self.feature_extract(img_stack)
        disp_prob = self.disp_extracct(feature)
        disp = self.regression(disp_prob)
        return disp

def unit_test():
    max_disp = 192
    # imgs_left = torch.zeros(4, 3, 256, 512)
    # imgs_right = torch.zeros(4, 3, 256, 512)
    model = SNet_0(max_disp)
    # out = model(imgs_left, imgs_right)

    print('Parameter number:', parameter_number(model))
    # print('Output size',out.size())
    
if __name__ == '__main__':
    unit_test()