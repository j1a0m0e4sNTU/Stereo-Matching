import torch
import torch.nn as nn
import torch.nn.functional as F

class SPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.rate = [8, 16, 32, 64]
        self.conv = Conv2dBlock(in_c, out_c, kernel_size= 3, stride= 1, padding= 1)
        self.branch0 = self.__getBranch(in_c, out_c, kernel_size= self.rate[0], stride= self.rate[0])
        self.branch1 = self.__getBranch(in_c, out_c, kernel_size= self.rate[1], stride= self.rate[1])
        self.branch2 = self.__getBranch(in_c, out_c, kernel_size= self.rate[2], stride= self.rate[2])
        self.branch3 = self.__getBranch(in_c, out_c, kernel_size= self.rate[3], stride= self.rate[3])
        

    def __getBranch(self, in_c, out_c, kernel_size, stride, padding= 0):
        branch = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride, padding),
            Conv2dBlock(in_c, out_c, kernel_size= 3, stride= 1, padding= 1)
        )
        return branch

    def forward(self, inputs): # (b, in_c, H, W)
        out = self.conv(inputs)
        out0 = F.upsample(self.branch0(inputs), scale_factor= self.rate[0], mode= 'bilinear', align_corners= False)
        out1 = F.upsample(self.branch1(inputs), scale_factor= self.rate[1], mode= 'bilinear', align_corners= False)
        out2 = F.upsample(self.branch2(inputs), scale_factor= self.rate[2], mode= 'bilinear', align_corners= False)
        out3 = F.upsample(self.branch3(inputs), scale_factor= self.rate[3], mode= 'bilinear', align_corners= False)
        out = torch.cat([out, out0, out1, out2, out3], dim= 1) #(b, out_c * 5, H, W) 
        return out

class StackedResidual(nn.Module):
    def __init__(self, block_num , in_c, out_c, kernel_size, stride, padding):
        super().__init__()
        net = [ResidualBlock(in_c, out_c, kernel_size, stride, padding)]
        for _ in range(block_num - 1):
            net.append(ResidualBlock(out_c, out_c))
        self.net = nn.Sequential(* net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size= 3, stride =1, padding= 1):
        super().__init__()
        self.downSample = None
        if stride != 1 or in_c != out_c: 
            self.downSample = Conv2dBlock(in_c, out_c, kernel_size, stride, padding, use_relu= False)

        self.residual = nn.Sequential(
            Conv2dBlock(out_c, out_c, kernel_size= 3, stride= 1, padding= 1, use_relu= True),
            Conv2dBlock(out_c, out_c, kernel_size= 3, stride= 1, padding= 1, use_relu= False)
        )

    def forward(self, inputs):
        if self.downSample:
            inputs = self.downSample(inputs)
        res = self.residual(inputs)
        out = inputs + res
        return out
    

class FeatureNet(nn.Module):
    '''
    Downsample RGB image to feature map (H/8, W/8)
    '''
    def __init__(self, in_c, out_c, depth_unit= 32):
        super().__init__()
        self.net = nn.Sequential(
            Conv2dBlock(in_c, depth_unit * 1, kernel_size= 3, stride=1, padding= 1),
            Conv2dBlock(depth_unit * 1, depth_unit * 2, kernel_size= 4, stride= 2, padding= 1), # (H/2, W/2)
            Conv2dBlock(depth_unit * 2, depth_unit * 4, kernel_size= 3, stride= 1, padding= 1),
            Conv2dBlock(depth_unit * 4, depth_unit * 8, kernel_size= 4, stride= 2, padding= 1), # (H/4, W/4)
            Conv2dBlock(depth_unit * 8, depth_unit * 8, kernel_size= 3, stride= 1, padding= 1),
            Conv2dBlock(depth_unit * 8, out_c, kernel_size= 4, stride= 2, padding= 1)           # (H/8, W/8)
        )

    def forward(self, inputs):
        out = self.net(inputs)
        return out 

class DispNet(nn.Module):
    '''
    Predict disparity map from feature map (upsample * 8)
    '''
    def __init__(self, in_c, out_c, depth_unit = 32):
        super().__init__()
        self.net = nn.Sequential(
            Conv2dBlock(in_c, depth_unit * 8, kernel_size= 3, stride= 1, padding= 1),
            ConvTranspose2dBlock(depth_unit * 8, depth_unit * 8, kernel_size= 4, stride= 2, padding= 1), #(H/4, W/4)
            Conv2dBlock(depth_unit * 8, depth_unit * 8, kernel_size= 3, stride= 1, padding= 1),
            ConvTranspose2dBlock(depth_unit * 8, depth_unit * 8, kernel_size= 4, stride= 2, padding= 1), #(H/2, W/2)
            Conv2dBlock(depth_unit * 8, depth_unit * 8, kernel_size= 3, stride= 1, padding= 1),
            ConvTranspose2dBlock(depth_unit * 8, depth_unit * 8, kernel_size= 4, stride= 2, padding= 1), #(H, W)
            Conv2dBlock(depth_unit * 8, out_c, kernel_size= 3, stride= 1, padding= 1, use_relu= False),
            nn.Softmax(dim= 1)
        )
    
    def forward(self, inputs):
        out = self.net(inputs)
        return out

class DisparityRegression(nn.Module):
    def __init__(self, max_disp):
        super().__init__()
        self.disp_score = torch.arange(max_disp)  # [D]
        self.disp_score = self.disp_score.view(1, -1, 1, 1)  # [1, D, 1, 1]

    def forward(self, prob):
        disp_score = self.disp_score.expand_as(prob).type_as(prob)  # [B, D, H, W]
        out = torch.sum(disp_score * prob, dim=1)  # [B, H, W]
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding, use_relu= True):
        super().__init__()
        block = [nn.Conv2d(in_c, out_c, kernel_size, stride, padding= padding),
                nn.BatchNorm2d(out_c)]
        if use_relu:
            block.append(nn.ReLU(inplace= True))
        self.net = nn.Sequential(* block)

    def forward(self, inputs):
        out = self.net(inputs)
        return out

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding, output_padding=0, use_relu= True):
        super().__init__()
        block = [nn.ConvTranspose2d(in_c, out_c, kernel_size, stride,padding= padding, output_padding= output_padding),
                nn.BatchNorm2d(out_c)]
        if use_relu:
            block.append(nn.ReLU(inplace= True))
        self.net = nn.Sequential(* block)
    
    def forward(self, inputs):
        out = self.net(inputs)
        return out

def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unit_test():
    zeroTensor = torch.zeros(4, 6, 64, 64)
    model = SPP(6, 8)
    out = model(zeroTensor)

    print('Parameter number:', parameter_number(model))
    print('Output size',out.size())
    
if __name__ == '__main__':
    unit_test()
