import torch
import torch.nn as nn

def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class FeatureNet(nn.Module):
    '''
    Downsample RGB image to feature map (H/8, W/8)
    '''
    def __init__(self, in_c, out_c, depth_unit= 32):
        super().__init__()
        self.net = nn.Sequential(
            Conv2d_block(in_c, depth_unit * 1, kernel_size= 3, stride=1, padding= 1),
            Conv2d_block(depth_unit * 1, depth_unit * 2, kernel_size= 4, stride= 2, padding= 1), # (H/2, W/2)
            Conv2d_block(depth_unit * 2, depth_unit * 4, kernel_size= 3, stride= 1, padding= 1),
            Conv2d_block(depth_unit * 4, depth_unit * 8, kernel_size= 4, stride= 2, padding= 1), # (H/4, W/4)
            Conv2d_block(depth_unit * 8, depth_unit * 8, kernel_size= 3, stride= 1, padding= 1),
            Conv2d_block(depth_unit * 8, out_c, kernel_size= 4, stride= 2, padding= 1)           # (H/8, W/8)
        )

    def forward(self, x):
        out = self.net(x)
        return out 

class DispNet(nn.Module):
    '''
    Predict disparity map from feature map (upsample * 8)
    '''
    def __init__(self, in_c, out_c, depth_unit = 32):
        super().__init__()
        self.net = nn.Sequential(
            Conv2d_block(in_c, depth_unit * 8, kernel_size= 3, stride= 1, padding= 1),
            ConvTranspose2d_block(depth_unit * 8, depth_unit * 8, kernel_size= 4, stride= 2, padding= 1), #(H/4, W/4)
            Conv2d_block(depth_unit * 8, depth_unit * 8, kernel_size= 3, stride= 1, padding= 1),
            ConvTranspose2d_block(depth_unit * 8, depth_unit * 8, kernel_size= 4, stride= 2, padding= 1), #(H/2, W/2)
            Conv2d_block(depth_unit * 8, depth_unit * 8, kernel_size= 3, stride= 1, padding= 1),
            ConvTranspose2d_block(depth_unit * 8, depth_unit * 8, kernel_size= 4, stride= 2, padding= 1), #(H, W)
            Conv2d_block(depth_unit * 8, out_c, kernel_size= 3, stride= 1, padding= 1, use_relu= False),
            nn.Softmax(dim= 1)
        )
    
    def forward(self, x):
        out = self.net(x)
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


class Conv2d_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding, use_relu= True):
        super().__init__()
        block = [nn.Conv2d(in_c, out_c, kernel_size, stride, padding= padding),
                nn.BatchNorm2d(out_c)]
        if use_relu:
            block.append(nn.ReLU(inplace= True))
        self.net = nn.Sequential(* block)

    def forward(self, x):
        out = self.net(x)
        return out

class ConvTranspose2d_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding, output_padding=0, use_relu= True):
        super().__init__()
        block = [nn.ConvTranspose2d(in_c, out_c, kernel_size, stride,padding= padding, output_padding= output_padding),
                nn.BatchNorm2d(out_c)]
        if use_relu:
            block.append(nn.ReLU(inplace= True))
        self.net = nn.Sequential(* block)
    
    def forward(self, x):
        out = self.net(x)
        return out

def unit_test():
    imgs_left = torch.zeros(4, 3, 8, 8)
    imgs_right = torch.zeros(4, 3, 8, 8)
    model = Net_1(16, 256)
    out = model(imgs_left, imgs_right)

    print('Parameter number:', parameter_number(model))
    print('Output size',out.size())
    
if __name__ == '__main__':
    unit_test()