import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()

    def forward(self, x):
        return x
  
def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unit_test():
    imgs = torch.zeros(10, 3, 64, 64)
    model = Model()
    out = model(imgs)

    print('Parameter number: ',parameter_number(model))
    print('Input size: ', imgs.size())
    print('Output size:', out.size())
    
if __name__ == '__main__':
    unit_test()