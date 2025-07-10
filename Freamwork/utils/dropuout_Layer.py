import torch

class Droupout_layer:
    def __init__(self, p):
        self.p=p
def forward(self,x):
    mask = torch.ones_like(x)
    indices = torch.randperm(mask.numel())[:int(mask.numel()*self.p)]
    mask = mask.flatten()
    mask[indices] = 0
    mask = mask.reshape(x.shape)
    x = x * mask * (1.0 / (1.0 - self.p))
    return x

