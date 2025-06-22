import torch
def weight_decay(loss,l,w):
 return loss +l*torch.sum(torch.abs(w)**2)