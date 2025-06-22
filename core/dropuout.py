import torch
def dropout(outputs, p):
    mask = torch.ones_like(outputs)
    indices = torch.randperm(mask.numel())[:int(mask.numel()*p)]
    mask = mask.flatten()
    mask[indices] = 0
    mask = mask.reshape(outputs.shape)
    out_drop = outputs * mask * (1.0 / (1.0 - p))
    return out_drop
