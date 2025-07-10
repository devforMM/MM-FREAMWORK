import torch
class Batch_normalization_layer:
    def __init__(self, beta, gamma, eps=1e-8):
        self.beta=beta
        self.gamma=gamma
        self.eps=eps
    def forward(self,x):
        # Moyenne et variance du batch
        mean = torch.mean(x)
        var = torch.var(x)
        
        # Normalisation
        x_norm = (x - mean) / torch.sqrt(var +self.eps)
        
        # Recalage avec gamma (échelle) et beta (biais)
        out =  self.gamma* x_norm + self.normalizebeta
        
        return out