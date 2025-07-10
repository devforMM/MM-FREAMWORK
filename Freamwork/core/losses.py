import torch

# MAE : Mean Absolute Error
class MAE:
    def compute_loss(self,y_pred, y_true):
        return torch.abs(y_pred - y_true).mean()


# MSE : Mean Squared Error
class MSE:
    def compute_loss(self,y_pred, y_true):
        return 0.5 * torch.mean((y_pred - y_true) ** 2)


# Cross-Entropy (avec softmax intégré)
class CrossEntropy:
    def __init__(self):
        ()
    def softmax(self,ypred):
        exp=torch.exp(ypred)
        sum=torch.sum(exp,dim=1,keepdim=True)
        return exp/sum
    
    def compute_loss(self,y_pred, y_true):
        res = self.softmax(y_pred)
        loss = -torch.mean(y_true*torch.log(res))
        return loss

class BinaryCrossEntropy:
        def __init__(self):
         ()
        def sigmoid(self,ypred):
           return 1/(1+torch.exp(ypred))
        def compute_loss(self,y_pred, y_true):
           res=self.sigmoid(y_pred)
           loss = -torch.mean(y_true * torch.log(res) + (1-y_true) * torch.log(1-res))
           return loss


