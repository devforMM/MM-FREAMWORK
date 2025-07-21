import torch
import string
def tri_gram_vocab():
    alphabets = string.ascii_lowercase
    indices = torch.randperm(len(alphabets)).tolist()
    return {f"{alphabets[i]}": indices[i] for i in range(len(indices)) }

class AddNorm_Layer:
    def __init__(self, d_model, eps=1e-6):
        self.weights=[]
        self.gamma = torch.ones(d_model, requires_grad=True)
        self.beta = torch.zeros(d_model, requires_grad=True)
        self.eps = eps
        self.weights.extend([self.gamma,self.beta])
    def forward(self,x,y):
        y =x+y
        mean = y.mean(dim=-1, keepdim=True)
        var = y.var(dim=-1, keepdim=True)

        norm_y = (y - mean) / torch.sqrt(var + self.eps)
        norm_y = self.gamma * norm_y + self.beta
        return norm_y

class Feed_Forward:
    def __init__(self,dmodel):
        self.d_model=dmodel
        dff=512*self.d_model
        self.w1=torch.randn(self.d_model,dff,requires_grad=True)
        self.w2=torch.randn(dff,self.d_model,requires_grad=True)
        self.b1=torch.randn(dff,requires_grad=True)
        self.b2=torch.randn(self.d_model,requires_grad=True)
        self.weights=[]
        self.weights.extend([self.w1,self.w2,self.b1,self.b2])
    def forward(self,x):
        y=x@self.w1+self.b1    
        return torch.relu(y)@self.w2+self.b2

class Linear:
    def __init__(self,dmodel,vocab_size):
        self.weights=[]
        self.w=torch.randn(dmodel,vocab_size,requires_grad=True)
        self.b=torch.randn(vocab_size,requires_grad=True)
        self.weights.extend([self.w,self.b])
    def forward(self,x):
        return x@self.w+self.b

class Embdeing_layer:
    def __init__(self,dmodel):
        self.embedings=torch.randn(dmodel,dmodel,requires_grad=True)
    def get_embedings(self, seq):
        out = self.embedings[seq]
        out.retain_grad()  # important si tu veux inspecter .grad
        return out


def get_postional_embedding(d_model, position):
    positional = []
    for i in range(d_model):
        if i % 2 == 0:
            value = torch.sin(torch.tensor(position / (10000 ** (i / d_model))))
        else:
            value = torch.cos(torch.tensor(position / (10000 ** ((i - 1) / d_model))))
        positional.append(value)
    return torch.tensor(positional).reshape(1, -1)

def pos_encoding(x,dmodel):
  for i in range(x.shape[1]):
    x[:,i,:]=x[:,i,:]+get_postional_embedding(dmodel,i)
  return x

