import torch
from core.initializers import *

class Layer:
     def __init__(self,input_shape,nbr_neurones,initializer,acitvation=None):
          self.activation=acitvation
          if initializer !=None:
            if initializer=="Xaviernormal":
                init= XavierNormal()
            elif initializer=="HeNormal":
                init=HeNormal()
            elif initializer=="XavierUniform":
                init=XavierUniform()
            elif initializer=="HeUniform":
                init=HeUniform()
            self.w= init.initialize(input_shape,nbr_neurones)
          else:
              self.w=torch.randn(input_shape,nbr_neurones,requires_grad=True)       

     def forward(self,x):
        z = x @ self.w
        if self.activation is None:
            return z
        elif self.activation == "relu":
            return torch.maximum(torch.tensor(0.0), z)
        elif self.activation == "lakyrelu":
            return torch.maximum(torch.tensor(0.001), z)
        elif self.activation == "sigmoid":
            return (torch.exp(z) - torch.exp(-z)) / (torch.exp(z) + torch.exp(-z))
        elif self.activation == "tanh":
            return torch.tanh(z)
        else:
            raise ValueError(f"Unknown activation function {self.activation}")


