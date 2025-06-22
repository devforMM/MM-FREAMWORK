import torch



def add_layer(neurons,inputs,initializer=None):
        if initializer==None:
            return  torch.rand(inputs.shape[1],neurons)
        else:
             return initializer((inputs.shape[1],neurons),inputs.shape[1],neurons)

def forward(x,w,activation=None):
        z = x @ w
        if activation is None:
            return z
        elif activation == "Relu":
            return torch.relu(z)
        elif activation == "sigmoid":
            return torch.sigmoid(z)
        elif activation == "tanh":
            return torch.tanh(z)
        else:
            raise ValueError(f"Unknown activation function {activation}")


