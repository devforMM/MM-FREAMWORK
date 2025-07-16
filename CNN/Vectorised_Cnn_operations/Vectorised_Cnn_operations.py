import torch

# Xavier Normal Initialization
class Vec_Conv_XavierNormal:
    @staticmethod
    def initialize(n_in, n_out,kernel_size):
        std = (2 / (n_in + n_out)) ** 0.5
        return torch.normal(0, std, size=(n_out,n_in,kernel_size[0],kernel_size[1]),requires_grad=True,dtype=torch.float32)


# He Normal Initialization
class Vec_Conv_HeNormal:
    @staticmethod
    def initialize(n_in, n_out,kernel_size):
        std = (2 / n_in) ** 0.5
        return torch.normal(0, std, size=(n_out,n_in,kernel_size[0],kernel_size[1]),requires_grad=True,dtype=torch.float32)


# Xavier Uniform Initialization
class Vec_Conv_XavierUniform:
    @staticmethod
    def initialize(n_in, n_out,kernel_size):
        limit = (6 / (n_in + n_out)) ** 0.5
        tensor= torch.empty(n_out,n_in,kernel_size[0],kernel_size[1],dtype=torch.float32).uniform_(-limit, limit)
        return tensor.requires_grad_()


# He Uniform Initialization
class Vec_Conv_HeUniform:
    @staticmethod
    def initialize(n_in,n_out,kernel_size):
        limit = (6 / n_in) ** 0.5
        tensor= torch.empty(n_out,n_in,kernel_size[0],kernel_size[1],dtype=torch.float32).uniform_(-limit, limit)
        return tensor.requires_grad_()
