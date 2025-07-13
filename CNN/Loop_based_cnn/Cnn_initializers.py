import torch

# Xavier Normal Initialization
class Conv_XavierNormal:
    @staticmethod
    def initialize(C_in,n_in, n_out,kernel_size):
        std = (2 / (n_in + n_out)) ** 0.5
        return torch.normal(0, std, size=(C_in,kernel_size[0],kernel_size[1]),requires_grad=True,dtype=torch.float32)


# He Normal Initialization
class Conv_HeNormal:
    @staticmethod
    def initialize(C_in,n_in, n_out,kernel_size):
        std = (2 / n_in) ** 0.5
        return torch.normal(0, std, size=(C_in,kernel_size[0],kernel_size[1]),requires_grad=True,dtype=torch.float32)


# Xavier Uniform Initialization
class Conv_XavierUniform:
    @staticmethod
    def initialize(C_in,n_in, n_out,kernel_size):
        limit = (6 / (n_in + n_out)) ** 0.5
        return torch.empty(C_in,kernel_size[0],kernel_size[1]).uniform_(-limit, limit,requires_grad=True,dtype=torch.float32)


# He Uniform Initialization
class Conv_HeUniform:
    @staticmethod
    def initialize(C_in,n_in,n_out,kernel_size):
        limit = (6 / n_in) ** 0.5
        return torch.empty(C_in,kernel_size[0],kernel_size[1]).uniform_(-limit, limit,requires_grad=True,dtype=torch.float32)
