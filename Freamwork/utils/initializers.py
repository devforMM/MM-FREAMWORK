import torch

# Xavier Normal Initialization
class XavierNormal:
    @staticmethod
    def initialize(n_in, n_out):
        std = (2 / (n_in + n_out)) ** 0.5
        return torch.normal(0, std, size=(n_in, n_out),requires_grad=True)


# He Normal Initialization
class HeNormal:
    @staticmethod
    def initialize(n_in, n_out):
        std = (2 / n_in) ** 0.5
        return torch.normal(0, std, size=(n_in, n_out),requires_grad=True)


# Xavier Uniform Initialization
class XavierUniform:
    @staticmethod
    def initialize(n_in, n_out):
        limit = (6 / (n_in + n_out)) ** 0.5
        return torch.empty(n_in, n_out).uniform_(-limit, limit,requires_grad=True)


# He Uniform Initialization
class HeUniform:
    @staticmethod
    def initialize(n_in, n_out):
        limit = (6 / n_in) ** 0.5
        return torch.empty(n_in, n_out).uniform_(-limit, limit,requires_grad=True)
