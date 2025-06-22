import torch


def xavier_normal(w_matrice_size, n_in, n_out):

    std = (2 / (n_in + n_out)) ** 0.5
    return torch.normal(0, std, size=w_matrice_size)


def he_normal(w_matrice_size, n_in):

    std = (2 / n_in) ** 0.5
    return torch.normal(0, std, size=w_matrice_size)


def xavier_uniform(w_matrice_size, n_in, n_out):

    limit = (6 / (n_in + n_out)) ** 0.5
    return torch.empty(w_matrice_size).uniform_(-limit, limit)


def he_uniform(w_matrice_size, n_in):
    limit = (6 / n_in) ** 0.5
    return torch.empty(w_matrice_size).uniform_(-limit, limit)


