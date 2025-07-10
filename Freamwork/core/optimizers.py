
import torch

class GradientDescent:
    def __init__(self):
        pass

    def update(self, w, lr):
        return w - lr * w.grad


class Momentum:
    def __init__(self, beta):
        self.beta = beta
        self.v = {}

    def update(self, w, lr):
        if w not in self.v:
            self.v[w] = torch.zeros_like(w)
        self.v[w] = self.beta * self.v[w] + (1 - self.beta) * w.grad
        return w - lr * self.v[w]


class Adagrad:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.g = {}

    def update(self, w, lr):
        if w not in self.g:
            self.g[w] = torch.zeros_like(w)
        self.g[w] += w.grad ** 2
        return w - lr / (torch.sqrt(self.g[w]) + self.epsilon) * w.grad


class RMSProp:
    def __init__(self, beta=0.9, epsilon=1e-8):
        self.beta = beta
        self.epsilon = epsilon
        self.g = {}

    def update(self, w, lr):
        if w not in self.g:
            self.g[w] = torch.zeros_like(w)
        self.g[w] = self.beta * self.g[w] + (1 - self.beta) * w.grad ** 2
        return w - lr / (torch.sqrt(self.g[w]) + self.epsilon) * w.grad


class Adam:
    def __init__(self, beta=0.9, beta2=0.999, epsilon=1e-8):
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v = {}
        self.g = {}

    def update(self, w, lr, t):
        if w not in self.v:
            self.v[w] = torch.zeros_like(w)
            self.g[w] = torch.zeros_like(w)
        self.v[w] = self.beta * self.v[w] + (1 - self.beta) * w.grad
        self.g[w] = self.beta2 * self.g[w] + (1 - self.beta2) * w.grad ** 2
        v_corr = self.v[w] / (1 - self.beta ** t)
        g_corr = self.g[w] / (1 - self.beta2 ** t)
        return w - lr * v_corr / (torch.sqrt(g_corr) + self.epsilon)
