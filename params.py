import torch
import torch.nn as nn
import torch.optim as optim

class Params:
    def __init__(self):
        self._a = nn.Parameter(torch.randn(3, 5), requires_grad=True)
        self._b = nn.Parameter(torch.randn(3, 1), requires_grad=True)
        self._w = nn.Parameter(torch.randn(3, 5), requires_grad=True)
        self.training_setup()

    @property
    def get_a(self):
        return self._a
    
    @property
    def get_b(self):
        return self._b
    
    @property
    def get_w(self):
        return self._w
    
    def training_setup(self):
        l = [
            {'params': [self._a], "name": "a"},
            {'params': [self._b], "name": "b"},
            {'params': [self._w], "name": "w"}
        ]
        self.optimizer = optim.Adam(l, lr=0.0001)
