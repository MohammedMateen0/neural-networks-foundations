import torch
import torch.nn as nn

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(2,4),
            nn.ReLU(),
            nn.Linear(4,1)
        )
    def forward(self,x):
        return self.network(x)
    