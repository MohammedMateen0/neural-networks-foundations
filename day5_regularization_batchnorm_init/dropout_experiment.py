import torch
import torch.nn as nn

class NoDropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(20,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,x):
        return self.network(x)
class DropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(20,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,1)
        )
    def forward(self,x):
        return self.network(x)

model1=NoDropoutNet()
model2=DropoutNet()

print(model1)
print()
print(model2)