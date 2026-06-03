import torch
import torch.nn as nn

layer1=nn.Linear(100,50)

nn.init.xavier_uniform_(
    layer1.weight
)

print("Xavier",
      layer1.weight.mean().item(),
      layer1.weight.std().item())

layer2=nn.Linear(100,50)

nn.init.kaiming_uniform_(
    layer2.weight,
    nonlinearity="relu"
)

print("\n He",
      layer2.weight.mean().item(),
      layer2.weight.std().item())
