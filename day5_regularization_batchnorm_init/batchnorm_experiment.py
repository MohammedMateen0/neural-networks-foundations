import torch
import torch.nn as nn
X=torch.tensor([
    [100.,200.,300.],
    [110.,210.,310.],
    [120.,220.,320.],
    [130.,230.,330.]
])

print("Original Data",
      X)

print("Mean",
      X.mean(dim=0))

print("Std",
      X.std(dim=0))

bn=nn.BatchNorm1d(
    num_features=3,
    affine=False
)
output=bn(X)
print("\n After BatchNorm",
      output,
      "\n Mean",
      output.mean(dim=0),
      "\n Std",
      output.std(dim=0))