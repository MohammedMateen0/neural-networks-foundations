import torch
import torch.nn as nn

x=torch.randn(5,10)

dropout=nn.Dropout(0.5)

print("Training Mode")

dropout.train()
print(dropout(x),
      "\nEvaluation Mode")

dropout.eval()
print(dropout(x))