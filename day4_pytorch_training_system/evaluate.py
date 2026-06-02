import torch

from model import XORNet

model = XORNet()

model.load_state_dict(
    torch.load("xor_model.pth")
)

model.eval()

X = torch.tensor([
    [0.,0.],
    [0.,1.],
    [1.,0.],
    [1.,1.]
])

with torch.no_grad():

    logits = model(X)

    predictions = torch.sigmoid(
        logits
    )

print(predictions)