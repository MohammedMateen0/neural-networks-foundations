import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from custom_dataset import XORDataSet
from model import XORNet

dataset=XORDataSet()

train_loader=DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
model=XORNet()
loss_fn=nn.BCEWithLogitsLoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs=100

for epoch in range(epochs):
    model.train()
    epoch_loss=0
    for X_batch,y_batch in train_loader:
        optimizer.zero_grad()
        logits=model(X_batch)
        loss=loss_fn(
            logits,
            y_batch
        )
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    if epoch%10==0:
        print(
            f"Epoch {epoch}"
            f'Loss={epoch_loss:.4f}'
        )

torch.save(
    model.state_dict(),
    "xor_model.pth"
)

print("Model Saved")