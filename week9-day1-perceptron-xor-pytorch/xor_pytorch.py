import torch
import torch.nn as nn

torch.manual_seed(42)

X = torch.tensor([
    [0.,0.],
    [0.,1.],
    [1.,0.],
    [1.,1.]
])

y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

model = nn.Sequential(
    nn.Linear(2,2),
    nn.Sigmoid(),
    nn.Linear(2,1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for epoch in range(10000):

    pred = model(X)

    loss = loss_fn(pred, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 1000 == 0:
        print(epoch, loss.item())

with torch.no_grad():
    print(model(X))