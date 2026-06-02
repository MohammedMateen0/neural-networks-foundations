import torch

x = torch.tensor(
    10.0,
    requires_grad=True
)

y = torch.sigmoid(x)

y.backward()

print("Sigmoid Gradient:", x.grad.item())

x = torch.tensor(
    10.0,
    requires_grad=True
)

y = torch.tanh(x)

y.backward()

print("Tanh Gradient:", x.grad.item())

x = torch.tensor(
    10.0,
    requires_grad=True
)

y = torch.relu(x)

y.backward()

print("ReLU Gradient:", x.grad.item())