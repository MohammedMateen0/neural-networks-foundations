import torch

activations = {
    "Sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
    "ReLU": torch.nn.ReLU(),
    "LeakyReLU": torch.nn.LeakyReLU(0.01)
}

for name, activation in activations.items():

    x = torch.tensor(
        5.0,
        requires_grad=True
    )

    y = activation(x)

    y.backward()

    print(name)
    print("Output:", y.item())
    print("Gradient:", x.grad.item())
    print("-"*40)