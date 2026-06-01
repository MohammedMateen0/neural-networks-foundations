import torch

print("="*50)
print("AUTOGRAD BASICS")
print("="*50)

x = torch.tensor(
    2.0,
    requires_grad=True
)

y = x**2

print("Tensor:", x)
print("Grad Function:", y.grad_fn)

y.backward()

print("Gradient:", x.grad)


print("\nSecond Example")

x = torch.tensor(
    4.0,
    requires_grad=True
)

y = x**3

y.backward()

print("Gradient:", x.grad)