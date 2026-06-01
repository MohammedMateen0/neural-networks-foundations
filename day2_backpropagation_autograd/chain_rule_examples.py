import torch

print("="*50)
print("CHAIN RULE VERIFICATION")
print("="*50)

w = torch.tensor(
    3.0,
    requires_grad=True
)

x = torch.tensor(2.0)

y = w*x + 1

loss = y**2

loss.backward()

print("\nExpected Gradient: 28")
print("PyTorch Gradient:", w.grad.item())


print("\nSecond Example")

w = torch.tensor(
    2.0,
    requires_grad=True
)

y = w**2

z = y**3

z.backward()

print("Expected Gradient: 192")
print("PyTorch Gradient:", w.grad.item())