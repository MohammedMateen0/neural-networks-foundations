import torch

print("="*50)
print("GRADIENT ACCUMULATION")
print("="*50)

w = torch.tensor(
    2.0,
    requires_grad=True
)

loss = w**2

loss.backward()

print("After First Backward:")
print(w.grad)

loss = w**3

loss.backward()

print("\nAfter Second Backward:")
print(w.grad)

print("""
Explanation:

First gradient = 4

Second gradient = 12

Accumulated gradient = 16
""")