import torch

print("="*50)
print("COMPUTATIONAL GRAPH")
print("="*50)

w = torch.tensor(
    2.0,
    requires_grad=True
)

y1 = w**2
y2 = w**3

loss = y1 + y2

print("Loss Grad Function:")
print(loss.grad_fn)

print("\nNext Functions:")
print(loss.grad_fn.next_functions)

print("\nFirst Branch:")
print(
    loss.grad_fn.next_functions[0][0].next_functions
)

loss.backward()

print("\nGradient:")
print(w.grad)