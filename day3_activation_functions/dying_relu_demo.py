import torch
import torch.nn as nn

X=torch.tensor([-5.,-6.,-8.,-2.,0.])

relu=nn.ReLU()

outputs=relu(X)

print("inputs:",X)

print("OutPuts After ReLU",outputs)

x=torch.tensor(-5.0,requires_grad=True)
y=relu(x)

y.backward()

print("Gradient at x=-5:",x.grad)

print("\nExplanation:")
print("Input < 0")
print("Output = 0")
print("Gradient = 0")
print("Neuron cannot update through this path")
