# Backpropagation and Autograd in PyTorch

## Overview

This project explores the mathematical and computational foundations of neural network training.

The focus is understanding how gradients are computed using the chain rule and how PyTorch's autograd engine automatically performs backpropagation.

---

## Topics Covered

- Forward pass
- Loss functions
- Computational graphs
- Chain rule
- Backpropagation
- requires_grad
- loss.backward()
- Gradient accumulation
- optimizer.zero_grad()
- optimizer.step()
- Computational graph inspection
- Gradient inspection in neural networks

---

## Manual Gradient Example

Given:

y = wx

loss = y²

Chain rule:

d(loss)/dw =
d(loss)/dy × dy/dw

Example:

w = 3
x = 2

loss = (wx)²

Gradient:

24

---

## Autograd Example

```python
x = torch.tensor(
    2.0,
    requires_grad=True
)

y = x**2

y.backward()

print(x.grad)
````

Output:

```text
tensor(4.)
```

---

## Gradient Accumulation

PyTorch accumulates gradients by default.

```python
optimizer.zero_grad()
```

must be called before every backward pass.

---

## Computational Graph

Example graph:

```text
w
↓
w²
↓
(w²)³
↓
loss
```

Backpropagation:

```text
loss
↑
(w²)³
↑
w²
↑
w
```

---

## Key Learnings

* Backpropagation is repeated application of the chain rule.
* PyTorch builds computational graphs dynamically.
* loss.backward() computes gradients.
* optimizer.step() updates parameters.
* Gradients accumulate unless cleared.
* Every modern neural network is trained using backpropagation.

---

## Technologies

* Python
* PyTorch

```

