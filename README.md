# Neural Networks Foundations: Perceptron, XOR and PyTorch

## Overview

This project explores the fundamental building blocks of neural networks.

The implementation progresses from a single perceptron built from scratch using NumPy to a trainable neural network implemented in PyTorch.

The project demonstrates why a single perceptron cannot solve XOR and how hidden layers enable neural networks to learn non-linear decision boundaries.

---

## Topics Covered

* Perceptron from scratch
* Weighted sum and bias
* Step activation function
* AND gate implementation
* OR gate implementation
* NOT gate implementation
* XOR problem
* Linear separability
* Hidden layers
* Neural network representation learning
* PyTorch training loop
* Binary cross entropy loss
* SGD and Adam optimizers

---

## Project Structure

```text
perceptron_from_scratch.py
logic_gates.py
xor_manual_network.py
xor_pytorch.py
```

---

## Results

### Logic Gates

Single perceptrons successfully implemented:

* AND
* OR
* NOT

### XOR

A single perceptron failed to solve XOR because XOR is not linearly separable.

A two-layer neural network successfully learned XOR.

Final predictions:

```text
[0,0] -> 0
[0,1] -> 1
[1,0] -> 1
[1,1] -> 0
```

### PyTorch Training

Architecture:

```text
Input (2)
↓
Hidden Layer (2)
↓
Output Layer (1)
```

Optimizer:

* Adam

Loss:

* Binary Cross Entropy

Final loss:

```text
0.00015
```

Predictions:

```text
[[0.000088]
 [0.99990 ]
 [0.99989 ]
 [0.000085]]
```

---

## Key Learnings

* A perceptron learns a linear decision boundary.
* Bias controls the activation threshold.
* XOR cannot be represented by a single perceptron.
* Hidden layers transform the feature space.
* Backpropagation updates model parameters using gradients.
* Adam often converges faster than vanilla SGD.

---

## Technologies Used

* Python
* NumPy
* PyTorch

---

## Future Work

* Backpropagation derivation
* Activation function comparison
* MNIST classifier
* Convolutional Neural Networks
