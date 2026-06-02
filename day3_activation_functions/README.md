# Day 3 - Activation Functions & Gradient Flow

## Overview

This project explores activation functions in deep learning and their impact on gradient flow during neural network training.

Activation functions introduce non-linearity into neural networks. Without them, multiple stacked layers collapse into a single linear transformation and cannot learn complex patterns such as XOR.

The experiments in this folder demonstrate activation outputs, gradient behavior, vanishing gradients, and the dying ReLU problem.

---

## Learning Objectives

* Understand why activation functions are required
* Compare Sigmoid, Tanh, ReLU, and Leaky ReLU
* Understand the vanishing gradient problem
* Understand the dying ReLU problem
* Compare gradient flow across activations
* Learn why ReLU replaced Sigmoid in deep networks
* Understand why GELU is used in modern Transformer architectures

---

## Files

### activation_functions_demo.py

Demonstrates the outputs of:

* Sigmoid
* Tanh
* ReLU
* Leaky ReLU

for positive, negative, and zero inputs.

---

### activation_gradient_comparison.py

Computes and compares gradients produced by different activation functions.

Topics covered:

* Activation derivatives
* Gradient magnitude
* Saturation effects

---

### relu_vs_sigmoid.py

Compares gradient flow between ReLU and Sigmoid.

Key observation:

* ReLU maintains strong gradients for positive inputs
* Sigmoid gradients shrink as activations saturate

---

### dying_relu_demo.py

Demonstrates the dying ReLU problem.

Shows:

* Negative inputs producing zero output
* Zero gradients stopping learning
* How Leaky ReLU keeps gradients alive

---

## Activation Functions

### Sigmoid

Range:

(0,1)

Advantages:

* Probability interpretation
* Useful for binary classification outputs

Limitations:

* Saturation
* Vanishing gradients

---

### Tanh

Range:

(-1,1)

Advantages:

* Zero-centered outputs
* Larger gradients near zero

Limitations:

* Still suffers from vanishing gradients

---

### ReLU

Formula:

f(x) = max(0,x)

Advantages:

* Fast computation
* Strong gradient flow
* Sparse activations

Limitations:

* Dying ReLU problem

---

### Leaky ReLU

Formula:

f(x) = max(0.01x,x)

Advantages:

* Prevents zero gradients in the negative region
* Reduces dying ReLU issues

---

### GELU

Used in:

* BERT
* GPT-family models
* Modern Transformer architectures

Advantages:

* Smooth activation
* Strong optimization performance
* Better gradient behavior than ReLU in many large-scale models

---

## Vanishing Gradient Problem

Sigmoid and Tanh saturate for large positive and negative inputs.

As a result:

Gradient → Small Value

Repeated multiplication of small gradients causes early layers to receive almost no learning signal.

This is known as the vanishing gradient problem.

---

## Dying ReLU Problem

When a ReLU neuron consistently receives negative inputs:

Output = 0

Gradient = 0

The neuron stops updating and may never recover.

Leaky ReLU addresses this issue by maintaining a small non-zero gradient in the negative region.

---

## Output Layer Activations

### Binary Classification

Activation:

Sigmoid

Examples:

* Fraud Detection
* Spam Classification
* Disease Prediction

---

### Multiclass Classification

Activation:

Softmax

Examples:

* Image Classification
* Document Classification

---

### Regression

Activation:

Linear (No Activation)

Examples:

* House Price Prediction
* Sales Forecasting

---

## Key Takeaways

* Activation functions introduce non-linearity into neural networks.
* Without activations, deep networks reduce to a single linear transformation.
* Sigmoid and Tanh suffer from vanishing gradients.
* ReLU enabled effective training of deep neural networks.
* Leaky ReLU reduces the dying ReLU problem.
* GELU is widely used in modern Transformer architectures.
* Activation choice directly affects optimization and model performance.

---

## Technologies Used

* Python
* PyTorch
* Autograd

---

## Next Step

Day 4 - Complete PyTorch Training System

Topics:

* Dataset
* DataLoader
* Training Loop
* Evaluation Loop
* Optimizers
* Learning Rate Scheduling
* Checkpointing
* Early Stopping
