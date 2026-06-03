# Day 5 - Regularization, Batch Normalization & Weight Initialization

## Overview

This project explores three fundamental concepts that make deep neural networks train effectively and generalize to unseen data:

1. Regularization
2. Normalization
3. Weight Initialization

A model that performs well on training data but poorly on validation data has learned to memorize rather than generalize. Modern deep learning relies on techniques such as Dropout, Weight Decay, Batch Normalization, and proper initialization strategies to prevent overfitting and improve optimization.

The experiments in this project demonstrate how these techniques influence training stability, gradient flow, and model performance.

---

# Learning Objectives

After completing this project, I can:

* Explain overfitting and underfitting
* Implement Dropout in PyTorch
* Explain why Dropout reduces overfitting
* Apply L2 Regularization using Weight Decay
* Explain why zero initialization fails
* Explain symmetry breaking in neural networks
* Compare Xavier and He initialization
* Implement Batch Normalization
* Explain BatchNorm's learnable parameters (γ and β)
* Compare BatchNorm and LayerNorm
* Explain why Transformers use LayerNorm
* Understand gradient clipping and exploding gradients

---

# Project Structure

```text
day5_regularization_batchnorm_init/
│
├── dropout_experiment.py
├── batchnorm_experiment.py
├── initialization_experiment.py
├── dropout_train_vs_eval.py
└── README.md
```

---

# 1. Overfitting and Generalization

## Overfitting

Occurs when a model memorizes training data instead of learning useful patterns.

Typical behavior:

```text
Training Accuracy  ↑
Validation Accuracy ↓
```

Symptoms:

* Excellent training performance
* Poor performance on unseen data
* Poor generalization

---

## Underfitting

Occurs when a model is too simple to learn meaningful patterns.

Typical behavior:

```text
Training Accuracy  Low
Validation Accuracy Low
```

Common causes:

* Excessive regularization
* Insufficient model capacity
* Too few training epochs

---

# 2. Dropout

## Definition

Dropout randomly disables a percentage of neurons during each training forward pass.

Example:

```python
nn.Dropout(0.5)
```

Meaning:

```text
50% of neurons are randomly disabled
during each training forward pass
```

---

## Why Dropout Works

Without Dropout:

```text
Neuron A
↓
Neuron B
↓
Prediction
```

The network may become overly dependent on a small set of neurons.

This is called:

```text
Co-adaptation
```

---

With Dropout:

```text
Random neurons disappear
every forward pass
```

The network is forced to learn multiple redundant representations.

Result:

```text
Better Generalization
Less Overfitting
```

---

## Training vs Evaluation

### Training Mode

```python
model.train()
```

Dropout is active.

Example:

```text
[1,1,1,1]
↓
[1,0,1,0]
```

---

### Evaluation Mode

```python
model.eval()
```

Dropout is disabled.

All neurons remain active.

Reason:

Predictions must be deterministic.

---

# 3. L2 Regularization (Weight Decay)

## Motivation

Large weights often indicate overly complex decision boundaries.

Example:

### Model A

```text
Weights:
0.8
-0.6
1.2
```

### Model B

```text
Weights:
80
-50
120
```

Model B is more likely to overfit.

---

## Solution

Penalize large weights during training.

PyTorch:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

---

## Effect

### Small Weight Decay

```text
Allows learning
Controls complexity
```

### Large Weight Decay

```text
Weights → 0
Model becomes too simple
Underfitting
```

---

# 4. Weight Initialization

## Why Initialization Matters

Training starts from initial weights.

Bad initialization can cause:

* Slow training
* Vanishing gradients
* Exploding gradients
* Failure to converge

---

# 5. Why Zero Initialization Fails

Suppose:

```text
Neuron A = [0,0,0]
Neuron B = [0,0,0]
```

Both neurons:

```text
Produce identical outputs
Receive identical gradients
Remain identical forever
```

The network cannot learn diverse features.

---

## Symmetry Breaking

Random initialization ensures:

```text
Different Outputs
↓
Different Gradients
↓
Different Features Learned
```

This process is called:

```text
Symmetry Breaking
```

---

# 6. Xavier (Glorot) Initialization

Designed for:

```text
Sigmoid
Tanh
```

Purpose:

Maintain stable activations and gradients across layers.

Typical Formula:

```text
std = sqrt(1 / fan_in)
```

---

## When To Use

```text
Sigmoid Networks
Tanh Networks
```

---

# 7. He (Kaiming) Initialization

Designed for:

```text
ReLU
Leaky ReLU
```

Purpose:

Compensate for activations lost by ReLU.

Typical Formula:

```text
std = sqrt(2 / fan_in)
```

---

## Why He Works Better For ReLU

ReLU removes negative activations:

```text
x < 0
↓
0
```

Approximately half of activations are discarded.

He initialization uses a larger variance to preserve signal flow.

---

## Initialization Summary

| Activation | Initialization |
| ---------- | -------------- |
| Sigmoid    | Xavier         |
| Tanh       | Xavier         |
| ReLU       | He             |
| Leaky ReLU | He             |

---

# 8. Batch Normalization

## Motivation

During training, activation distributions shift across layers.

Example:

```text
Batch 1:
Mean = 10
Std = 20

Batch 2:
Mean = -5
Std = 50
```

Changing distributions make optimization harder.

---

## BatchNorm Solution

Normalize activations:

```text
Mean ≈ 0
Std ≈ 1
```

before passing them to the next layer.

PyTorch:

```python
nn.BatchNorm1d(...)
```

---

## Benefits

* Faster convergence
* More stable training
* Higher learning rates
* Improved gradient flow
* Mild regularization

---

# 9. Gamma (γ) and Beta (β)

BatchNorm does not permanently force:

```text
Mean = 0
Std = 1
```

Instead it learns:

```text
γ (scale)
β (shift)
```

Transformation:

```text
Normalized Output
↓
γ × Output + β
```

This preserves flexibility while maintaining stable optimization.

---

# 10. BatchNorm Placement

Typical order:

```text
Linear
↓
BatchNorm
↓
ReLU
```

Reason:

Normalize activations before non-linear transformation.

Benefits:

* Stable input distribution for ReLU
* Better gradient flow
* Faster training

---

# 11. Layer Normalization

BatchNorm computes statistics across:

```text
Batch Dimension
```

LayerNorm computes statistics across:

```text
Feature Dimension
```

within a single sample.

---

## Why Transformers Use LayerNorm

Transformer inference often operates with:

```text
Batch Size = 1
```

BatchNorm becomes unreliable.

LayerNorm remains stable because it does not depend on other samples in the batch.

Used in:

* BERT
* GPT
* Llama
* Modern Transformer Architectures

---

# BatchNorm vs LayerNorm

| Feature                      | BatchNorm | LayerNorm |
| ---------------------------- | --------- | --------- |
| Uses Batch Statistics        | Yes       | No        |
| Depends on Batch Size        | Yes       | No        |
| Works Well With Batch Size 1 | No        | Yes       |
| Common In CNNs               | Yes       | Rare      |
| Common In Transformers       | No        | Yes       |

---

# 12. Gradient Clipping

## Problem

During backpropagation:

```text
Gradient → Extremely Large
```

Result:

```text
Huge Weight Updates
↓
Training Instability
↓
Divergence
```

This is called:

```text
Exploding Gradients
```

---

## Solution

PyTorch:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

Limits gradient magnitude before optimizer updates.

---

## Common Usage

Especially useful in:

* RNNs
* LSTMs
* GRUs

where gradients propagate through many timesteps.

---

# Experiments Performed

## dropout_experiment.py

Compared:

```text
No Dropout
vs
Dropout(0.5)
```

Purpose:

Understand how random neuron deactivation acts as regularization.

---

## dropout_train_vs_eval.py

Compared:

```text
model.train()
vs
model.eval()
```

Purpose:

Observe how Dropout behaves differently during training and inference.

---

## batchnorm_experiment.py

Observed:

```text
Mean Before BatchNorm
Std Before BatchNorm
```

and

```text
Mean After BatchNorm ≈ 0
Std After BatchNorm ≈ 1
```

Purpose:

Visualize normalization directly.

---

## initialization_experiment.py

Compared:

```text
Xavier Initialization
vs
He Initialization
```

Purpose:

Observe differences in weight distributions and variance.

---

# Key Takeaways

* Dropout reduces overfitting by preventing neuron co-adaptation.
* Weight decay penalizes large weights and improves generalization.
* Zero initialization fails because neurons remain identical.
* Random initialization enables symmetry breaking.
* Xavier initialization is designed for Sigmoid and Tanh networks.
* He initialization is designed for ReLU-based networks.
* BatchNorm stabilizes activations and accelerates training.
* BatchNorm learns γ and β to preserve representational flexibility.
* LayerNorm is independent of batch size and is preferred in Transformers.
* Gradient clipping prevents exploding gradients and stabilizes sequence models.

---

# Technologies Used

* Python
* PyTorch
* Neural Networks
* Autograd

---

# Next Step

Week 9 Project: MNIST Digit Classifier

Topics to apply:

* Custom Datasets
* DataLoaders
* Training Loops
* Dropout
* BatchNorm
* Weight Initialization
* Evaluation Metrics
* Model Checkpointing
* Neural Network Optimization
