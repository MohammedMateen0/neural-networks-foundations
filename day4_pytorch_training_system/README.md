# Day 4 - Complete PyTorch Training System

## Overview

This project implements a complete deep learning training pipeline in PyTorch. The goal of this project was not to achieve high accuracy on a complex dataset but to understand and implement every component of a modern neural network training workflow from scratch.

The implementation covers model creation, custom datasets, data loading, training loops, evaluation, loss functions, optimization, checkpointing, and the practical concepts required in machine learning engineering interviews.

---

# Learning Objectives

By completing this project I learned:

* How PyTorch models are built using `nn.Module`
* How custom datasets are created using `Dataset`
* How mini-batch training works through `DataLoader`
* The difference between training and evaluation modes
* The purpose of gradient tracking and `torch.no_grad()`
* How loss functions are selected for different ML tasks
* How optimizers update model parameters
* Why learning rate scheduling is important
* How gradient clipping stabilizes training
* How early stopping prevents overfitting
* How model checkpointing preserves the best model
* How a complete PyTorch training pipeline is structured

---

# Project Structure

```text
day4_pytorch_training_system/
│
├── custom_dataset.py
├── model.py
├── train.py
├── evaluate.py
├── README.md
└── xor_model.pth
```

---

# Concepts Covered

## 1. nn.Module

Every PyTorch model inherits from `nn.Module`.

Benefits:

* Automatic parameter registration
* Model saving and loading
* GPU support
* Integration with optimizers
* Support for train and evaluation modes

Example:

```python
class XORNet(nn.Module):
    ...
```

---

## 2. Dataset

Custom datasets are created by inheriting from:

```python
torch.utils.data.Dataset
```

Required methods:

### **len**()

Returns dataset size.

### **getitem**()

Returns a single sample and label.

Benefits:

* Flexible data loading
* Supports images, text, tabular data
* Integrates directly with DataLoader

---

## 3. DataLoader

DataLoader converts individual samples into mini-batches.

Features:

* Batching
* Shuffling
* Efficient iteration
* Parallel loading support

Example:

```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

---

## 4. Mini-Batch Training

Instead of training on the entire dataset at once:

```text
Dataset
↓
Mini Batches
↓
Gradient Updates
```

Advantages:

* Faster training
* Better memory efficiency
* More stable optimization

---

## 5. Training Mode

```python
model.train()
```

Used during training.

Important because:

* Dropout becomes active
* BatchNorm updates running statistics

---

## 6. Evaluation Mode

```python
model.eval()
```

Used during validation and testing.

Effects:

* Dropout disabled
* BatchNorm uses learned statistics

Provides deterministic predictions.

---

## 7. torch.no_grad()

```python
with torch.no_grad():
```

Purpose:

* Disable gradient tracking
* Reduce memory usage
* Speed up inference

Difference from `model.eval()`:

| Feature                    | model.eval() | no_grad() |
| -------------------------- | ------------ | --------- |
| Disables Dropout           | Yes          | No        |
| Disables BatchNorm Updates | Yes          | No        |
| Stops Gradient Tracking    | No           | Yes       |

Both are typically used together during evaluation.

---

## 8. Forward Pass

The model receives input data and generates predictions.

```python
predictions = model(X_batch)
```

Flow:

```text
Input
↓
Layers
↓
Activation Functions
↓
Output
```

---

## 9. Loss Functions

Loss functions measure prediction error.

### Binary Classification

Used:

```python
nn.BCEWithLogitsLoss()
```

Examples:

* Fraud Detection
* Spam Classification
* Disease Prediction

Important:

Do NOT add a Sigmoid layer before `BCEWithLogitsLoss()`.

The loss function applies Sigmoid internally.

---

### Multi-Class Classification

```python
nn.CrossEntropyLoss()
```

Examples:

* Image Classification
* Document Classification

Important:

Do NOT apply Softmax before `CrossEntropyLoss()`.

The loss function applies Softmax internally.

---

### Regression

```python
nn.MSELoss()
```

Examples:

* House Price Prediction
* Sales Forecasting
* Temperature Prediction

Alternative:

```python
nn.L1Loss()
```

for MAE.

---

## 10. Backpropagation

```python
loss.backward()
```

Purpose:

Compute gradients for all trainable parameters.

Process:

```text
Loss
↓
Chain Rule
↓
Gradients
↓
Stored in .grad
```

---

## 11. Gradient Reset

```python
optimizer.zero_grad()
```

Required before every backward pass.

Reason:

PyTorch accumulates gradients by default.

Without zeroing:

```text
Old Gradients
+
New Gradients
```

which produces incorrect updates.

---

## 12. Optimizers

### SGD

```python
torch.optim.SGD
```

Characteristics:

* Simple
* Low memory usage
* Sensitive to learning rate

---

### Adam

```python
torch.optim.Adam
```

Characteristics:

* Adaptive learning rates
* Momentum
* Fast convergence

Common default choice.

---

### AdamW

```python
torch.optim.AdamW
```

Characteristics:

* Adam + proper weight decay
* Better regularization
* Used in Transformers

Common in:

* BERT
* GPT
* Llama

---

## 13. Parameter Updates

```python
optimizer.step()
```

Purpose:

Update weights using gradients.

Concept:

```text
Weights
↓
Gradients
↓
Optimizer
↓
New Weights
```

---

## 14. Learning Rate Scheduling

Learning rates should decrease during training.

Reason:

```text
Far From Minimum
↓
Large Steps

Near Minimum
↓
Small Steps
```

Schedulers Learned:

### StepLR

Reduces learning rate at fixed intervals.

### ReduceLROnPlateau

Reduces learning rate when validation performance stops improving.

### CosineAnnealingLR

Gradually decreases learning rate using a cosine schedule.

---

## 15. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_
```

Purpose:

Prevent exploding gradients.

Especially useful for:

* RNNs
* LSTMs
* GRUs

Problem:

```text
Very Large Gradient
↓
Huge Weight Update
↓
Training Instability
```

Solution:

Clip gradients before optimizer updates.

---

## 16. Validation Loop

Validation measures generalization performance.

Characteristics:

* No weight updates
* No training
* Used for model selection

Typically executed with:

```python
model.eval()

with torch.no_grad():
```

---

## 17. Overfitting

Observed when:

```text
Training Loss ↓
Validation Loss ↑
```

Meaning:

The model is memorizing training data rather than learning general patterns.

---

## 18. Early Stopping

Purpose:

Stop training when validation performance stops improving.

Benefits:

* Prevents overfitting
* Saves training time
* Improves generalization

Key Parameter:

```python
patience
```

---

## 19. Model Checkpointing

Save the best-performing model.

Example:

```python
torch.save(
    model.state_dict(),
    "best_model.pth"
)
```

Benefits:

* Preserves best validation performance
* Allows recovery after interruption
* Works together with early stopping

Important:

The best model is not always the final epoch.

---

## 20. Complete Training Pipeline

The full workflow implemented and understood during this project:

```text
Dataset
↓
DataLoader
↓
Model (nn.Module)
↓
Forward Pass
↓
Loss Calculation
↓
Backward Pass
↓
Gradient Clipping
↓
Optimizer Step
↓
Learning Rate Scheduler
↓
Validation Loop
↓
Checkpointing
↓
Early Stopping
↓
Inference
```

---

# Technologies Used

* Python
* PyTorch
* Autograd
* Neural Networks

---

# Key Takeaways

* A complete training system involves much more than defining a neural network.
* Dataset and DataLoader form the data pipeline.
* Training and evaluation modes serve different purposes.
* Loss functions must match the prediction task.
* Adam and AdamW are practical optimizer defaults.
* Learning rate scheduling improves convergence.
* Gradient clipping stabilizes deep sequence models.
* Early stopping prevents overfitting.
* Checkpointing ensures the best model is preserved.
* These components form the foundation of all modern deep learning systems.

---

# Next Step

Day 5 - Regularization, Batch Normalization, and Weight Initialization

Topics:

* Dropout
* Batch Normalization
* Overfitting Reduction
* Xavier Initialization
* He Initialization
* Regularization Experiments
* Ablation Studies
