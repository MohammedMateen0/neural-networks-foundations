import torch

print("="*50)
print("MANUAL GRADIENT EXAMPLES")
print("="*50)

# Example 1
w = 3
x = 2

loss = (w*x)**2

gradient = 2*w*(x**2)

print("\nExample 1")
print("Loss:", loss)
print("Expected Gradient:", gradient)

# Example 2

w = 2
x = 3

gradient = 3*(w**2)*(x**3)

print("\nExample 2")
print("Expected Gradient:", gradient)

# Example 3

w = 2
x = 2

gradient = 10*(w**9)*(x**10)

print("\nExample 3")
print("Expected Gradient:", gradient)