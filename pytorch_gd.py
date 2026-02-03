import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Generate dummy data (from gradientdescent.py)
np.random.seed(0)
x_np = np.linspace(3, 9, 11)
y_np = 2 * x_np + 0.3 + np.random.randn(len(x_np))

# Data- convert to float tensors
x = torch.tensor(x_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

# Initialize weights (requires_grad=True to track gradients)
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Hyperparameters
learning_rate = 0.001
epochs = 20

# For plotting
plt.figure()
plt.scatter(x_np, y_np)

for epoch in range(epochs):
    # Forward pass
    y_pred = w * x + b

    # Compute loss (MSE)
    loss = torch.mean((y_pred - y) ** 2)

    # Backward pass
    loss.backward()

    # Update weights manually (no optimizer for now)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Zero the gradients
    w.grad.zero_()
    b.grad.zero_()

    # Detach tensors before plotting to avoid the error
    with torch.no_grad():
        yp = (w * x + b).numpy()
        plt.plot(x_np, yp, color='red', alpha=0.3)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PyTorch - Gradient Descent")
    plt.savefig("pytorch_gradient_descent.png")

# Logging (Epoch 0 only for this small loop)
print(f"Epoch {0}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

print(f"\nTrained model: y = {w.item():.2f}x + {b.item():.2f}")  