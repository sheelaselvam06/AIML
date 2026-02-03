import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 1. Load data
dt = load_digits(n_class=3)
y_np = dt.target
images = dt.images

print(f"Sample target: {y_np[-2]}")
print(f"Image shape: {images[0].shape}")
print(f"Total dataset shape: {images.shape}")

# 2. Reshape and convert to tensors
# Each image is 8x8, so input feature size is 64
X = torch.tensor(images.reshape(-1, 64), dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.long)

print(f"Tensor X shape: {X.shape}")

# 3. Define Model
model = nn.Sequential(
    nn.Linear(64, 64),  # Input size is 64, not 537
    nn.ReLU(),
    nn.Linear(64, 3)
)

# 4. Training Setup
lossfn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Training Loop
print("Training...")
for epoch in range(100):
    optimizer.zero_grad()
    yp = model(X)
    loss = lossfn(yp, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 6. Evaluation
with torch.no_grad():
    predictions = torch.argmax(model(X), axis=1)
    accuracy = (predictions == y).float().mean()
    print(f"\nFinal Training Accuracy: {accuracy.item():.4f}")


