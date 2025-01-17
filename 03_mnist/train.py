# Copyright 2025 ETH Zurich.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Victor Jung <jungvi@iis.ee.ethz.ch>

# %%

### Systems Imports ###
from pathlib import Path

### PyTorch Imports ###
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

### Model Import ###
from model import SimpleFCModel

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
MODEL_NAME = "mnist_fc.pth"

EXPORT_FOLDER = Path().cwd()
if Path().cwd().name != "03_mnist":
    EXPORT_FOLDER = EXPORT_FOLDER / "03_mnist"


# %%
# Transform: Normalize MNIST images
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # Normalize with mean=0.5, std=0.5
)

# Load MNIST dataset
train_dataset = datasets.MNIST(root=EXPORT_FOLDER / "data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=EXPORT_FOLDER / "data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# %%
# Initialize model, loss, and optimizer
model = SimpleFCModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")


# %%
# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test set: {100 * correct / total:.2f}%")


# %% Export the model
torch.save(model.state_dict(), EXPORT_FOLDER / MODEL_NAME)
print(f"Model saved to {EXPORT_FOLDER / MODEL_NAME}")
