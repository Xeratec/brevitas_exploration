# Copyright 2025 ETH Zurich.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Victor Jung <jungvi@iis.ee.ethz.ch>

### Torch Imports ###
from torch import nn


# Define a simple FC model
class SimpleFCModel(nn.Module):
    def __init__(self):
        super(SimpleFCModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),  # Flatten 28x28 images into 784 vectors
            nn.Linear(28 * 28, 128),  # First FC layer
            nn.ReLU(),
            nn.Linear(128, 64),  # Second FC layer
            nn.ReLU(),
            nn.Linear(64, 10),  # Output layer for 10 classes
        )

    def forward(self, x):
        return self.fc(x)
