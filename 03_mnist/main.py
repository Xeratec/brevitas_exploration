# Copyright 2025 ETH Zurich.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Victor Jung <jungvi@iis.ee.ethz.ch>

# %%
### Systems Imports ###
import warnings
from pathlib import Path

### Torch Imports ###
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch", message=".*experimental feature.*")

### Brevitas Imports ###
import brevitas.nn as qnn
from brevitas.fx import brevitas_symbolic_trace

### Model Imports ###
from model import SimpleFCModel

# Hyperparameters
DEVICE_CPU = "cpu"
DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MODEL_NAME = "mnist_fc.pth"
FQ_MODEL_NAME = "fq_mnist_fc.pth"
TQ_MODEL_NAME = "tq_mnist_fc.pth"

EXPORT_FOLDER = Path().cwd()
if Path().cwd().name != "03_mnist":
    EXPORT_FOLDER = EXPORT_FOLDER / "03_mnist"


# %% WIP: Load the fake-quantize model and trace it
