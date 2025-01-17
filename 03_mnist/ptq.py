# Copyright 2025 ETH Zurich.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Victor Jung <jungvi@iis.ee.ethz.ch>

# %%
### Systems Imports ###
import copy
import warnings
from tqdm import tqdm
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
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.quantize import quantize
from brevitas.graph.calibrate import calibration_mode
from brevitas.fx import brevitas_symbolic_trace

### Model Imports ###
from model import SimpleFCModel

# Hyperparameters
DEVICE_CPU = "cpu"
DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MODEL_NAME = "mnist_fc.pth"
FQ_MODEL_NAME = "fq_mnist_fc.pth"

EXPORT_FOLDER = Path().cwd()
if Path().cwd().name != "03_mnist":
    EXPORT_FOLDER = EXPORT_FOLDER / "03_mnist"


# %% Load trained model
model = SimpleFCModel()
state_dict = torch.load(EXPORT_FOLDER / MODEL_NAME)
model.load_state_dict(state_dict)

# %% Tracing and pre-processing transformations
model = preprocess_for_quantize(model)


# %% Insert quantizers in the graph
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import Uint8ActPerTensorFloat

COMPUTE_LAYER_MAP = {
    nn.Linear: (
        qnn.QuantLinear,
        {
            # 'input_quant': Int8ActPerTensorFloat,
            "weight_quant": Int8WeightPerTensorFloat,
            "output_quant": Int8ActPerTensorFloat,
            "bias_quant": Int32Bias,
            "return_quant_tensor": True,
            # 'input_bit_width': 8,
            "output_bit_width": 8,
            "weight_bit_width": 8,
        },
    ),
}

QUANT_ACT_MAP = {
    nn.ReLU: (
        qnn.QuantReLU,
        {
            # 'input_quant': Int8ActPerTensorFloat,
            # 'input_bit_width': 8,
            "act_quant": Uint8ActPerTensorFloat,
            "return_quant_tensor": True,
            "bit_width": 7,
        },
    ),
}

QUANT_IDENTITY_MAP = {
    "signed": (qnn.QuantIdentity, {"act_quant": Int8ActPerTensorFloat, "return_quant_tensor": True, "bit_width": 7}),
    "unsigned": (qnn.QuantIdentity, {"act_quant": Uint8ActPerTensorFloat, "return_quant_tensor": True, "bit_width": 7}),
}

model_quant = quantize(
    copy.deepcopy(model),
    compute_layer_map=COMPUTE_LAYER_MAP,
    quant_act_map=QUANT_ACT_MAP,
    quant_identity_map=QUANT_IDENTITY_MAP,
)


def calibrate_model(model, calib_loader, device):
    model.eval()
    model.to(device)
    with torch.no_grad(), calibration_mode(model), tqdm(calib_loader, desc="Calibrating") as pbar:
        for images, _ in pbar:
            images = images.to(device)
            images = images.to(torch.float)
            model(images)


# Load Calibration Dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # Normalize with mean=0.5, std=0.5
)
test_dataset = datasets.MNIST(root=EXPORT_FOLDER / "data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

calibrate_model(model=model_quant, calib_loader=test_loader, device=DEVICE_GPU)

# %% Evaluate the fake-quantized calibrated model
correct = 0
total = 0
model_quant = model_quant.to(DEVICE_CPU)
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model_quant(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test set: {100 * correct / total:.2f}%")


# %% Trace the calibrated fq network
traced_fq_model = brevitas_symbolic_trace(model_quant)

print(traced_fq_model)
print(traced_fq_model.graph)


# %% Export the fake-quantized model (WIP)
# from brevitas.export.inference import quant_inference_mode
# from brevitas.export.onnx.manager import ONNXBaseManager

# ref_input = torch.ones(1, 1, 28, 28, device=DEVICE_CPU, dtype=torch.float)

# with torch.no_grad(), quant_inference_mode(traced_fq_model):
#     traced_fq_model(ref_input)
#     ONNXBaseManager.export(traced_fq_model, args=ref_input, export_path=EXPORT_FOLDER / "03_quant_mnist_base_inf.onnx", opset_version=13)

# JUNGVI: WIP, torch export of the traced model does not work...
# torch.save(traced_fq_model, EXPORT_FOLDER / FQ_MODEL_NAME)
# TorchManager.export(model_quant, args=ref_input, export_path=EXPORT_FOLDER / FQ_MODEL_NAME)
# print(f"Model saved to {EXPORT_FOLDER / FQ_MODEL_NAME}")
