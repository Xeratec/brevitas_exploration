# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Philip Wiese <wiesep@iis.ee.ethz.ch>

# %% Import and setup model

## System Imports
import os
import copy
import warnings
import gc

## Other imports
from tqdm import tqdm

## Local Imports
from utils import generate_dataloader, validate

### PyTorch Imports ###
import torch
import torch.nn as nn

## Disable some annoying warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch", message=".*experimental feature.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision", message=".*experimental feature.*")
warnings.filterwarnings(
    "ignore", category=torch.jit.TracerWarning, message=".*results are registered as constants in the trace.*"
)
warnings.filterwarnings(
    "ignore",
    category=torch.jit.TracerWarning,
    message=".*Converting a tensor to a Python boolean might cause the trace to be incorrect.*",
)
warnings.filterwarnings(
    "ignore",
    category=torch.jit.TracerWarning,
    message=".*Converting a tensor to a Python number might cause the trace to be incorrect.*",
)

### Brevitas Import ###
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.quantize import quantize
from brevitas.graph.calibrate import calibration_mode
from brevitas.export.inference import quant_inference_mode
from brevitas.export import export_onnx_qcdq
from brevitas.export import export_torch_qcdq
from brevitas.export import export_qonnx

import brevitas.nn as qnn

# %% Setup configuration and data loaders

# DATASETS = os.environ.get('DATASETS')
DATASETS = "/usr/scratch/sassauna1/ml_datasets/"
DTYPE = torch.float
DEVICE_CPU = "cpu"
DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INP_SHAPE = 224
RESIZE_SHAPE = 256
N_VALIDATION_SAMPLES = 1024  # Set to None to use all samples
N_CALIBRATION_SAMPLES = 256

assert DATASETS is not None, "DATASETS environment variable not set"

calib_loader = generate_dataloader(
    os.path.join(DATASETS, "ILSVRC2012/val"),
    batch_size=32,
    num_workers=8,
    resize_shape=RESIZE_SHAPE,
    center_crop_shape=INP_SHAPE,
    subset_size=N_CALIBRATION_SAMPLES,
)

val_loader = generate_dataloader(
    dir=os.path.join(DATASETS, "ILSVRC2012/val"),
    batch_size=32,
    num_workers=8,
    resize_shape=RESIZE_SHAPE,
    center_crop_shape=INP_SHAPE,
    subset_size=N_VALIDATION_SAMPLES,
)

ref_input = torch.ones(1, 3, INP_SHAPE, INP_SHAPE, device=DEVICE_CPU, dtype=DTYPE)


# %% Define useful helper functions
def calibrate_model(model, calib_loader, device):
    model.eval()
    model.to(device)
    with torch.no_grad(), calibration_mode(model), tqdm(calib_loader, desc="Calibrating") as pbar:
        for images, _ in pbar:
            images = images.to(device)
            images = images.to(dtype)
            model(images)


# %% Get the model from torchvision
model = torch.hub.load("pytorch/vision:v0.6.0", model="resnet18", weights="DEFAULT")

model = model.to(DTYPE)
model = model.to(DEVICE_CPU)

dtype = next(model.parameters()).dtype
device = next(model.parameters()).device

print(model)
print("Device: ", device)
print("Dtype: ", dtype)


model.eval()
model = model.to(DEVICE_GPU)
validate(val_loader, model)


# %% Prepare for quantization
from brevitas.graph.per_input import AdaptiveAvgPoolToAvgPool

model.eval()
model.to(DEVICE_CPU)
model = preprocess_for_quantize(model, equalize_iters=20, equalize_scale_computation="range")

# FN_TO_MODULE_MAP = ((torch.add, qnn.QuantEltwiseAdd), (operator.add, qnn.QuantEltwiseAdd), )
# model = TorchFunctionalToModule(fn_to_module_map=FN_TO_MODULE_MAP).apply(model)

model = AdaptiveAvgPoolToAvgPool().apply(model, ref_input)

print(model)
print(model.graph.print_tabular())


# %% Quantize model activation
from brevitas.quant import Int8ActPerTensorFloat

# from brevitas.quant import Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import Uint8ActPerTensorFloat

# from brevitas.quant import Uint8ActPerTensorFloatMaxInit

COMPUTE_LAYER_MAP = {
    nn.AvgPool2d: (qnn.TruncAvgPool2d, {"return_quant_tensor": True}),
    nn.Conv2d: (
        qnn.QuantConv2d,
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

# Free GPU memory
del model
gc.collect()
torch.cuda.empty_cache()

print(model_quant)
print(model_quant.graph.print_tabular())

model_quant.eval()
model_quant = model_quant.to(DEVICE_GPU)
calibrate_model(model_quant, calib_loader, DEVICE_GPU)


# %% Evaluate ResNet model using TorchMetrics
model_quant.eval()
model_quant = model_quant.to(DEVICE_GPU)
validate(val_loader, model_quant)


# %% Export model
model_quant.eval()
model_quant.to(device)


# %% Export QCDQ model
export_onnx_qcdq(model_quant, args=ref_input, export_path="02_quant_model_qcdq.onnx", opset_version=13)
export_torch_qcdq(model_quant, args=ref_input, export_path="02_quant_model_qcdq.pt")


# %% Export QONNX model
export_qonnx(model_quant, args=ref_input, export_path="02_quant_model_qonnx.onnx", opset_version=13)
