# Copyright 2025 ETH Zurich.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# 
# Victor Jung <jungvi@iis.ee.ethz.ch>

### System Imports ###
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

### PyTorch Imports ###
import torch

### Brevitas Import ###
import brevitas.nn as qnn
from brevitas.fx import brevitas_symbolic_trace
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias

### Local Imports ###
from export import exportBrevitasCalibratedModel

### Hyperparameters ###
DTYPE = torch.float

BATCH_SIZE = 1
IN_CH = 3
OUT_CH = 16
IMG_SIZE = 32
IN_FEATURES = 16
OUT_FEATURES = 32

### Model Definition ###

class ModelQuantConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.conv = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
        )

    def forward(self, inp):
        inp = self.input_quant(inp)
        inp = self.conv(inp)
        return inp


class ModelQuantLinear(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.linear = qnn.QuantLinear(
            in_features=in_features,
            out_features=out_features,
            kernel_size=3,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
        )

    def forward(self, inp):
        inp = self.input_quant(inp)
        inp = self.linear(inp)
        return inp


### Input definition and model instanciation ###
ref_input_conv = torch.randn(BATCH_SIZE, IN_CH, IMG_SIZE, IMG_SIZE, dtype=DTYPE)
ref_input_linear = torch.randn(BATCH_SIZE, 42, IN_FEATURES, dtype=DTYPE)

quant_linear = ModelQuantLinear(IN_FEATURES, OUT_FEATURES)
quant_linear.eval()
quant_linear_fx_model = brevitas_symbolic_trace(quant_linear)

quant_conv = ModelQuantConv2d(IN_CH, OUT_CH)
quant_conv.eval()
quant_conv_fx_model = brevitas_symbolic_trace(quant_conv)

### Functional equivalence test ###

original_output_conv = quant_conv_fx_model(ref_input_conv)
original_output_linear = quant_linear_fx_model(ref_input_linear)

quant_conv_exported_fx = exportBrevitasCalibratedModel(quant_conv_fx_model)
quant_linear_exported_fx = exportBrevitasCalibratedModel(quant_linear_fx_model)

assert quant_conv_fx_model(ref_input_conv).value.all() == quant_conv_exported_fx(ref_input_conv).value.all(), "QuantConv2d exported model is not equivalent to the original!"
assert quant_linear_fx_model(ref_input_linear).value.all() == quant_linear_exported_fx(ref_input_linear).value.all(), "QuantLinear exported model is not equivalent to the original!"

print("\u2705 WBIOL Injector Test Passed")
