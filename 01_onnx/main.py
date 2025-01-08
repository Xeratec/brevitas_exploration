# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Philip Wiese <wiesep@iis.ee.ethz.ch>

# %% Import and setup model

### PyTorch Imports ###
import torch

### Brevitas Import ###
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias
from brevitas.export import export_onnx_qcdq, export_qonnx
import brevitas.nn as qnn

# %% Setup model

IN_CH = 3
IMG_SIZE = 128
OUT_CH = 128
BATCH_SIZE = 1


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.linear = qnn.QuantConv2d(
            IN_CH,
            OUT_CH,
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


inp = torch.randn(BATCH_SIZE, IN_CH, IMG_SIZE, IMG_SIZE)
model = Model()
model.eval()

export_onnx_qcdq(model, args=inp, export_path="01_quant_model_qcdq.onnx", opset_version=13)

export_qonnx(model, args=inp, export_path="01_quant_model_qonnx.onnx", opset_version=13)
