# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Philip Wiese <wiesep@iis.ee.ethz.ch>

# %% Import and setup model
from pathlib import Path

### PyTorch Imports ###
import torch

### Brevitas Import ###
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias
from brevitas.export import export_onnx_qcdq, export_qonnx
from brevitas.export.inference import quant_inference_mode
from brevitas.export.onnx.manager import ONNXBaseManager

# %% Setup model

EXPORT_FOLDER = Path().cwd()
# JUNGVI: Make sure we can run this script from anywhere and the exported ONNX still end up in the right folder
if Path().cwd().name != "01_onnx":
    EXPORT_FOLDER = EXPORT_FOLDER / "01_onnx"

DTYPE = torch.float

IN_CH = 3
IMG_SIZE = 128
OUT_CH = 128
BATCH_SIZE = 1

ref_input = torch.ones(1, IN_CH, IMG_SIZE, IMG_SIZE, dtype=DTYPE)


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

ONNXBaseManager.target_name = 'StdONNX'

# JUNGVI: Showcase the different exports from brevitas.
# Inference mode exports a static quantization version of the graph with atomic ONNX nodes.
with torch.no_grad(), quant_inference_mode(model):
    model(inp)
    ONNXBaseManager.export(
        model, args=inp, export_path=EXPORT_FOLDER / "01_quant_model_base_inf.onnx", opset_version=13
    )

ONNXBaseManager.export(model, args=inp, export_path=EXPORT_FOLDER / "01_quant_model_base.onnx", opset_version=13)
export_onnx_qcdq(model, args=inp, export_path=EXPORT_FOLDER / "01_quant_model_qcdq.onnx", opset_version=13)
export_qonnx(model, args=inp, export_path=EXPORT_FOLDER / "01_quant_model_qonnx.onnx", opset_version=13)
