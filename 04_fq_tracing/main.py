# Copyright 2025 ETH Zurich.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Victor Jung <jungvi@iis.ee.ethz.ch>


# %% Import and setup model
import copy
from pathlib import Path

### PyTorch Imports ###
import brevitas
import torch

### Brevitas Import ###
from brevitas.fx import brevitas_symbolic_trace
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer

### Local Imports ###
from tracer import custom_brevitas_symbolic_trace
from tracer import InnerForwardImplWrapper
from models import ModelQuantConv2d, ModelQuantLinear


EXPORT_FOLDER = Path().cwd()
# JUNGVI: Make sure we can run this script from anywhere and the exported ONNX still end up in the right folder
if Path().cwd().name != "04_fq_tracing":
    EXPORT_FOLDER = EXPORT_FOLDER / "04_fq_tracing"

### Hyperparameters ###
DTYPE = torch.float

BATCH_SIZE = 1
IN_CH = 3
OUT_CH = 16
IMG_SIZE = 32
IN_FEATURES = 16
OUT_FEATURES = 32

ref_input_conv = torch.randn(BATCH_SIZE, IN_CH, IMG_SIZE, IMG_SIZE, dtype=DTYPE)
ref_input_linear = torch.randn(BATCH_SIZE, 42, IN_FEATURES, dtype=DTYPE)

quant_linear = ModelQuantLinear(IN_FEATURES, OUT_FEATURES)
quant_linear.eval()

quant_conv = ModelQuantConv2d(IN_CH, OUT_CH)
quant_conv.eval()


def quantWBIOL_forward(self, inp):
    quant_input = self.input_quant(inp)
    quant_weight = self.weight_quant(self.weight)
    quant_bias = self.bias_quant(self.bias, quant_input, quant_weight)

    output = self.wrapped_inner_forward_impl(quant_input, quant_weight, quant_bias)

    quant_output = self.output_quant(output)
    return quant_output


# JUNGVI: Inject a new forward function for the layers using QuantWBIO
traced_quant_conv = brevitas_symbolic_trace(quant_conv)
for node in traced_quant_conv.graph.nodes:
    if node.op == "call_module":
        target_module = getattr(traced_quant_conv, node.target)
        if isinstance(target_module, QuantWeightBiasInputOutputLayer):
            target_module.wrapped_inner_forward_impl = InnerForwardImplWrapper(target_module.inner_forward_impl)
            target_module.forward = quantWBIOL_forward.__get__(target_module)
export_ready_quant_conv = custom_brevitas_symbolic_trace(quant_conv)

traced_quant_linear = brevitas_symbolic_trace(quant_linear)
for node in traced_quant_linear.graph.nodes:
    if node.op == "call_module":
        target_module = getattr(traced_quant_linear, node.target)
        if isinstance(target_module, QuantWeightBiasInputOutputLayer):
            target_module.wrapped_inner_forward_impl = InnerForwardImplWrapper(target_module.inner_forward_impl)
            target_module.forward = quantWBIOL_forward.__get__(target_module)
export_ready_quant_linear = custom_brevitas_symbolic_trace(quant_linear)


# JUNGVI: Look at the new graphs and test that they are still functionally equivalent to the original graph
print(traced_quant_conv.graph)
print(export_ready_quant_conv.graph)
assert traced_quant_conv(ref_input_conv).value.all() == export_ready_quant_conv(ref_input_conv).value.all()

print(traced_quant_linear.graph)
print(export_ready_quant_linear.graph)
assert traced_quant_linear(ref_input_linear).value.all() == export_ready_quant_linear(ref_input_linear).value.all()
