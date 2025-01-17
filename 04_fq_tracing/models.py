# Copyright 2025 ETH Zurich.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Victor Jung <jungvi@iis.ee.ethz.ch>


### PyTorch Imports ###
import torch

### Brevitas Import ###
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias


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
