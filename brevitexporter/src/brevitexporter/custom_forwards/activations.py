# Copyright 2025
# ETH Zurich and University of Bologna. Licensed under the Apache License,
# Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Custom forward implementations for Brevitas activation layers, 
e.g., QuantReLU, QuantSigmoid, etc.
"""

from typing import Union
from torch import Tensor
from brevitas.quant_tensor import QuantTensor
from brevitas.nn.quant_layer import QuantNonLinearActLayer


def quant_activation_forward(
    self: QuantNonLinearActLayer, inp: Union[Tensor, QuantTensor]
) -> Union[Tensor, QuantTensor]:
    """
    Unrolled forward pass for a QuantNonLinearActLayer:

    Steps:
      1) self.input_quant
      2) self.act_quant

    Args:
        self: The QuantNonLinearActLayer instance (passed automatically).
        inp: Input Tensor or QuantTensor.

    Returns:
        The output after passing through input_quant and act_quant.
    """
    quant_input = self.input_quant(inp)
    quant_output = self.act_quant(quant_input)
    return quant_output
