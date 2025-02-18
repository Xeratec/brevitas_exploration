# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Basic implementation of Quant and Dequant modules.
These are placeholder implementations that will be enhanced later
with proper quantization parameters from Brevitas.
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Union


class PlaceholderQuant(nn.Module):
    """
    Placeholder quantization module. Will be enhanced with proper quantization
    parameters from Brevitas in future implementations.
    """

    def __init__(self, original_module: nn.Module) -> None:
        """
        Initialize the quantization module.

        Args:
            original_module: The original Brevitas quantization module
                           from which parameters will be extracted later.
        """
        super().__init__()
        self.original_module = original_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantization to the input tensor.

        Args:
            x: Input tensor to be quantized.

        Returns:
            The quantized tensor (currently just returns input unchanged).
        """
        return x


class PlaceholderDequant(nn.Module):
    """
    Placeholder dequantization module. Will be enhanced with proper dequantization
    parameters from Brevitas in future implementations.
    """

    def __init__(self, original_module: nn.Module) -> None:
        """
        Initialize the dequantization module.

        Args:
            original_module: The original Brevitas quantization module
                           from which parameters will be extracted later.
        """
        super().__init__()
        self.original_module = original_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dequantization to the input tensor.

        Args:
            x: Input tensor to be dequantized.

        Returns:
            The dequantized tensor (currently just returns input unchanged).
        """
        return x
