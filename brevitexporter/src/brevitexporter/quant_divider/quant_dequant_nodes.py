# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Basic implementation of Quant and Dequant modules.
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Union


class Quant(nn.Module):
    """
    Fake-quant module that applies a “saturating” approach using scale/zero_point/bit_width
    extracted from a Brevitas param dictionary.
    """

    def __init__(
        self,
        original_module: nn.Module,
        scale: float,
        zero_point: float,
        bit_width: float,
    ) -> None:
        """
        Initialize the Quant module.

        Args:
            original_module: The original Brevitas quant module (unused here, but kept for reference).
            scale: Scale factor from extracted parameters (None if not available).
            zero_point: Zero-point from extracted parameters (None if not available).
            bit_width: Bit width from extracted parameters (e.g. 8.0, 32.0).
        """
        super().__init__()
        self.original_module = original_module
        self.scale = scale
        self.zero_point = zero_point
        self.bit_width = bit_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fake quantization to the input tensor.

        The process is:
          1) Scale the input by (1/scale) and shift by zero_point.
          2) Round to the nearest integer.
          3) Clamp to the representable range determined by bit_width.
          4) Reconstruct the float value.

        Args:
            x: Input tensor to be fake quantized.

        Returns:
            The fake quantized tensor.
        """
        # If scale or zero_point are None, pass through
        if self.scale is None or self.zero_point is None:
            return x

        # 1) Convert x to "scaled" domain
        x_scaled = x / self.scale
        # 2) Shift by zero_point
        x_shifted = x_scaled + self.zero_point
        # 3) Round
        x_rounded = torch.round(x_shifted)

        # 3b) If we have a valid bit_width, clamp the integer domain
        #    Distinguish between a “signed” range or “unsigned” range if needed.
        #    Below is a simple guess: if zero_point == 0 => assume unsigned, else signed.
        if self.bit_width is not None:
            bw_int = int(self.bit_width)
            if abs(self.zero_point) < 1e-5:
                # Assume unsigned range: [0, 2^bw - 1]
                min_val = 0
                max_val = (2**bw_int) - 1
            else:
                # Assume symmetric signed range: [-2^(bw-1), 2^(bw-1) - 1]
                min_val = -(2 ** (bw_int - 1))
                max_val = (2 ** (bw_int - 1)) - 1

            x_rounded = torch.clamp(x_rounded, min_val, max_val)

        # 4) Reconstruct the float tensor from quantized values.
        x_dequant = (x_rounded - self.zero_point) * self.scale
        return x_dequant


class Dequant(nn.Module):
    """
    Dequant module that re-applies scale and zero_point to invert the integer domain.
    """

    def __init__(
        self,
        original_module: nn.Module,
        scale: float,
        zero_point: float,
        bit_width: float,
    ) -> None:
        """
        Initialize the Dequant module.

        Args:
            original_module: The original Brevitas quant module.
            scale: Scale factor from extracted parameters.
            zero_point: Zero-point from extracted parameters.
            bit_width: Bit width from extracted parameters.
        """
        super().__init__()
        self.original_module = original_module
        self.scale = scale
        self.zero_point = zero_point
        self.bit_width = bit_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Undo the fake quantization by shifting and scaling the input.
        """
        if self.scale is None or self.zero_point is None:
            return x

        # In a real integer scenario:
        # x_deq = (x - zero_point) * scale
        # But since x is already float, just do it anyway:
        x_dequant = (x - self.zero_point) * self.scale
        return x_dequant
