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
    Fake-quant module that applies a "saturating" approach using scale, zero_point, and bit_width
    parameters extracted from a Brevitas parameter dictionary.

    This module is used in the quantization export process to simulate quantization effects
    on tensors by scaling, shifting, rounding, and clamping their values.
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
            original_module: The original Brevitas quant module (kept for reference).
            scale: Scale factor used for quantization.
            zero_point: Zero-point used for quantization.
            bit_width: Bit width for the quantized representation (e.g., 8.0, 32.0).
        """
        super().__init__()
        self.original_module = original_module
        self.scale = scale
        self.zero_point = zero_point
        self.bit_width = bit_width

        # Precompute clamping bounds based on the bit width and zero_point.
        if self.bit_width is not None:
            bw_int = int(self.bit_width)
            if abs(self.zero_point) < 1e-5:
                # For unsigned quantization, the range is [0, 2^bw - 1].
                self.min_val = 0
                self.max_val = (2**bw_int) - 1
            else:
                # For signed quantization, the range is [-2^(bw-1), 2^(bw-1) - 1].
                self.min_val = -(2 ** (bw_int - 1))
                self.max_val = (2 ** (bw_int - 1)) - 1
        else:
            self.min_val = None
            self.max_val = None

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Apply fake quantization to the input tensor.

        This method takes variable positional and keyword arguments.
        The first positional argument is expected to be the input tensor.
        Any additional arguments are ignored. This design choice is made to ensure
        compatibility with FX graph nodes, which may pass extra arguments that are not
        needed for the quantization computation.

        The quantization process is as follows:
          1) Scale the input tensor by 1/scale.
          2) Shift the scaled tensor by the zero_point.
          3) Round the shifted tensor to the nearest integer.
          4) Clamp the rounded tensor to the representable range based on bit_width.
          5) Reconstruct the dequantized tensor by reversing the shift and scale.

        Args:
            *args: The first argument should be the input tensor. Additional positional
                   arguments, if any, are ignored.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            The fake quantized tensor.
        """
        if not args:
            raise ValueError(
                "Expected at least one positional argument for the input tensor"
            )
        # Use the first argument as the input tensor.
        x = args[0]
        if self.scale is None or self.zero_point is None:
            return x

        # Step 1: Scale the input tensor.
        x_scaled = x / self.scale
        # Step 2: Shift the scaled tensor by the zero_point.
        x_shifted = x_scaled + self.zero_point
        # Step 3: Round the shifted tensor to the nearest integer.
        x_rounded = torch.round(x_shifted)
        # Step 4: Clamp the rounded values to the representable range.
        if self.bit_width is not None:
            x_rounded = torch.clamp(x_rounded, self.min_val, self.max_val)
        # Step 5: Reconstruct the dequantized tensor.
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
