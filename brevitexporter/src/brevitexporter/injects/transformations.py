# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Transformation classes for different types of Brevitas modules.

This module provides specific transformation passes for each type of quantized module:
- Linear layers (QuantLinear, QuantConv2d)
- Activation functions (QuantReLU, QuantSigmoid)
- Multi-head attention (QuantMultiheadAttention)

Each transformation class customizes the base TransformationPass with appropriate
module_cls, injection_fn, and validation settings.
"""

import torch.nn as nn
from brevitas.nn.quant_layer import (
    QuantWeightBiasInputOutputLayer,
    QuantNonLinearActLayer,
)
from brevitas.nn.quant_mha import QuantMultiheadAttention

from .base import TransformationPass
from ..custom_forwards.linear import InnerForwardImplWrapperLinear, quantWBIOL_forward
from ..custom_forwards.activations import quant_activation_forward
from ..custom_forwards.multiheadattention import unrolled_quant_mha_forward
from ..custom_tracer import CustomBrevitasTracer


class LinearTransformation(TransformationPass):
    """
    Transformation pass for quantized linear layers (QuantLinear, QuantConv2d).

    Replaces the default forward with an unrolled implementation that exposes
    all quantization steps in the computation graph.
    """

    def __init__(self) -> None:
        """
        Initialize the linear transformation pass.

        Sets up the injection function to:
        1. Install the wrapped inner forward implementation
        2. Replace the forward method
        3. Register appropriate leaf/non-leaf module classes
        """

        def injection_fn(module: nn.Module, tracer: CustomBrevitasTracer) -> None:
            # For Linear layers, wrap the inner forward implementation
            module.wrapped_inner_forward_impl = InnerForwardImplWrapperLinear(
                module.inner_forward_impl
            )
            module.forward = quantWBIOL_forward.__get__(module)
            tracer.register_leaf_module(InnerForwardImplWrapperLinear)
            tracer.register_non_leaf_module(QuantWeightBiasInputOutputLayer)

        super().__init__(
            module_cls=QuantWeightBiasInputOutputLayer,
            injection_fn=injection_fn,
            validation_tol=1e-6,
        )


class ActivationTransformation(TransformationPass):
    """
    Transformation pass for quantized activation functions.

    Replaces the default forward with an unrolled implementation that exposes
    the input quantization and activation quantization steps.
    """

    def __init__(self) -> None:
        """
        Initialize the activation transformation pass.

        Sets up the injection function to:
        1. Replace the forward method
        2. Register the module as non-leaf for proper tracing
        """

        def injection_fn(module: nn.Module, tracer: CustomBrevitasTracer) -> None:
            module.forward = quant_activation_forward.__get__(module)
            tracer.register_non_leaf_module(QuantNonLinearActLayer)

        super().__init__(
            module_cls=QuantNonLinearActLayer,
            injection_fn=injection_fn,
            validation_tol=1e-6,
        )


class MHATransformation(TransformationPass):
    """
    Transformation pass for quantized multi-head attention layers.

    Replaces the default forward with an unrolled implementation that exposes
    all attention operations and their associated quantization steps.
    """

    def __init__(self) -> None:
        """
        Initialize the MHA transformation pass.

        Sets up the injection function to:
        1. Replace the forward method with unrolled implementation
        2. Register the module as non-leaf for proper tracing
        """

        def injection_fn(module: nn.Module, tracer: CustomBrevitasTracer) -> None:
            # For MHA, we need to pass the class itself as the second arg to __get__().
            module.forward = unrolled_quant_mha_forward.__get__(
                module, QuantMultiheadAttention
            )
            tracer.register_non_leaf_module(QuantMultiheadAttention)

        super().__init__(
            module_cls=QuantMultiheadAttention,
            injection_fn=injection_fn,
            validation_tol=1e-5,
        )
