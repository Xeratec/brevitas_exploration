# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Transformation classes for different types of Brevitas modules.
Each class inherits from the single TransformationPass in base.py,
customizing module_cls, injection_fn, and validation_tol.
"""

import torch
import torch.nn as nn
from typing import Callable

from brevitas.nn.quant_layer import (
    QuantWeightBiasInputOutputLayer,
    QuantNonLinearActLayer,
)
from brevitas.nn.quant_mha import QuantMultiheadAttention

from .base import TransformationPass
from ..custom_forwards.linear import InnerForwardImplWrapperLinear, quantWBIOL_forward
from ..custom_forwards.activations import quant_activation_forward
from ..custom_forwards.multiheadattention import unrolled_quant_mha_forward


class LinearTransformation(TransformationPass):
    """
    TransformationPass specialized for Brevitas QuantLinear (QuantWeightBiasInputOutputLayer).
    Wraps the existing 'inner_forward_impl' and replaces the forward with 'quantWBIOL_forward'.
    """

    def __init__(self) -> None:
        """
        Initializes a TransformationPass for QuantLinear modules, injecting
        quantWBIOL_forward and setting up the wrapped_inner_forward_impl.
        """

        def injection_fn(module: nn.Module) -> None:
            # For Linear, we need to wrap the existing inner_forward_impl.
            module.wrapped_inner_forward_impl = InnerForwardImplWrapperLinear(
                module.inner_forward_impl
            )
            module.forward = quantWBIOL_forward.__get__(module)

        super().__init__(
            module_cls=QuantWeightBiasInputOutputLayer,
            injection_fn=injection_fn,
            validation_tol=1e-6,
        )


class ActivationTransformation(TransformationPass):
    """
    TransformationPass specialized for Brevitas activation layers (QuantNonLinearActLayer),
    injecting 'quant_activation_forward' as the new forward.
    """

    def __init__(self) -> None:
        """
        Initializes a TransformationPass for QuantNonLinearActLayer modules,
        injecting quant_activation_forward.
        """

        def injection_fn(module: nn.Module) -> None:
            module.forward = quant_activation_forward.__get__(module)

        super().__init__(
            module_cls=QuantNonLinearActLayer,
            injection_fn=injection_fn,
            validation_tol=1e-6,
        )


class MHATransformation(TransformationPass):
    """
    TransformationPass specialized for Brevitas QuantMultiheadAttention,
    injecting 'unrolled_quant_mha_forward' as the new forward.
    """

    def __init__(self) -> None:
        """
        Initializes a TransformationPass for QuantMultiheadAttention modules,
        injecting unrolled_quant_mha_forward with a looser tolerance (1e-5).
        """

        def injection_fn(module: nn.Module) -> None:
            # For MHA, we need to pass the class itself as the second arg to __get__().
            module.forward = unrolled_quant_mha_forward.__get__(
                module, QuantMultiheadAttention
            )

        super().__init__(
            module_cls=QuantMultiheadAttention,
            injection_fn=injection_fn,
            validation_tol=1e-5,  # MHA might need a looser tolerance
        )
