# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Custom Brevitas tracer and tracing function.
"""

import torch.nn as nn
import torch.fx as fx
from brevitas.fx.brevitas_tracer import (
    _symbolic_trace,
    _is_brevitas_leaf_module,
    Tracer,
)
from torch.fx.graph_module import GraphModule

from brevitexporter.custom_forwards.linear import InnerForwardImplWrapperLinear
from brevitas.nn.quant_layer import (
    QuantNonLinearActLayer,
    QuantWeightBiasInputOutputLayer,
)
from brevitas.nn.quant_mha import QuantMultiheadAttention


class CustomBrevitasTracer(Tracer):
    """
    Custom tracer to expose unrolled forward passes.
    """

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        """
        Determines whether 'm' should be treated as a leaf during symbolic tracing.

        Args:
            m: The module to check.
            module_qualified_name: String that identifies the module's name in the graph.

        Returns:
            True if the module should be treated as a leaf, False otherwise.
        """
        if isinstance(m, InnerForwardImplWrapperLinear):
            return True
        if isinstance(m, QuantWeightBiasInputOutputLayer):
            return False
        if isinstance(m, QuantNonLinearActLayer):
            return False
        if isinstance(m, QuantMultiheadAttention):
            return False

        return _is_brevitas_leaf_module(m, module_qualified_name)


def custom_brevitas_trace(root: nn.Module, concrete_args=None) -> GraphModule:
    """
    Helper to create an FX GraphModule using the CustomBrevitasTracer.

    Args:
        root: The root module to trace.
        concrete_args: Optional dictionary of concrete arguments for partial tracing.

    Returns:
        A GraphModule representing the traced computation.
    """
    tracer = CustomBrevitasTracer()
    return _symbolic_trace(tracer, root, concrete_args)
