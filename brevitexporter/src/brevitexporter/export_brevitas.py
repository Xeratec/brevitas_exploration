# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Main export functionality for Brevitas quantized networks.

This module provides the primary exportBrevitas function that handles:
- Transformation pass orchestration
- Custom tracer configuration
- FX graph generation
"""

import torch
import torch.nn as nn
from .injects.transformations import (
    LinearTransformation,
    ActivationTransformation,
    MHATransformation,
)
from .injects.executor import TransformationExecutor
from .custom_tracer import CustomBrevitasTracer, custom_brevitas_trace

# ANSI color codes
BLUE = "\033[94m"
ENDC = "\033[0m"


def exportBrevitas(
    model: nn.Module, example_input: torch.Tensor, debug: bool = False
) -> nn.Module:
    """
    Export a Brevitas model to an FX GraphModule with unrolled quantization operations.

    This function applies a series of transformations to make the quantization steps
    explicit in the model's computation graph, then traces the transformed model using
    a custom FX tracer.

    Args:
        model: The Brevitas-based model to export.
        example_input: A representative input tensor for shape tracing.
        debug: If True, prints transformation progress information.

    Returns:
        nn.Module: An FX GraphModule with explicit quantization operations.
    """
    # Create transformation sequence
    transformations = [
        MHATransformation(),
        LinearTransformation(),
        ActivationTransformation(),
    ]

    # Initialize custom tracer
    tracer = CustomBrevitasTracer(debug=debug)

    # Create and execute transformation sequence
    executor = TransformationExecutor(transformations, debug=debug, tracer=tracer)
    transformed_model = executor.execute(model, example_input)

    # Generate FX graph using the same tracer
    fx_model = custom_brevitas_trace(
        transformed_model, concrete_args=(example_input,), tracer=tracer
    )

    if debug:
        print(f"{BLUE} ✓ All transformations completed successfully!{ENDC}")

    return fx_model
