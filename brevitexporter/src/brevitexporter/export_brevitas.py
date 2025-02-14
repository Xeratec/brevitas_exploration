# Copyright
# 2025 ETH Zurich and University of Bologna. Licensed under the Apache License,
# Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Main export function that applies the injection passes and performs FX tracing
using CustomBrevitasTracer.
"""

import torch
import torch.nn as nn
from .injects.transformations import (
    LinearTransformation,
    ActivationTransformation,
    MHATransformation,
)
from .injects.executor import TransformationExecutor
from .custom_tracer import custom_brevitas_trace

# ANSI color codes
BLUE = "\033[94m"
ENDC = "\033[0m"
CHECK = "✓"


def exportBrevitas(
    model: nn.Module, example_input: torch.Tensor, debug: bool = False
) -> nn.Module:
    """
    Export a Brevitas-based model to an FX GraphModule with unrolled quantization steps.

    Args:
        model: The Brevitas-based PyTorch model to export.
        example_input: A representative input tensor (used for shape inference).
        debug: If True, prints transformation progress and a success message at the end.

    Returns:
        An FX GraphModule of the transformed model, with unrolled quantization steps.
    """
    # Define transformation sequence
    transformations = [
        MHATransformation(),
        LinearTransformation(),
        ActivationTransformation(),
    ]

    # Create and execute the transformation sequence
    executor = TransformationExecutor(transformations, debug=debug)
    transformed_model = executor.execute(model, example_input)

    # Perform final FX tracing
    fx_model = custom_brevitas_trace(transformed_model, concrete_args=(example_input,))

    if debug:
        print(f"\n{BLUE}All transformations completed successfully!{ENDC}")

    return fx_model
