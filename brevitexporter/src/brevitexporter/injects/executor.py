# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Executor class for handling transformation sequences.
"""

import torch
import torch.nn as nn
from typing import List
from .base import TransformationPass

# ANSI color codes
BLUE = "\033[94m"
ENDC = "\033[0m"
CHECK = "✓"


class TransformationExecutor:
    """
    Executes a list of TransformationPass objects in sequence.
    """

    def __init__(
        self, transformations: List[TransformationPass], debug: bool = False
    ) -> None:
        """
        Args:
            transformations: A list of TransformationPass objects.
            debug: If True, prints a success message after each transformation.
        """
        self.transformations = transformations
        self.debug = debug

    def execute(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """
        Executes all transformations in sequence on the provided model.
        Validates the outputs before and after each transformation step.

        Args:
            model: The PyTorch model to be transformed.
            example_input: A sample input for validation after each transformation pass.

        Returns:
            The modified model (in-place transformations).
        """
        model.eval()
        with torch.no_grad():
            output_before = model(example_input)
            if isinstance(output_before, tuple):
                output_before = output_before[0]

            for transformation in self.transformations:
                # transform() modifies any submodules of 'model' that match the pass
                if transformation.transform(model):
                    output_after = model(example_input)
                    if isinstance(output_after, tuple):
                        output_after = output_after[0]

                    # Validate the transformation
                    if not transformation.validate_transformation(
                        output_before, output_after
                    ):
                        raise RuntimeError(
                            f"{transformation.__class__.__name__} failed - outputs mismatch"
                        )

                    # Debug message if everything is fine
                    if self.debug:
                        print(
                            f"{BLUE}{CHECK} {transformation.__class__.__name__} "
                            f"transformation successful - outputs match{ENDC}"
                        )

                    # Update for next pass
                    output_before = output_after

        return model
