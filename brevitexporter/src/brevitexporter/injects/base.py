# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Single transformation class that handles:
- module type matching
- forward injection
- validation (output_before vs output_after)
- applying the transform to all submodules
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Callable, Union, Tuple


class TransformationPass:
    """
    A generic transformation pass:
      - module_cls: the module type(s) to match (e.g. QuantWeightBiasInputOutputLayer)
      - injection_fn: a function that modifies the module's forward
      - validation_tol: numeric tolerance for output comparison
    """

    def __init__(
        self,
        module_cls: Union[type, Tuple[type, ...]],
        injection_fn: Callable[[nn.Module], None],
        validation_tol: float = 1e-6,
    ) -> None:
        """
        Args:
            module_cls: The class (or tuple of classes) this pass should target.
            injection_fn: A callable that receives the module and modifies its forward.
            validation_tol: Tolerance for output comparison in validate_transformation.
        """
        self.module_cls = module_cls
        self.injection_fn = injection_fn
        self.validation_tol = validation_tol

    def check_module_type(self, module: nn.Module) -> bool:
        """
        Returns True if 'module' is an instance of self.module_cls, False otherwise.

        Args:
            module: A PyTorch module to check.

        Returns:
            True if module is an instance of self.module_cls, else False.
        """
        return isinstance(module, self.module_cls)

    def inject_forward(self, module: nn.Module) -> None:
        """
        Calls the user-provided injection_fn to modify the module's forward pass.

        Args:
            module: The module whose forward will be replaced.
        """
        self.injection_fn(module)

    def validate_transformation(
        self, output_before: Any, output_after: Any, atol: Optional[float] = None
    ) -> bool:
        """
        Checks if output_before and output_after match within a tolerance.
        By default, uses self.validation_tol if 'atol' is not specified.

        Args:
            output_before: The output of the model before transformation.
            output_after: The output of the model after transformation.
            atol: Absolute tolerance for comparison.

        Returns:
            True if outputs match within the given tolerance, False otherwise.
        """
        if atol is None:
            atol = self.validation_tol
        return torch.allclose(output_before, output_after, atol=atol)

    def transform(self, model: nn.Module) -> bool:
        """
        Applies the injection to all matching submodules in 'model'.

        Args:
            model: The PyTorch model containing submodules to be transformed.

        Returns:
            True if at least one submodule was modified, False otherwise.
        """
        transform_done = False
        for _, submodule in model.named_modules():
            if self.check_module_type(submodule):
                self.inject_forward(submodule)
                transform_done = True
        return transform_done
