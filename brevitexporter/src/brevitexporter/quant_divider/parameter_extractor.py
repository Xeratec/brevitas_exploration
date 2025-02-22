# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Module for extracting quantization proxy parameters from an exported FX model.
"""

from typing import Any, Dict
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.export.inference import quant_inference_mode
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias

from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.proxy.parameter_quant import (
    WeightQuantProxyFromInjector,
    BiasQuantProxyFromInjector,
)
from colorama import Fore, Style


def safe_get_scale(quant_obj: Any) -> Any:
    """
    Safely retrieve the scale from a Brevitas proxy object.

    Args:
        quant_obj: A Brevitas proxy object (e.g. ActQuantProxyFromInjector).

    Returns:
        The floating scale value if available, otherwise None.
    """
    if quant_obj is None:
        return None
    maybe_scale = quant_obj.scale() if callable(quant_obj.scale) else quant_obj.scale
    if maybe_scale is None:
        return None
    if isinstance(maybe_scale, torch.Tensor):
        return maybe_scale.item()
    elif isinstance(maybe_scale, float):
        return maybe_scale
    try:
        return float(maybe_scale)
    except Exception:
        return None


def safe_get_zero_point(quant_obj: Any) -> Any:
    """
    Safely retrieve the zero_point from a Brevitas proxy object.

    Args:
        quant_obj: A Brevitas proxy object (e.g. ActQuantProxyFromInjector).

    Returns:
        The floating zero point value if available, otherwise None.
    """
    if quant_obj is None:
        return None
    maybe_zp = (
        quant_obj.zero_point()
        if callable(quant_obj.zero_point)
        else quant_obj.zero_point
    )
    if maybe_zp is None:
        return None
    if isinstance(maybe_zp, torch.Tensor):
        return maybe_zp.item()
    elif isinstance(maybe_zp, float):
        return maybe_zp
    try:
        return float(maybe_zp)
    except Exception:
        return None


def extract_brevitas_proxy_params(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Recursively scan the post-exportBrevitas model to find all submodules of type:
    ActQuantProxyFromInjector, WeightQuantProxyFromInjector, BiasQuantProxyFromInjector,
    and retrieve scale, zero_point, and bit_width for each.

    Args:
        model: A model that has already undergone exportBrevitas (custom forward injection).

    Returns:
        A dictionary of the form:
        {
            'module_name': {
                'scale': float or None,
                'zero_point': float or None,
                'bit_width': float or None
            },
            ...
        }
    """
    params_dict = {}

    def recurse_modules(parent_mod: nn.Module, prefix: str = "") -> None:
        for child_name, child_mod in parent_mod.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            # If it's one of the known proxies, extract params
            if isinstance(
                child_mod,
                (
                    ActQuantProxyFromInjector,
                    WeightQuantProxyFromInjector,
                    BiasQuantProxyFromInjector,
                ),
            ):
                scl = safe_get_scale(child_mod)
                zp = safe_get_zero_point(child_mod)
                bw = child_mod.bit_width()  # .bit_width() is typically a method

                # Save to dictionary under a single "quant_params" sub-key if desired,
                # but here we directly store them for simplicity.
                params_dict[full_name] = {
                    "scale": scl,
                    "zero_point": zp,
                    "bit_width": bw,
                }

            # Recurse deeper
            recurse_modules(child_mod, prefix=full_name)

    recurse_modules(model)
    return params_dict


def print_quant_params(params_dict: Dict[str, Dict[str, Any]]) -> None:
    """
    Print the quantization parameters for each proxy in a color-coded format.

    Args:
        params_dict: Dictionary containing the quantization parameters.

    Example structure:
    {
      'mha.q_proj.weight_quant': {
         'scale': 0.0034,
         'zero_point': 0.0,
         'bit_width': 8.0
      },
      ...
    }
    """
    print(f"\n{Fore.BLUE}Extracted Parameters from the Network:{Style.RESET_ALL}")
    for layer_name, quant_values in params_dict.items():
        print(f"  {Fore.BLUE}{layer_name}:{Style.RESET_ALL}")
        for param_key, param_val in quant_values.items():
            print(f"    {param_key}: {param_val}")
        print()
