# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Module for transforming FX graphs by splitting quantization nodes
into separate quantization and dequantization nodes.
"""

import torch
import torch.fx as fx
from typing import Dict, Any, List, Tuple
from .quant_dequant_nodes import Quant, Dequant
import torch.nn as nn

# ANSI color codes
BLUE = "\033[94m"
ENDC = "\033[0m"


def create_quant_dequant_nodes(
    graph: fx.Graph,
    node: fx.Node,
    fx_model: fx.GraphModule,
    quant_name: str,
    dequant_name: str,
    original_module: nn.Module,
    param_dict: Dict[str, Any],
) -> Tuple[fx.Node, fx.Node]:
    """
    Create Quant and Dequant nodes in the correct topological order.

    Args:
        graph: The FX graph being modified.
        node: The original node being replaced (call_module to "xxx_quant").
        fx_model: The FX GraphModule containing the modules.
        quant_name: Name for the new quant submodule.
        dequant_name: Name for the new dequant submodule.
        original_module: The original Brevitas quant module being replaced.
        param_dict: A dictionary with keys 'scale', 'zero_point', and 'bit_width'.

    Returns:
        A tuple containing the new quant node and dequant node.
    """
    scale_val = param_dict.get("scale", None)
    zp_val = param_dict.get("zero_point", None)
    bw_val = param_dict.get("bit_width", None)

    # Insert new modules into the FX model
    fx_model.add_module(quant_name, Quant(original_module, scale_val, zp_val, bw_val))
    fx_model.add_module(
        dequant_name, Dequant(original_module, scale_val, zp_val, bw_val)
    )

    # Create the quant node after the original node
    with graph.inserting_after(node):
        quant_node = graph.call_module(quant_name, args=node.args)

    # Create the dequant node after the quant node
    with graph.inserting_after(quant_node):
        dequant_node = graph.call_module(dequant_name, args=(quant_node,))

    return quant_node, dequant_node


def split_quant_nodes(
    fx_model: fx.GraphModule, full_params_dict: Dict[str, Dict[str, Any]], debug: bool
) -> fx.GraphModule:
    """
    Transforms an FX graph by splitting each "call_module(...quant...)" node
    into a Quant -> Dequant pair. scale, zero_point, bit_width are read from
    full_params_dict to initialize the modules.

    Args:
        fx_model: The input FX GraphModule to be transformed.
        full_params_dict: A dictionary mapping module names to their quantization
                          parameters (scale, zero_point, bit_width).

    Returns:
        A new FX GraphModule with the original quant calls replaced by
        quant + dequant nodes, each referencing the stored parameters.
    """
    graph = fx_model.graph
    nodes_to_erase: List[fx.Node] = []

    if debug:
        print(f"{BLUE} › Starting Quantization Node Splitting...{ENDC}")

    all_nodes = list(graph.nodes)

    for node in all_nodes:
        if (
            node.op == "call_module"
            and "quant" in node.target.lower()
            and "act_impl" not in node.target.lower()
        ):
            # The original module
            original_module = fx_model.get_submodule(node.target)

            # Build a "safe" name for the new submodules
            safe_target = node.target.replace(".", "_")
            safe_target = safe_target.replace("_quant", "")

            quant_name = f"{safe_target}_quant_1"
            dequant_name = f"{safe_target}_dequant"

            # Fetch parameter info (scale, zero_point, bit_width) if available
            param_info = full_params_dict.get(node.target, {})

            # Create new Quant and Dequant nodes
            quant_node, dequant_node = create_quant_dequant_nodes(
                graph,
                node,
                fx_model,
                quant_name,
                dequant_name,
                original_module,
                param_info,
            )

            # Re-route all users of the original node to the new dequant node
            users = list(node.users.keys())
            for user_node in users:
                new_args = list(user_node.args)
                for i, arg in enumerate(new_args):
                    if arg is node:
                        new_args[i] = dequant_node
                user_node.args = tuple(new_args)

            nodes_to_erase.append(node)

    # Remove the old quant nodes
    for erase_node in nodes_to_erase:
        graph.erase_node(erase_node)

    graph.lint()
    if debug:
        print(f"{BLUE} › Quantization Node Splitting completed Successfully{ENDC}")

    return fx_model
