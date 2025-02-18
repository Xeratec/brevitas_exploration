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
from ..modules.quant_dequant import PlaceholderQuant, PlaceholderDequant

# ANSI color codes
BLUE = "\033[94m"
ENDC = "\033[0m"


def create_quant_dequant_nodes(
    graph: fx.Graph,
    node: fx.Node,
    fx_model: torch.fx.GraphModule,
    quant_name: str,
    dequant_name: str,
    original_module: torch.nn.Module,
) -> Tuple[fx.Node, fx.Node]:
    """
    Creates quant and dequant nodes in the correct topological order.

    Args:
        graph: The FX graph being modified
        node: The original node being replaced
        fx_model: The FX model containing the modules
        quant_name: Name for the new quant module
        dequant_name: Name for the new dequant module
        original_module: The original quantization module

    Returns:
        A tuple of (quant_node, dequant_node)
    """
    # Insert new modules into the FX model
    fx_model.add_module(quant_name, PlaceholderQuant(original_module))
    fx_model.add_module(dequant_name, PlaceholderDequant(original_module))

    # First create the quant node
    with graph.inserting_after(node):
        quant_node = graph.call_module(quant_name, args=node.args)

    # Then create the dequant node after the quant node
    with graph.inserting_after(quant_node):
        dequant_node = graph.call_module(dequant_name, args=(quant_node,))

    return quant_node, dequant_node


def split_quant_nodes(fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Transforms an FX graph by splitting each quantization node into
    a quantization-dequantization pair.

    Args:
        fx_model: The input FX GraphModule to be transformed.

    Returns:
        A new FX GraphModule with split quantization nodes.
    """
    graph = fx_model.graph

    # Keep track of nodes to be removed
    nodes_to_erase: List[fx.Node] = []

    print(f"{BLUE} ✓ Starting quantization node splitting...{ENDC}")

    # Create a list of all nodes first to avoid modifying while iterating
    all_nodes = list(graph.nodes)

    # First pass: create new nodes and update users
    for node in all_nodes:
        if node.op == "call_module" and "quant" in node.target.lower():
            # Get the original module
            original_module = fx_model.get_submodule(node.target)

            # Create safe module names by replacing dots with underscores
            safe_target = node.target.replace(".", "_")
            quant_name = f"{safe_target}_split_quant"
            dequant_name = f"{safe_target}_split_dequant"

            # Create new nodes ensuring topological order
            quant_node, dequant_node = create_quant_dequant_nodes(
                graph, node, fx_model, quant_name, dequant_name, original_module
            )

            # Update users of the original node to use the dequant node
            users = list(
                node.users.keys()
            )  # Create a list to avoid modification during iteration
            for user in users:
                new_args = list(user.args)
                for i, arg in enumerate(new_args):
                    if arg is node:
                        new_args[i] = dequant_node
                user.args = tuple(new_args)

            # Add to list of nodes to be removed
            nodes_to_erase.append(node)

    # Second pass: remove original nodes after all replacements are done
    for node in nodes_to_erase:
        graph.erase_node(node)

    # Verify graph is valid
    graph.lint()
    print(f"{BLUE} ✓ Quantization node splitting completed successfully{ENDC}")

    return fx_model
