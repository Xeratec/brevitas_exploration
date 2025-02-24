# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import torch
import torch.nn as nn
from pathlib import Path

from .injects.transformations import (
    LinearTransformation,
    ActivationTransformation,
    MHATransformation,
)
from .injects.executor import TransformationExecutor
from .custom_tracer import CustomBrevitasTracer, custom_brevitas_trace
from .quant_divider.parameter_extractor import (
    extract_brevitas_proxy_params,
    print_quant_params,
)
from .quant_divider.quant_nodes_divider import split_quant_nodes
from brevitas.export.inference import quant_inference_mode
from brevitas.export import export_onnx_qcdq, export_qonnx
from brevitas.export.onnx.manager import ONNXBaseManager
from torch.fx.graph_module import GraphModule

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

    Note:
        The ONNX export commands are commented out. Do not remove these comments.
    """

    from brevitas.fx import brevitas_symbolic_trace

    model = brevitas_symbolic_trace(model)
    if debug:
        print("\n=== Original Network ===\n")
        model.graph.print_tabular()
        print()

    # Create transformation sequence
    transformations = [
        MHATransformation(),
        LinearTransformation(),
        ActivationTransformation(),
    ]

    with torch.no_grad(), quant_inference_mode(model):
        model(example_input)

    # EXPORT_FOLDER = Path().cwd()
    # print(EXPORT_FOLDER)
    # if Path().cwd().name != "onnx":
    #     EXPORT_FOLDER = EXPORT_FOLDER / "onnx"

    # export_onnx_qcdq(
    #     model,
    #     args=example_input,
    #     export_path=EXPORT_FOLDER / "model_qcdq.onnx",
    #     opset_version=13,
    # )
    # export_qonnx(
    #     model,
    #     args=example_input,
    #     export_path=EXPORT_FOLDER / "model_qonnx.onnx",
    #     opset_version=13,
    # )

    # Initialize custom tracer
    tracer = CustomBrevitasTracer(debug=debug)

    # Create and execute transformation sequence
    executor = TransformationExecutor(transformations, debug=debug, tracer=tracer)
    transformed_model = executor.execute(model, example_input)

    # Generate FX graph using the same tracer
    fx_model = custom_brevitas_trace(
        transformed_model, concrete_args=(example_input,), tracer=tracer
    )
    fx_model.recompile()
    output_fx_model = fx_model(example_input)

    if debug:
        print(f"{BLUE} ✓ All transformations completed successfully!{ENDC}")

    if debug:
        print("\n=== Network after the Transformation ===\n")
        fx_model.graph.print_tabular()

    # Extract the parameters from the network
    proxy_params = extract_brevitas_proxy_params(fx_model)

    if debug:
        print_quant_params(proxy_params)

    # At the end of exportBrevitas, after we have fx_model, split quant nodes
    split_fx_model = split_quant_nodes(fx_model, proxy_params, debug)
    split_fx_model.recompile()
    output_split_fx_model = split_fx_model(example_input)

    if debug:
        print("\n=== Network after the Split of Quant Nodes ===\n")
        fx_model.graph.print_tabular()
        print()

    if torch.allclose(output_fx_model, output_split_fx_model, atol=1e-5):
        if debug:
            print(f"{BLUE} ✓ Passed: final graph output matches the default.{ENDC}")

    # export_onnx_qcdq(
    #     split_fx_model,
    #     args=example_input,
    #     export_path=EXPORT_FOLDER / "transformed_model_qcdq.onnx",
    #     opset_version=13,
    # )
    # export_qonnx(
    #     split_fx_model,
    #     args=example_input,
    #     export_path=EXPORT_FOLDER / "transformed_model_qonnx.onnx",
    #     opset_version=13,
    # )

    return split_fx_model
