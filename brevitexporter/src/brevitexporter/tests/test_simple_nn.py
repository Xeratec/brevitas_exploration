# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Test file demonstrating an example of a small neural network
(Linear + ReLU + Linear + Sigmoid) and exporting it via the exportBrevitas function.
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias
from brevitexporter.export_brevitas import exportBrevitas
from brevitexporter.transform.graph_transformer import split_quant_nodes

# from brevitexporter.transform.graph_transformer import split_quant_nodes

# %% Import and setup model
from pathlib import Path

### PyTorch Imports ###
import torch

### Brevitas Import ###
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias
from brevitas.export import export_onnx_qcdq, export_qonnx
from brevitas.export.inference import quant_inference_mode
from brevitas.export.onnx.manager import ONNXBaseManager


class SimpleQuantNN(nn.Module):
    """
    A simple model that includes:
      - a QuantIdentity
      - two QuantLinear layers
      - two activation layers (QuantReLU and QuantSigmoid)
    """

    def __init__(self, in_features: int = 16, hidden_features: int = 32) -> None:
        """
        Args:
            in_features: Number of input features for the first linear layer.
            hidden_features: Number of output features from the first linear,
                             and input features for the second linear.
        """
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)

        self.linear1 = qnn.QuantLinear(
            in_features=in_features,
            out_features=hidden_features,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        self.relu = qnn.QuantReLU(
            bit_width=4,
            return_quant_tensor=True,
        )

        self.linear2 = qnn.QuantLinear(
            in_features=hidden_features,
            out_features=1,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        self.sigmoid = qnn.QuantSigmoid(
            bit_width=4,
            return_quant_tensor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantized neural network.

        Args:
            x: Input tensor of shape [batch_size, any_dim, in_features].

        Returns:
            A tensor with shape [batch_size, any_dim, 1] after the final sigmoid.
        """
        x = self.input_quant(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


def test_simple_quant_nn() -> None:
    """
    Test function for the SimpleQuantNN using exportBrevitas.
    Tests both the model's functionality and the export process.

    Returns:
        None
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize model in eval mode
    model = SimpleQuantNN().eval()
    sample_input = torch.randn(1, 4, 16)  # [batch=1, 4, 16 features]

    EXPORT_FOLDER = Path().cwd()
    print(EXPORT_FOLDER)
    if Path().cwd().name != "onnx":
        EXPORT_FOLDER = EXPORT_FOLDER / "onnx"

    # export_onnx_qcdq(
    #     model,
    #     args=sample_input,
    #     export_path=EXPORT_FOLDER / "01_quant_model_qcdq.onnx",
    #     opset_version=13,
    # )
    # export_qonnx(
    #     model,
    #     args=sample_input,
    #     export_path=EXPORT_FOLDER / "01_quant_model_qonnx.onnx",
    #     opset_version=13,
    # )

    # Export the model using Brevitas
    print("\n=== ExportBrevitas on a simple Quant Neural Network ===\n")
    fx_model = exportBrevitas(model, sample_input, debug=True)

    print("\nOriginal Model FX Graph Structure:\n")
    fx_model.graph.print_tabular()
    fx_model_output = fx_model(sample_input)

    # export_onnx_qcdq(
    #     fx_model,
    #     args=sample_input,
    #     export_path=EXPORT_FOLDER / "exported_01_quant_model_qcdq.onnx",
    #     opset_version=13,
    # )
    # export_qonnx(
    #     fx_model,
    #     args=sample_input,
    #     export_path=EXPORT_FOLDER / "exported_01_quant_model_qonnx.onnx",
    #     opset_version=13,
    # )

    print("\n=== Transforming Graph ===\n")
    transformed_model = split_quant_nodes(fx_model)

    print("\nTransformed Model FX Graph Structure:\n")
    transformed_model.graph.print_tabular()
    transformed_model_output = transformed_model(sample_input)

    export_onnx_qcdq(
        transformed_model,
        args=sample_input,
        export_path=EXPORT_FOLDER / "transformed_01_quant_model_qcdq.onnx",
        opset_version=13,
    )
    export_qonnx(
        transformed_model,
        args=sample_input,
        export_path=EXPORT_FOLDER / "transformed_01_quant_model_qonnx.onnx",
        opset_version=13,
    )

    # Check if the output are the same
    assert torch.allclose(fx_model_output, transformed_model_output, atol=1e-6)

    print("\nTest completed successfully!")
