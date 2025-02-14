# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Test file demonstrating an example of a model with Brevitas QuantMultiheadAttention
and exporting it via the exportBrevitas function.
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from torch import Tensor
from brevitexporter.export_brevitas import exportBrevitas


class SimpleQuantMHA(nn.Module):
    """
    Simple example model that includes a Brevitas QuantMultiheadAttention layer.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """
        Args:
            embed_dim: The dimension of each embedding vector.
            num_heads: The number of attention heads.
        """
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.mha = qnn.QuantMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            packed_in_proj=False,  # separate Q, K, V
            batch_first=False,  # expects (sequence, batch, embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass that first quantizes the input, then applies multi-head attention.

        Args:
            x: Input tensor of shape [sequence_len, batch_size, embed_dim].

        Returns:
            A tuple (output, None) as per the Brevitas MHA API, where output has shape
            [sequence_len, batch_size, embed_dim].
        """
        x = self.input_quant(x)
        out = self.mha(x, x, x)  # brevitas version returns (output, None)
        return out


def test_simple_quant_mha() -> None:
    """
    Test function for the SimpleQuantMHA using exportBrevitas.
    Verifies both the forward pass and the export tracing.

    Returns:
        None
    """
    torch.manual_seed(42)

    model = SimpleQuantMHA(embed_dim=16, num_heads=4).eval()
    sample_input = torch.randn(10, 2, 16)  # [sequence=10, batch=2, embed_dim=16]

    print("\n=== ExportBrevitas on a simple QuantMultiheadAttention model ===\n")
    fx_model = exportBrevitas(model, sample_input, debug=True)

    print("Traced Model FX Graph Structure:\n")
    fx_model.graph.print_tabular()

    print("\nTest completed successfully!")
