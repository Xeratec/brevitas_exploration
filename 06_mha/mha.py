# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import warnings

warnings.filterwarnings("ignore", message="Named tensors.*")
warnings.filterwarnings("ignore", message="Defining your.*__torch_function__.*")

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.fx as fx
from torch.fx.graph_module import GraphModule
from torch.nn import Module
from typing import Optional, Tuple

# ----------------------------------------------------------------------------
# Brevitas imports
# ----------------------------------------------------------------------------
import brevitas.nn as qnn
from brevitas.fx import brevitas_symbolic_trace
from brevitas.fx.brevitas_tracer import _symbolic_trace, _is_brevitas_leaf_module, Tracer
from brevitas.nn.quant_mha import QuantMultiheadAttention
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer


# ----------------------------------------------------------------------------
# 1) Unrolled QuantMultiheadAttention Forward + Injector
# ----------------------------------------------------------------------------


def unrolled_quant_mha_forward(self: QuantMultiheadAttention, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    Export-friendly forward that explicitly unrolls the multihead logic:
      - Q/K/V projections
      - Reshapes & permutes for multi-head
      - Scales the queries
      - Applies softmax and intermediate quantizations
      - Out projection
    """
    # 1) Q, K, V projections
    q_out = self.q_proj(query)
    k_out = self.k_proj(key)
    v_out = self.v_proj(value)

    # 2) Multi-head reshape
    L, N, E = q_out.shape  # (sequence_len, batch_size, embed_dim)
    head_dim = E // self.num_heads

    q_out = q_out.view(L, N, self.num_heads, head_dim).permute(1, 2, 0, 3).reshape(N * self.num_heads, L, head_dim)
    k_out = k_out.view(L, N, self.num_heads, head_dim).permute(1, 2, 0, 3).reshape(N * self.num_heads, L, head_dim)
    v_out = v_out.view(L, N, self.num_heads, head_dim).permute(1, 2, 0, 3).reshape(N * self.num_heads, L, head_dim)

    # 3) Scale Q, then quantize
    q_scaled = q_out / math.sqrt(head_dim)
    q_scaled = self.q_scaled_quant(q_scaled)

    # 4) Transpose + quantize K
    k_t = k_out.transpose(-2, -1)
    k_t = self.k_transposed_quant(k_t)

    # 5) Compute attention weights
    attn_weights = torch.bmm(q_scaled, k_t)
    attn_weights = self.softmax_input_quant(attn_weights)
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = self.attn_output_weights_quant(attn_weights)

    # 6) Quantize V + bmm
    v_out = self.v_quant(v_out)
    attn_output = torch.bmm(attn_weights, v_out)

    # 7) Reshape back to (L, N, E)
    attn_output = attn_output.view(N, self.num_heads, L, head_dim).permute(2, 0, 1, 3).reshape(L, N, E)

    # 8) Out projection
    attn_output = self.out_proj(attn_output)

    return attn_output


def inject_unrolled_quant_mha(module: QuantMultiheadAttention) -> None:
    """
    Replaces the forward method of a Brevitas QuantMultiheadAttention
    with the unrolled version above.
    """
    if not isinstance(module, QuantMultiheadAttention):
        raise TypeError(f"Expected a QuantMultiheadAttention, got {type(module)}.")
    module.forward = unrolled_quant_mha_forward.__get__(module, QuantMultiheadAttention)


# ----------------------------------------------------------------------------
# 2) Unrolled QuantLinear Forward + Injector
# ----------------------------------------------------------------------------


class InnerForwardWrapperLinear(nn.Module):
    """
    Wrapper around the original 'inner_forward_impl' from Brevitas.
    We keep it here if we want to preserve that logic or reference it.
    """

    def __init__(self, inner_forward_impl):
        super().__init__()
        self.inner_forward_impl = inner_forward_impl

    def forward(self, quant_input, quant_weight, quant_bias):
        return self.inner_forward_impl(quant_input, quant_weight, quant_bias)


def unrolled_quant_linear_forward(self: QuantWeightBiasInputOutputLayer, inp: Tensor) -> Tensor:
    """
    Unrolled forward for a Brevitas QuantLinear:
      - input_quant
      - weight_quant
      - bias_quant
      - inner_forward_impl
      - output_quant
    """
    quant_input = self.input_quant(inp)
    quant_weight = self.weight_quant(self.weight)
    quant_bias = self.bias_quant(self.bias, quant_input, quant_weight)
    output = self.wrapped_inner_forward_impl(quant_input, quant_weight, quant_bias)
    quant_output = self.output_quant(output)
    return quant_output


def inject_unrolled_quant_linear(module: QuantWeightBiasInputOutputLayer) -> None:
    """
    Replaces the forward method of a Brevitas QuantLinear
    with an unrolled version exposing all quant steps.
    """
    if not isinstance(module, QuantWeightBiasInputOutputLayer):
        return

    # Wrap original inner_forward_impl
    module.wrapped_inner_forward_impl = InnerForwardWrapperLinear(module.inner_forward_impl)
    # Override forward
    module.forward = unrolled_quant_linear_forward.__get__(module)


# ----------------------------------------------------------------------------
# 3) Custom Tracers
# ----------------------------------------------------------------------------


class MHAUnrollTracer(Tracer):
    """
    Tracer that does NOT treat QuantMultiheadAttention as a leaf.
    So, it will expand the MHA logic in the FX graph.

    But it DOES treat QuantLinear as a leaf,
    so we do NOT unroll q_proj, k_proj, v_proj, out_proj by default.
    """

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, QuantMultiheadAttention):
            return False  # Expand it
        if isinstance(m, qnn.QuantLinear):
            return True  # Keep it as black-box
        return _is_brevitas_leaf_module(m, module_qualified_name)


class LinearUnrollTracer(Tracer):
    """
    Tracer that does NOT treat Brevitas QuantLinear as a leaf,
    exposing its internal input/weight/bias/output quantization.
    """

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # Expand all modules from brevitas.nn.quant_linear
        if m.__module__.startswith("brevitas.nn.quant_linear"):
            return False
        # Keep the 'InnerForwardWrapperLinear' as leaf
        if isinstance(m, InnerForwardWrapperLinear):
            return True
        return _is_brevitas_leaf_module(m, module_qualified_name)


# ----------------------------------------------------------------------------
# 4) Transform Functions
# ----------------------------------------------------------------------------


def transform_quant_mha(model: nn.Module) -> GraphModule:
    """
    1) Trace with brevitas_symbolic_trace.
    2) Inject unrolled MHA forward into each QuantMultiheadAttention.
    3) Re-trace with MHAUnrollTracer.
    """
    # First pass: default Brevitas FX trace
    fx_model = brevitas_symbolic_trace(model)

    # Inject unrolled MHA forward
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            submod = fx_model.get_submodule(node.target)
            if isinstance(submod, QuantMultiheadAttention):
                inject_unrolled_quant_mha(submod)

    # Second pass: custom tracer unrolling MHA
    # IMPORTANT: pass 'concrete_args=None' to satisfy _symbolic_trace signature
    fx_model = _symbolic_trace(MHAUnrollTracer(), fx_model, concrete_args=None)
    return fx_model


def transform_quant_linear(model: nn.Module) -> GraphModule:
    """
    1) Trace with brevitas_symbolic_trace.
    2) Inject unrolled forward into each QuantLinear.
    3) Re-trace with LinearUnrollTracer.
    """
    # First pass: default Brevitas FX trace
    fx_model = brevitas_symbolic_trace(model)

    # Inject unrolled linear forward
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            submod = fx_model.get_submodule(node.target)
            if isinstance(submod, QuantWeightBiasInputOutputLayer):
                inject_unrolled_quant_linear(submod)

    # Second pass: custom tracer for linear
    fx_model = _symbolic_trace(LinearUnrollTracer(), fx_model, concrete_args=None)
    return fx_model


# ----------------------------------------------------------------------------
# 5) Example Model that uses MHA
# ----------------------------------------------------------------------------


class ExampleQuantMHA(nn.Module):
    """
    Simple example model with a Brevitas QuantMultiheadAttention.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.mha = qnn.QuantMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            packed_in_proj=False,  # separate Q, K, V
            batch_first=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_quant(x)
        out = self.mha(x, x, x)
        return out


# ----------------------------------------------------------------------------
# 6) Main Test
# ----------------------------------------------------------------------------


def main():
    """
    Demonstration steps:
      1) Default trace (everything as black-box).
      2) Unroll the MHA logic (but keep the internal QuantLinear as black-box).
      3) Unroll the QuantLinear modules inside the partially unrolled model.
      4) Compare outputs to ensure numerical equivalence.
    """
    torch.manual_seed(42)

    model = ExampleQuantMHA(embed_dim=16, num_heads=4).eval()
    sample_input = torch.randn(10, 2, 16)

    # 1) Default trace
    fx_default = brevitas_symbolic_trace(model)
    print("\n=== Default FX Graph (MHA black-box) ===")
    fx_default.graph.print_tabular()

    # 2) Transform to unroll MHA only
    fx_mha_unrolled = transform_quant_mha(model)
    print("\n=== FX Graph: MHA unrolled, QuantLinear still black-box ===")
    fx_mha_unrolled.graph.print_tabular()

    # 3) Further unroll the QuantLinear modules in that partially unrolled MHA
    fx_final = transform_quant_linear(fx_mha_unrolled)
    print("\n=== FX Graph: MHA unrolled + QuantLinear unrolled ===")
    fx_final.graph.print_tabular()

    # 4) Compare outputs
    out_default = fx_default(sample_input)
    out_final = fx_final(sample_input)

    if torch.allclose(out_default, out_final, atol=1e-5):
        print("\n✓ Test passed: final graph output matches the default.")
    else:
        print("\n✗ Test failed: numerical mismatch.")


if __name__ == "__main__":
    main()
