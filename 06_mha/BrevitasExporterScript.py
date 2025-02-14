# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import warnings
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.graph_module import GraphModule
from torch import Tensor
import torch.nn.functional as F
import math

# -----------------------------------------------------------------------------
# Brevitas imports
# -----------------------------------------------------------------------------
import brevitas.nn as qnn
from brevitas.fx import brevitas_symbolic_trace
from brevitas.fx.brevitas_tracer import (
    _symbolic_trace,
    _is_brevitas_leaf_module,
    Tracer,
)
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias
from brevitas.nn.quant_layer import (
    QuantWeightBiasInputOutputLayer,
    QuantNonLinearActLayer,
)
from brevitas.nn.quant_mha import QuantMultiheadAttention
from brevitas.quant_tensor import QuantTensor

# -----------------------------------------------------------------------------
# Suppress some common PyTorch FX warnings for cleaner console output
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", message="Named tensors.*")
warnings.filterwarnings("ignore", message="Defining your.*__torch_function__.*")
warnings.filterwarnings("ignore", message=".*has_cuda.*")
warnings.filterwarnings("ignore", message=".*has_cudnn.*")
warnings.filterwarnings("ignore", message=".*has_mps.*")
warnings.filterwarnings("ignore", message=".*has_mkldnn.*")
warnings.filterwarnings("ignore", message=".*input x to specialized function.*")


###############################################################################
#         1) CUSTOM FORWARDS FOR QUANTLINEAR, ACTIVATION E MHA                #
###############################################################################


class InnerForwardImplWrapperLinear(nn.Module):
    """
    A small wrapper around the 'inner_forward_impl' of a QuantLinear
    (QuantWeightBiasInputOutputLayer).
    """

    def __init__(self, inner_forward_impl):
        super().__init__()
        self.inner_forward_impl = inner_forward_impl

    def forward(self, quant_input, quant_weight, quant_bias):
        return self.inner_forward_impl(quant_input, quant_weight, quant_bias)


def quantWBIOL_forward(self, inp: Tensor) -> Tensor:
    """
    Unrolled forward pass for a QuantLinear:
      1) self.input_quant
      2) self.weight_quant
      3) self.bias_quant
      4) self.inner_forward_impl (wrapped)
      5) self.output_quant
    """
    quant_input = self.input_quant(inp)
    quant_weight = self.weight_quant(self.weight)
    quant_bias = None
    if self.bias is not None:
        quant_bias = self.bias_quant(self.bias, quant_input, quant_weight)
    output = self.wrapped_inner_forward_impl(quant_input, quant_weight, quant_bias)
    quant_output = self.output_quant(output)
    return quant_output


def quant_activation_forward(
    self, inp: Union[Tensor, QuantTensor]
) -> Union[Tensor, QuantTensor]:
    """
    Unrolled forward pass for a QuantNonLinearActLayer (e.g. QuantSigmoid, QuantReLU):
      1) input_quant
      2) act_quant
    """
    quant_input = self.input_quant(inp)
    quant_output = self.act_quant(quant_input)
    return quant_output


def unrolled_quant_mha_forward(
    self: QuantMultiheadAttention, query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    """
    Export-friendly forward that explicitly unrolls the multi-head logic:
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
    L, N, E = q_out.shape  # (seq_len, batch, embed_dim)
    head_dim = E // self.num_heads

    # reshape e permute
    q_out = (
        q_out.view(L, N, self.num_heads, head_dim)
        .permute(1, 2, 0, 3)
        .reshape(N * self.num_heads, L, head_dim)
    )
    k_out = (
        k_out.view(L, N, self.num_heads, head_dim)
        .permute(1, 2, 0, 3)
        .reshape(N * self.num_heads, L, head_dim)
    )
    v_out = (
        v_out.view(L, N, self.num_heads, head_dim)
        .permute(1, 2, 0, 3)
        .reshape(N * self.num_heads, L, head_dim)
    )

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
    attn_output = (
        attn_output.view(N, self.num_heads, L, head_dim)
        .permute(2, 0, 1, 3)
        .reshape(L, N, E)
    )

    # 8) Out projection
    attn_output = self.out_proj(attn_output)

    return attn_output


###############################################################################
#                      2) INJECT FUNCTIONS AND PASSES                         #
###############################################################################


def inject_unrolled_forward_in_quantlinear(
    module: QuantWeightBiasInputOutputLayer,
) -> None:
    """
    Inject the unrolled forward for QuantLinear
    """
    if not isinstance(module, QuantWeightBiasInputOutputLayer):
        raise TypeError(
            f"Expected QuantWeightBiasInputOutputLayer, found {type(module)}"
        )
    module.wrapped_inner_forward_impl = InnerForwardImplWrapperLinear(
        module.inner_forward_impl
    )
    module.forward = quantWBIOL_forward.__get__(module)


def inject_unrolled_forward_in_activation(module: QuantNonLinearActLayer) -> None:
    """
    Inject the unrolled forward for QuantNonLinearActLayer (activation).
    """
    if not isinstance(module, QuantNonLinearActLayer):
        raise TypeError(f"Expected QuantNonLinearActLayer, found {type(module)}")
    module.forward = quant_activation_forward.__get__(module)


def inject_unrolled_quant_mha(module: QuantMultiheadAttention) -> None:
    """
    Replaces the forward method of a Brevitas QuantMultiheadAttention
    with the unrolled version above.
    """
    if not isinstance(module, QuantMultiheadAttention):
        raise TypeError(f"Expected QuantMultiheadAttention, got {type(module)}")
    module.forward = unrolled_quant_mha_forward.__get__(module, QuantMultiheadAttention)


def transform_linear_pass(model: nn.Module) -> bool:
    """
    Scorre il modello (ricorsivamente) e inietta il forward unrolled in ogni QuantLinear.
    Restituisce True se ha modificato almeno un modulo.
    """
    transformDone = False
    for _, submodule in model.named_modules():
        if isinstance(submodule, QuantWeightBiasInputOutputLayer):
            inject_unrolled_forward_in_quantlinear(submodule)
            transformDone = True
    return transformDone


def transform_activation_pass(model: nn.Module) -> bool:
    """
    Scorre il modello (ricorsivamente) e inietta il forward unrolled in ogni QuantNonLinearActLayer.
    Restituisce True se ha modificato almeno un modulo.
    """
    transformDone = False
    for _, submodule in model.named_modules():
        if isinstance(submodule, QuantNonLinearActLayer):
            inject_unrolled_forward_in_activation(submodule)
            transformDone = True
    return transformDone


def transform_quant_mha_pass(model: nn.Module) -> bool:
    """
    Scorre il modello (ricorsivamente) e inietta l'unrolled forward nel QuantMultiheadAttention
    """
    transformDone = False
    for _, submodule in model.named_modules():
        if isinstance(submodule, QuantMultiheadAttention):
            inject_unrolled_quant_mha(submodule)
            transformDone = True
    return transformDone


###############################################################################
#                           3) CUSTOM BREITVAS TRACER                         #
###############################################################################


class CustomBrevitasTracer(Tracer):
    """
    Custom tracer per esporre i pass unrolled:
    - NON leaf per: QuantNonLinearActLayer, QuantMultiheadAttention, quant_linear
    - LEAF per: InnerForwardImplWrapperLinear
    """

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # Manteniamo la regola che se è un wrapper linear unrolled, lo consideriamo leaf
        if isinstance(m, InnerForwardImplWrapperLinear):
            return True
        # Non leaf per i layer di attivazione (così da esporre le chiamate)
        if isinstance(m, QuantNonLinearActLayer):
            return False
        # Non leaf per quant linear (così da esporre input_quant, etc.)
        if m.__module__.startswith("brevitas.nn.quant_linear"):
            return False
        # Non leaf per il MultiheadAtt
        if isinstance(m, QuantMultiheadAttention):
            return False

        # Per il resto fallback
        return _is_brevitas_leaf_module(m, module_qualified_name)


def custom_brevitas_trace(root: nn.Module, concrete_args=None) -> GraphModule:
    """
    Helper to trace with our CustomBrevitasTracer
    """
    return _symbolic_trace(CustomBrevitasTracer(), root, concrete_args)


###############################################################################
#                       4) FUNZIONE EXPORTBREVITAS (UNICA)                    #
###############################################################################


def exportBrevitas(model, input):
    outputBefore = model(input)

    if transform_quant_mha_pass(model):
        outputAfter = model(input)
        outputBefore = outputBefore[0]  # MHA returns a tuple
        if torch.allclose(outputBefore, outputAfter, atol=1e-5):
            print("✓ MHA pass done! The two outputs match within the tolerance.")
        else:
            raise RuntimeError(
                "The outputs of the unrolled model do NOT match the original one!"
            )

    if transform_linear_pass(model):
        outputAfter = model(input)
        if torch.allclose(outputBefore, outputAfter, atol=1e-6):
            print("✓ Linear pass done! The two outputs match within the tolerance.")
        else:
            raise RuntimeError(
                "The outputs of the unrolled model do NOT match the original one!"
            )

    if transform_activation_pass(model):
        outputAfter = model(input)
        if torch.allclose(outputBefore, outputAfter, atol=1e-6):
            print("✓ Activation pass done! The two outputs match within the tolerance.")
        else:
            raise RuntimeError(
                "The outputs of the unrolled model do NOT match the original one!"
            )

    fx_model = custom_brevitas_trace(model, concrete_args=(input,))

    print("\nAll the transformations have been applied successfully!")
    print("The model has been traced with Brevitas and the FX graph is:\n")
    fx_model.graph.print_tabular()

    return fx_model


###############################################################################
#                                5) MAIN TEST                                 #
###############################################################################


class CombinedModel(nn.Module):
    def __init__(self, in_features=16, out_features=32):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)

        self.linear1 = qnn.QuantLinear(
            in_features=in_features,
            out_features=out_features,
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
            in_features=out_features,
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

    def forward(self, x):
        x = self.input_quant(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


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
        # MHA expects: (query, key, value)
        out = self.mha(x, x, x)  # brevitas version, returns a QuantTensor
        return out


def main():
    torch.manual_seed(42)

    # ESEMPIO 1: CombinedModel con due Linear e due attivazioni
    modelA = CombinedModel().eval()
    dummy_inputA = torch.randn(1, 4, 16)
    print("\n=== ExportBrevitas su CombinedModel ===")
    exportBrevitas(modelA, dummy_inputA)

    # ESEMPIO 2: ExampleQuantMHA con MultiheadAttention
    modelB = ExampleQuantMHA(embed_dim=16, num_heads=4).eval()
    dummy_inputB = torch.randn(10, 2, 16)
    print("\n=== ExportBrevitas su ExampleQuantMHA ===")
    exportBrevitas(modelB, dummy_inputB)


if __name__ == "__main__":
    main()
