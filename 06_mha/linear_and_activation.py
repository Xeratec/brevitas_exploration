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

###############################################################################
#                          1) UNROLLING QUANT LINEAR                          #
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


def quantWBIOL_injector(module: QuantWeightBiasInputOutputLayer) -> None:
    """
    Replaces the forward method of a QuantLinear (QuantWeightBiasInputOutputLayer)
    with the unrolled version that exposes the various quantization steps.
    """
    if not isinstance(module, QuantWeightBiasInputOutputLayer):
        raise TypeError(
            f"Expected a QuantWeightBiasInputOutputLayer, found: {type(module)}."
        )

    module.wrapped_inner_forward_impl = InnerForwardImplWrapperLinear(
        module.inner_forward_impl
    )
    module.forward = quantWBIOL_forward.__get__(module)


class CustomBrevitasSymbolicTracer(Tracer):
    """
    Custom tracer that does NOT treat modules from brevitas.nn.quant_linear as leaf nodes,
    in order to expose the internal quantization calls.
    """

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # If it is a QuantLinear (inside quant_linear), do NOT treat it as a leaf
        if m.__module__.startswith("brevitas.nn.quant_linear"):
            return False
        # If it is our wrapper, consider it as a leaf
        if isinstance(m, InnerForwardImplWrapperLinear):
            return True
        return _is_brevitas_leaf_module(m, module_qualified_name)


def custom_brevitas_symbolic_trace(root: nn.Module, concrete_args=None) -> GraphModule:
    """
    Shortcut to call _symbolic_trace with our custom tracer.
    """
    return _symbolic_trace(CustomBrevitasSymbolicTracer(), root, concrete_args)


def transform_brevitas_quant_linear_model(model: nn.Module) -> GraphModule:
    """
    1. Trace using brevitas_symbolic_trace (default).
    2. Inject the unrolled forward into the found QuantLinear modules.
    3. Re-trace with the custom tracer.
    """
    # 1) Initial trace
    fx_model = brevitas_symbolic_trace(model)

    # 2) Inject unrolled forward
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            target_module = fx_model.get_submodule(node.target)
            if isinstance(target_module, QuantWeightBiasInputOutputLayer):
                quantWBIOL_injector(target_module)

    # 3) Re-trace with custom tracer
    fx_model = custom_brevitas_symbolic_trace(fx_model, concrete_args=None)
    return fx_model


###############################################################################
#                      2) UNROLLING QUANT ACTIVATION (e.g. Sigmoid)            #
###############################################################################


def quant_activation_forward(
    self, input: Union[Tensor, QuantTensor]
) -> Union[Tensor, QuantTensor]:
    """
    Unrolled forward pass for a QuantNonLinearActLayer (e.g. QuantSigmoid):
      1) Unpack input
      2) input_quant
      3) act_quant
      4) Pack output
    """
    quant_input = self.input_quant(input)
    quant_output = self.act_quant(quant_input)
    return quant_output


def quant_activation_injector(module: QuantNonLinearActLayer) -> None:
    """
    Replaces the forward method of a quantized activation module
    (QuantNonLinearActLayer, e.g., QuantSigmoid) with the unrolled version.
    """
    if not isinstance(module, QuantNonLinearActLayer):
        raise TypeError(f"Expected a QuantNonLinearActLayer, found: {type(module)}.")

    module.forward = quant_activation_forward.__get__(module)


class CustomBrevitasActivationTracer(Tracer):
    """
    Custom tracer that does NOT treat QuantNonLinearActLayer as a leaf node,
    so as to expose the calls to input_quant and act_quant.
    """

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, QuantNonLinearActLayer):
            return False
        if isinstance(m, InnerForwardImplWrapperLinear):
            return True
        return _is_brevitas_leaf_module(m, module_qualified_name)


def custom_brevitas_activation_trace(
    root: nn.Module, concrete_args=None
) -> GraphModule:
    """
    Helper to trace with our CustomBrevitasActivationTracer.
    """
    return _symbolic_trace(CustomBrevitasActivationTracer(), root, concrete_args)


class CustomBrevitasTracer(Tracer):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, InnerForwardImplWrapperLinear):
            return True
        return _is_brevitas_leaf_module(m, module_qualified_name)


def custom_brevitas_trace(root: nn.Module, concrete_args=None) -> GraphModule:
    """
    Helper to trace with our CustomBrevitasActivationTracer.
    """
    return _symbolic_trace(CustomBrevitasTracer(), root, concrete_args)


def transform_brevitas_quant_activation_model(model: nn.Module) -> GraphModule:
    """
    1. Trace using brevitas_symbolic_trace (default).
    2. Inject the unrolled forward into the quantized activation modules.
    3. Re-trace with the custom tracer.
    """

    fx_model = custom_brevitas_trace(model, concrete_args=None)

    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            target_module = fx_model.get_submodule(node.target)
            if isinstance(target_module, QuantNonLinearActLayer):
                quant_activation_injector(target_module)

    fx_model = custom_brevitas_activation_trace(fx_model, concrete_args=None)
    return fx_model


###############################################################################
#                3) COMBINED MODEL: QuantLinear + QuantSigmoid                #
###############################################################################


class CombinedModel(nn.Module):
    """
    A simple model that performs:
      - input quantization
      - quantized linear layer
      - quantized sigmoid activation
    """

    def __init__(self, in_features=16, out_features=32):
        super().__init__()
        # Example of an initial quant identity (optional)
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)

        # QuantLinear with parameters similar to the example
        self.linear = qnn.QuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # QuantSigmoid configured to return a QuantTensor
        # (e.g. act_quant=Int8ActPerTensorFloat). If you leave act_quant=None,
        # you might encounter "QuantLayer is not correctly configured" when
        # return_quant_tensor=True.
        self.sigmoid = qnn.QuantSigmoid(
            act_quant=Int8ActPerTensorFloat,
            input_quant=None,  # do not re-quantize the input, use it "as is"
            return_quant_tensor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


###############################################################################
#                                4) MAIN TEST                                 #
###############################################################################


def main():
    torch.manual_seed(42)
    model = CombinedModel().eval()

    # Example input
    BATCH_SIZE = 1
    SEQ_LENGTH = 4
    IN_FEATURES = 16
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, IN_FEATURES)

    # 1) Default FX graph (without unrolling)
    fx_default = brevitas_symbolic_trace(model)
    print("\n=== Default FX Graph ===")
    fx_default.graph.print_tabular()

    # 2) Unroll the QuantLinear
    fx_unrolled_linear = transform_brevitas_quant_linear_model(model)
    print("\n=== After Unrolling QuantLinear ===")
    fx_unrolled_linear.graph.print_tabular()

    # 3) Unroll the QuantSigmoid
    fx_unrolled_final = transform_brevitas_quant_activation_model(fx_unrolled_linear)
    print("\n=== After Also Unrolling QuantSigmoid ===")
    fx_unrolled_final.graph.print_tabular()

    # 4) Compare the outputs
    #    - out_default: from the default graph
    #    - out_unrolled_final: from the graph after both unrolls
    out_default = fx_default(dummy_input)
    out_unrolled_final = fx_unrolled_final(dummy_input)

    # Compare them (both should be QuantTensor):
    default_val = (
        out_default.value if isinstance(out_default, QuantTensor) else out_default
    )
    unrolled_val = (
        out_unrolled_final.value
        if isinstance(out_unrolled_final, QuantTensor)
        else out_unrolled_final
    )

    if torch.allclose(default_val, unrolled_val, atol=1e-6):
        print("\n✓ Test passed! The two outputs match within the tolerance.")
    else:
        raise RuntimeError(
            "The outputs of the unrolled model do NOT match the original one!"
        )


if __name__ == "__main__":
    main()
