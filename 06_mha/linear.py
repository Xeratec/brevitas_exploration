import warnings
from pathlib import Path
from sys import modules
from typing import Optional, Tuple, Union

# Suppress warnings about named tensors and __torch_function__
warnings.filterwarnings("ignore", message="Named tensors.*")
warnings.filterwarnings("ignore", message="Defining your.*__torch_function__.*")

import math
import torch
import torch.fx as fx
import torch.nn.functional as F
from torch import Tensor
from torch.fx.graph_module import GraphModule
from torch.nn import Module


### Brevitas Imports ###
from brevitas.fx import brevitas_symbolic_trace
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer
from brevitas.nn.quant_mha import QuantMultiheadAttention
from brevitas.fx import Tracer
from brevitas.fx.brevitas_tracer import _is_brevitas_leaf_module, _symbolic_trace
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias
from brevitas.quant_tensor import QuantTensor

###############################################################################
# Model Definitions
###############################################################################


class ModelQuantLinear(torch.nn.Module):
    """
    A simple model using a quantized linear layer.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.linear = qnn.QuantLinear(
            in_features=in_features,
            out_features=out_features,
            kernel_size=3,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
        )

    def forward(self, inp: Tensor) -> Tensor:
        inp = self.input_quant(inp)
        out = self.linear(inp)
        return out


###############################################################################
# Injection & Custom Forward for QuantLinear
###############################################################################


# --- Wrapper for the inner forward of QuantLinear ---
class InnerForwardImplWrapperLinear(torch.nn.Module):
    def __init__(self, inner_forward_impl):
        super().__init__()
        self.inner_forward_impl = inner_forward_impl

    def forward(self, quant_input, quant_weight, quant_bias):
        return self.inner_forward_impl(quant_input, quant_weight, quant_bias)


# --- Custom forward for QuantLinear ---
def quantWBIOL_forward(self, inp: Tensor) -> Tensor:
    quant_input = self.input_quant(inp)
    quant_weight = self.weight_quant(self.weight)
    quant_bias = self.bias_quant(self.bias, quant_input, quant_weight)
    output = self.wrapped_inner_forward_impl(quant_input, quant_weight, quant_bias)
    quant_output = self.output_quant(output)
    return quant_output


# --- Injector for QuantLinear ---
def quantWBIOL_injector(module: QuantWeightBiasInputOutputLayer) -> QuantWeightBiasInputOutputLayer:
    assert isinstance(
        module, QuantWeightBiasInputOutputLayer
    ), f"{type(module)} is not an instance of QuantWeightBiasInputOutputLayer!"
    module.wrapped_inner_forward_impl = InnerForwardImplWrapperLinear(module.inner_forward_impl)
    module.forward = quantWBIOL_forward.__get__(module)
    return module


###############################################################################
# Custom FX Tracer
###############################################################################


class CustomBrevitasSymbolicTracer(Tracer):
    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        # Do not trace through quant_conv or quant_linear modules.
        if m.__module__.startswith("brevitas.nn.quant_linear"):
            return False
        # Treat the custom linear wrapper as a leaf.
        if isinstance(m, InnerForwardImplWrapperLinear):
            return True
        return _is_brevitas_leaf_module(m, module_qualified_name)


def custom_brevitas_symbolic_trace(root: Module, concrete_args=None) -> GraphModule:
    return _symbolic_trace(CustomBrevitasSymbolicTracer(), root, concrete_args)


###############################################################################
# Export Function
###############################################################################


def transformBrevitasCalibratedModel(model: torch.nn.Module) -> GraphModule:
    """
    Traces the model with Brevitas, injects export-friendly forward functions into
    quantized modules, and re-traces with a custom tracer to expose inner quantization operations.
    """
    # 1. Trace the model with Brevitas modules as leaf nodes.
    fx_model = brevitas_symbolic_trace(model)
    # 2. Inject export-friendly forward functions.
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            target_module = getattr(fx_model, node.target)
            if isinstance(target_module, QuantWeightBiasInputOutputLayer):
                quantWBIOL_injector(target_module)
    # 3. Re-trace with our custom tracer.
    fx_model = custom_brevitas_symbolic_trace(fx_model)
    return fx_model


###############################################################################
# Main: Test and Print FX Graphs for Linear and MHA
###############################################################################


def main():

    DTYPE = torch.float

    # ---------------------------
    # Test QuantLinear
    # ---------------------------
    BATCH_SIZE = 1
    IN_FEATURES = 16
    OUT_FEATURES = 32
    ref_input_linear = torch.randn(BATCH_SIZE, 42, IN_FEATURES, dtype=DTYPE)

    quant_linear = ModelQuantLinear(IN_FEATURES, OUT_FEATURES)
    quant_linear.eval()

    # Print the default FX graph for QuantLinear.
    quant_linear_fx_model = brevitas_symbolic_trace(quant_linear)
    print("\n=== Default FX Graph (Linear) ===")
    quant_linear_fx_model.graph.print_tabular()

    # Export the calibrated model for QuantLinear.
    quant_linear_exported_fx = transformBrevitasCalibratedModel(quant_linear_fx_model)
    print("\n=== Exported FX Graph (Linear) ===")
    quant_linear_exported_fx.graph.print_tabular()

    # Compare outputs (accessing the .value attribute of the QuantTensor).
    out_default = quant_linear_fx_model(ref_input_linear)
    # print("Default output: ", out_default.value)
    out_exported = quant_linear_exported_fx(ref_input_linear)
    # print("Exported output: ", out_exported.value)
    assert out_default.value.equal(out_exported.value), "QuantLinear exported model is not equivalent to the original!"
    print("\u2705 Injector Test Passed (Linear)")


if __name__ == "__main__":
    main()
