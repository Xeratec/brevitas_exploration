import warnings
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.fx as fx
from torch.fx.graph_module import GraphModule

# -----------------------------------------------------------------------------
# Suppress some common PyTorch FX warnings for cleaner console output
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", message="Named tensors.*")
warnings.filterwarnings("ignore", message="Defining your.*__torch_function__.*")

# -----------------------------------------------------------------------------
# Brevitas imports 
# -----------------------------------------------------------------------------
import brevitas.nn as qnn
from brevitas.fx import brevitas_symbolic_trace
from brevitas.fx.brevitas_tracer import (
    _symbolic_trace,
    _is_brevitas_leaf_module,
    Tracer
)
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias
from brevitas.quant_tensor import QuantTensor

# -----------------------------------------------------------------------------
# 1) Model with a single Brevitas QuantLinear
# -----------------------------------------------------------------------------

class ModelQuantLinear(nn.Module):
    """
    A simple example model that uses a Brevitas QuantLinear, which by default
    hides internal quantization (input_quant, weight_quant, bias_quant, output_quant)
    inside a single call_module. We want to "unroll" these for export.
    """
    def __init__(self, in_features: int, out_features: int):
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

# -----------------------------------------------------------------------------
# 2) Unrolled Forward for QuantLinear + Injection
# -----------------------------------------------------------------------------

class InnerForwardImplWrapperLinear(nn.Module):
    """
    A small wrapper around the existing Brevitas 'inner_forward_impl' of a QuantLinear.
    """
    def __init__(self, inner_forward_impl):
        super().__init__()
        self.inner_forward_impl = inner_forward_impl

    def forward(self, quant_input, quant_weight, quant_bias):
        # Calls the original, internal matmul/bias logic
        return self.inner_forward_impl(quant_input, quant_weight, quant_bias)


def quantWBIOL_forward(self, inp: Tensor) -> Tensor:
    """
    Unrolled forward for a Brevitas QuantLinear, exposing:
      - self.input_quant
      - self.weight_quant
      - self.bias_quant
      - self.inner_forward_impl  (wrapped)
      - self.output_quant
    """
    # 1) Quantize input
    quant_input = self.input_quant(inp)
    # 2) Quantize weight
    quant_weight = self.weight_quant(self.weight)
    # 3) Quantize bias
    quant_bias = self.bias_quant(self.bias, quant_input, quant_weight)
    # 4) Matmul + bias (the original forward) via a wrapped inner_forward_impl
    output = self.wrapped_inner_forward_impl(quant_input, quant_weight, quant_bias)
    # 5) Quantize output
    quant_output = self.output_quant(output)
    return quant_output


def quantWBIOL_injector(module: QuantWeightBiasInputOutputLayer) -> None:
    """
    Replaces the forward method of a Brevitas QuantLinear (QuantWeightBiasInputOutputLayer)
    with an unrolled version that exposes the internal quantization steps.
    """
    if not isinstance(module, QuantWeightBiasInputOutputLayer):
        raise TypeError(f"Expected a QuantWeightBiasInputOutputLayer, got {type(module)}.")

    # Wrap the original inner_forward_impl
    module.wrapped_inner_forward_impl = InnerForwardImplWrapperLinear(module.inner_forward_impl)
    # Override forward
    module.forward = quantWBIOL_forward.__get__(module)

# -----------------------------------------------------------------------------
# 3) Custom FX Tracer to expand brevitas.nn.quant_linear
# -----------------------------------------------------------------------------

class CustomBrevitasSymbolicTracer(Tracer):
    """
    A custom tracer that does not treat Brevitas QuantLinear modules as leaf
    (so we can see all quantization calls). We do treat the small wrapper
    module as leaf to keep it a single call_module.
    """
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # If this is a module from 'brevitas.nn.quant_linear', expand it
        if m.__module__.startswith("brevitas.nn.quant_linear"):
            return False
        # If it's our custom 'InnerForwardImplWrapperLinear', keep it as a leaf
        if isinstance(m, InnerForwardImplWrapperLinear):
            return True
        return _is_brevitas_leaf_module(m, module_qualified_name)


def custom_brevitas_symbolic_trace(
    root: nn.Module,
    concrete_args=None
) -> GraphModule:
    """
    Shortcut function to call the low-level _symbolic_trace with our custom tracer.
    """
    return _symbolic_trace(CustomBrevitasSymbolicTracer(), root, concrete_args)

# -----------------------------------------------------------------------------
# 4) Transformation: Unroll QuantLinear
# -----------------------------------------------------------------------------

def transform_brevitas_quant_linear_model(model: nn.Module) -> GraphModule:
    """
    1. Trace the model with the default Brevitas tracer (brevitas_symbolic_trace).
    2. Inject the unrolled forward logic into each QuantLinear found.
    3. Re-trace with our custom tracer (custom_brevitas_symbolic_trace) 
       so the final FX graph explicitly shows each quantization step.
    """
    # Step 1: default Brevitas trace
    fx_model = brevitas_symbolic_trace(model)

    # Step 2: inject unrolled forward for each QuantLinear found
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            # Use fx_model.get_submodule(node.target) to handle nested names
            target_module = fx_model.get_submodule(node.target)
            if isinstance(target_module, QuantWeightBiasInputOutputLayer):
                quantWBIOL_injector(target_module)

    # Step 3: Re-trace with our custom tracer
    fx_model = custom_brevitas_symbolic_trace(fx_model, concrete_args=None)
    return fx_model


# -----------------------------------------------------------------------------
# 5) Main test
# -----------------------------------------------------------------------------

def main():
    # Set a manual seed for reproducibility
    torch.manual_seed(42)

    # Example input shape
    BATCH_SIZE = 1
    SEQ_LENGTH = 42
    IN_FEATURES = 16

    # Create an instance of the model
    model = ModelQuantLinear(in_features=IN_FEATURES, out_features=32).eval()
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, IN_FEATURES)

    # 1) Show default FX graph (QuantLinear as black-box)
    fx_default = brevitas_symbolic_trace(model)
    print("\n=== Default FX Graph (Linear) ===")
    fx_default.graph.print_tabular()

    # 2) Transform to unroll QuantLinear
    fx_unrolled = transform_brevitas_quant_linear_model(model)
    print("\n=== Exported (Unrolled) FX Graph (Linear) ===")
    fx_unrolled.graph.print_tabular()

    # 3) Compare outputs
    out_default = fx_default(dummy_input)
    out_unrolled = fx_unrolled(dummy_input)
    # Both are QuantTensor objects; compare their .value
    if out_default.value.equal(out_unrolled.value):
        print("\n✓ Test passed! Outputs match.")
    else:
        raise RuntimeError(
            "Unrolled QuantLinear model does not produce the same output as default!"
        )

if __name__ == "__main__":
    main()
