# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import Optional
import torch
import torch.nn as nn
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
from brevitas.fx.brevitas_tracer import _symbolic_trace, _is_brevitas_leaf_module, Tracer
from brevitas.nn.quant_activation import QuantSigmoid, QuantReLU, QuantTanh, QuantHardTanh
from brevitas.nn.quant_layer import QuantNonLinearActLayer

# -----------------------------------------------------------------------------
# 1) Unrolled Forward for Quant Activation
# -----------------------------------------------------------------------------
def quant_activation_forward(self, input: torch.Tensor) -> torch.Tensor:
    """
    Unrolled forward for a Quant Activation layer (e.g., QuantSigmoid, QuantReLU, etc.)
    
    Steps:
      1. Unpack the input (if it is a QuantTensor).
      2. Quantize the input using input_quant.
      3. Apply activation quantization via act_quant.
      4. Pack the output.
    """
    # 1) Quantize the input
    quant_input = self.input_quant(input)
    # 2) Apply the activation quantization
    quant_output = self.act_quant(quant_input)
    return quant_output

# -----------------------------------------------------------------------------
# 2) Injector: Replace the forward of Quant Activation modules
# -----------------------------------------------------------------------------
def quant_activation_injector(module: QuantNonLinearActLayer) -> None:
    """
    Replaces the forward method of a Quant Activation layer with the unrolled version.
    """
    if not isinstance(module, QuantNonLinearActLayer):
        raise TypeError(f"Expected a QuantNonLinearActLayer, got {type(module)}.")
    # Bind the new forward to the module instance
    module.forward = quant_activation_forward.__get__(module)

# -----------------------------------------------------------------------------
# 3) Custom FX Tracer for Quant Activation
# -----------------------------------------------------------------------------
class CustomBrevitasActivationTracer(Tracer):
    """
    Custom FX tracer that does not treat Quant Activation layers as leaf modules,
    so that calls to input_quant and act_quant are visible in the FX graph.
    """
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, QuantNonLinearActLayer):
            return False
        return _is_brevitas_leaf_module(m, module_qualified_name)

def custom_brevitas_activation_trace(root: nn.Module, concrete_args=None) -> GraphModule:
    """
    Helper function to perform FX tracing using the custom tracer.
    """
    return _symbolic_trace(CustomBrevitasActivationTracer(), root, concrete_args)

# -----------------------------------------------------------------------------
# 4) Transformation: Unroll the Quant Activation Model
# -----------------------------------------------------------------------------
def transform_brevitas_quant_activation_model(model: nn.Module) -> GraphModule:
    """
    1. Trace the model using the standard Brevitas tracer.
    2. Inject the unrolled forward into each Quant Activation layer.
    3. Retrace the model using the custom tracer.
    """
    # Step 1: Standard Brevitas tracing
    fx_model = brevitas_symbolic_trace(model)
    
    # Step 2: Inject the unrolled forward into Quant Activation layers
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            # Use get_submodule to handle nested module names
            target_module = fx_model.get_submodule(node.target)
            if isinstance(target_module, QuantNonLinearActLayer):
                quant_activation_injector(target_module)
    
    # Step 3: Retrace using the custom tracer
    fx_model = custom_brevitas_activation_trace(fx_model, concrete_args=None)
    return fx_model

# -----------------------------------------------------------------------------
# 5) Test Models: Separate model for each activation function.
# -----------------------------------------------------------------------------
class ModelQuantSigmoid(nn.Module):
    """
    A simple model that uses QuantSigmoid.
    """
    def __init__(self):
        super().__init__()
        self.act = QuantSigmoid(return_quant_tensor=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)

class ModelQuantReLU(nn.Module):
    """
    A simple model that uses QuantReLU.
    """
    def __init__(self):
        super().__init__()
        self.act = QuantReLU(return_quant_tensor=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)

class ModelQuantTanh(nn.Module):
    """
    A simple model that uses QuantTanh.
    """
    def __init__(self):
        super().__init__()
        self.act = QuantTanh(return_quant_tensor=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)

class ModelQuantHardTanh(nn.Module):
    """
    A simple model that uses QuantHardTanh.
    
    Note: To avoid dependency errors for the default quant type
    (Int8ActPerTensorFloatMinMaxInit), we explicitly set the range (min_val and max_val).
    """
    def __init__(self):
        super().__init__()
        self.act = QuantHardTanh(return_quant_tensor=False, max_val=1.0, min_val=-1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)

# -----------------------------------------------------------------------------
# 6) Main Test: Compare Default and Unrolled FX Graphs for each activation
# -----------------------------------------------------------------------------
def test_activation_model(model: nn.Module, model_name: str, dummy_input: torch.Tensor):
    print(f"\n--- Testing {model_name} Model ---")
    model = model.eval()
    
    # Get the default FX graph (activation as a black box)
    fx_default = brevitas_symbolic_trace(model)
    print(f"\n=== Default FX Graph ({model_name}) ===")
    fx_default.graph.print_tabular()
    
    # Transform the model to unroll the Quant Activation steps
    fx_unrolled = transform_brevitas_quant_activation_model(model)
    print(f"\n=== Exported (Unrolled) FX Graph ({model_name}) ===")
    fx_unrolled.graph.print_tabular()
    
    # Compare the outputs
    out_default = fx_default(dummy_input)
    out_unrolled = fx_unrolled(dummy_input)
    
    if torch.allclose(out_default, out_unrolled, atol=1e-6):
        print(f"\n✓ {model_name} test passed! Outputs match.")
    else:
        raise RuntimeError(f"Unrolled {model_name} model does not produce the same output as default!")

def main():
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 16)  # Example input
    
    # Test QuantSigmoid
    test_activation_model(ModelQuantSigmoid(), "QuantSigmoid", dummy_input)
    
    # Test QuantReLU
    test_activation_model(ModelQuantReLU(), "QuantReLU", dummy_input)
    
    # Test QuantTanh
    test_activation_model(ModelQuantTanh(), "QuantTanh", dummy_input)
    
    # Test QuantHardTanh (with proper min/max values)
    test_activation_model(ModelQuantHardTanh(), "QuantHardTanh", dummy_input)

if __name__ == "__main__":
    main()
