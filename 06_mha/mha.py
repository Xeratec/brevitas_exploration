# Suppress warnings about named tensors and __torch_function__
import warnings

warnings.filterwarnings("ignore", message="Named tensors.*")
warnings.filterwarnings("ignore", message="Defining your.*__torch_function__.*")

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch.fx as fx
from torch.fx.graph_module import GraphModule

###############################################################################
# Import Brevitas modules
###############################################################################
import brevitas.nn as qnn
from brevitas.fx import brevitas_symbolic_trace
from brevitas.fx.brevitas_tracer import _symbolic_trace, _is_brevitas_leaf_module, Tracer


###############################################################################
# Custom export-friendly forward for QuantMultiheadAttention
###############################################################################
def quantMHA_forward(self: qnn.QuantMultiheadAttention, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    Simplified forward for QuantMultiheadAttention that:
      - Assumes input in shape (L, N, E) (no dropout, no masks).
      - Uses separate projections (q_proj, k_proj, v_proj).
      - Unrolls all intermediate operations (scaling, reshapes, quantizations, softmax, etc.).
    Returns only the attention output.
    """
    # --- Input Projections ---
    q = self.q_proj(query)
    k = self.k_proj(key)
    v = self.v_proj(value)

    # --- Reshape for multi-head attention ---
    # Expected input shape: (L, N, E)
    L, N, E = q.shape
    head_dim = E // self.num_heads
    # Reshape from (L, N, E) to (L, N, num_heads, head_dim),
    # then permute to (N, num_heads, L, head_dim) and merge N and num_heads.
    q = q.view(L, N, self.num_heads, head_dim).permute(1, 2, 0, 3).reshape(N * self.num_heads, L, head_dim)
    k = k.view(L, N, self.num_heads, head_dim).permute(1, 2, 0, 3).reshape(N * self.num_heads, L, head_dim)
    v = v.view(L, N, self.num_heads, head_dim).permute(1, 2, 0, 3).reshape(N * self.num_heads, L, head_dim)

    # --- Scale q and apply quantization ---
    q_scaled = q / math.sqrt(head_dim)
    q_scaled = self.q_scaled_quant(q_scaled)

    # --- Transpose and quantize k ---
    k_t = k.transpose(-2, -1)
    k_t = self.k_transposed_quant(k_t)

    # --- Compute attention weights ---
    attn_weights = torch.bmm(q_scaled, k_t)
    attn_weights = self.softmax_input_quant(attn_weights)
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = self.attn_output_weights_quant(attn_weights)

    # --- Quantize v and compute attention output ---
    v = self.v_quant(v)
    attn_output = torch.bmm(attn_weights, v)

    # --- Reshape attention output back to (L, N, E) ---
    attn_output = attn_output.view(N, self.num_heads, L, head_dim).permute(2, 0, 1, 3).reshape(L, N, E)

    # --- Output projection ---
    attn_output = self.out_proj(attn_output)
    return attn_output


###############################################################################
# Injector for QuantMultiheadAttention
###############################################################################
def quantMHA_injector(module: qnn.QuantMultiheadAttention) -> qnn.QuantMultiheadAttention:
    """
    Injects the export-friendly forward into a QuantMultiheadAttention module.
    """
    if not isinstance(module, qnn.QuantMultiheadAttention):
        raise TypeError(f"Module {type(module)} is not a QuantMultiheadAttention!")
    module.forward = quantMHA_forward.__get__(module, qnn.QuantMultiheadAttention)
    return module


###############################################################################
# Injector for QuantLinear (for unrolling inner quantization ops)
###############################################################################
def quant_linear_injector(module: qnn.QuantLinear) -> qnn.QuantLinear:
    """
    Injects a new forward into a QuantLinear module that unrolls its inner quantization operations.
    The new forward calls:
      - input_quant on the input,
      - weight_quant on the module's weight (and bias_quant if bias exists),
      - calls the wrapped inner forward (inner_forward_impl),
      - and then applies output_quant.
    This produces an FX graph similar to your Linear example.
    """
    # If not already wrapped, wrap inner_forward_impl
    if not hasattr(module, "wrapped_inner_forward_impl"):
        module.wrapped_inner_forward_impl = module.inner_forward_impl

    def new_forward(self, inp: Tensor) -> Tensor:
        quant_input = self.input_quant(inp)
        quant_weight = self.weight_quant(self.weight)
        quant_bias = self.bias_quant(self.bias, quant_input, quant_weight) if self.bias is not None else None
        inner_out = self.wrapped_inner_forward_impl(quant_input, quant_weight, quant_bias)
        return self.output_quant(inner_out)

    module.forward = new_forward.__get__(module, type(module))
    return module


###############################################################################
# Custom FX Tracer that inlines QuantMultiheadAttention and QuantLinear
###############################################################################
class CustomBrevitasSymbolicTracer(Tracer):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # Do not treat QuantMultiheadAttention or QuantLinear as leaf modules,
        # so that their internal operations are traced.
        if isinstance(m, qnn.QuantMultiheadAttention):
            return False
        if m.__class__.__name__ == "QuantLinear":
            return False
        return _is_brevitas_leaf_module(m, module_qualified_name)


def custom_brevitas_symbolic_trace(root: nn.Module, concrete_args=None) -> GraphModule:
    return _symbolic_trace(CustomBrevitasSymbolicTracer(), root, concrete_args)


###############################################################################
# Post-tracing graph transformation to remove spurious "rename" calls
###############################################################################
# def remove_rename_calls(g: fx.Graph) -> None:
#     """
#     Iterate over the graph and remove call_method nodes that call "rename"
#     (which may have been inserted automatically but cause errors).
#     For each such node, replace all uses with its input.
#     """
#     nodes_to_remove = []
#     for node in g.nodes:
#         if node.op == "call_method" and node.target == "rename":
#             node.replace_all_uses_with(node.args[0])
#             nodes_to_remove.append(node)
#             print("node: ", node)
#     for node in nodes_to_remove:
#         g.erase_node(node)


###############################################################################
# Transformation function for exporting the calibrated model
###############################################################################
def transformBrevitasCalibratedModel(model: nn.Module) -> GraphModule:
    """
    1. Trace the model using the default Brevitas tracer.
    2. Inject the export-friendly forward into QuantMultiheadAttention modules.
    3. Also inject the new forward into all QuantLinear submodules.
    4. Re-trace the model with a custom tracer that inlines inner operations.
    5. Remove any spurious "rename" calls from the graph.
    """
    # Step 1: Trace the model using the default tracer.
    fx_model = brevitas_symbolic_trace(model)
    # Step 2: For each node, if the module is a QuantMultiheadAttention, inject the new forward.
    for node in fx_model.graph.nodes:
        if node.op == "call_module":
            target_module = getattr(fx_model, node.target)
            if isinstance(target_module, qnn.QuantMultiheadAttention):
                quantMHA_injector(target_module)
    # Step 3: Inject the new forward into all QuantLinear modules.
    for mod in fx_model.modules():
        if isinstance(mod, qnn.QuantLinear):
            quant_linear_injector(mod)
    # Step 4: Re-trace with the custom tracer.
    fx_model = custom_brevitas_symbolic_trace(fx_model)
    # Step 5: Remove "rename" calls from the graph.
    # remove_rename_calls(fx_model.graph)
    # fx_model.graph.lint()
    # fx_model.recompile()
    return fx_model


###############################################################################
# Example model using QuantMultiheadAttention
###############################################################################
class ModelQuantMHA(torch.nn.Module):
    """
    A simple model using a quantized multihead attention module.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.quant_mha = qnn.QuantMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            packed_in_proj=False,  # Use separate projections: q_proj, k_proj, v_proj.
            batch_first=False,  # Using shape (L, N, E)
        )

    def forward(self, inp: Tensor) -> Tensor:
        # Quantize the input before passing it to the MHA module.
        inp = self.input_quant(inp)
        # For simplicity, assume the same quantized input is used for query, key, and value.
        out = self.quant_mha(inp, inp, inp)
        return out


###############################################################################
# Main: Test and print the FX graphs
###############################################################################
def main():
    torch.manual_seed(0)
    # Example parameters
    L = 10  # sequence length
    N = 2  # batch size
    E = 16  # embedding dimension
    num_heads = 4

    # Dummy input with shape (L, N, E)
    inp = torch.randn(L, N, E)

    # Instantiate the model
    model = ModelQuantMHA(embed_dim=E, num_heads=num_heads)
    model.eval()

    # Trace the default model (default FX graph for quant_mha)
    fx_model_default = brevitas_symbolic_trace(model)
    print("\n=== FX Graph Default (QuantMHA) ===")
    fx_model_default.graph.print_tabular()

    # Transform the model for export (inject new forwards and re-trace)
    fx_model_exported = transformBrevitasCalibratedModel(model)
    print("\n=== FX Graph Exported (QuantMHA) ===")
    fx_model_exported.graph.print_tabular()

    # Compare outputs
    out_default = fx_model_default(inp)
    out_exported = fx_model_exported(inp)

    # Print the first element of the outputs
    # print("out_default: ", out_default[0])
    # print("out_exported: ", out_exported[0])

    if torch.allclose(out_default, out_exported, atol=1e-5):
        print("\n \u2705 Injector Test Passed (QuantMHA)")
    else:
        print("\n \u274c Injector Test Failed (QuantMHA)")
        print("Output Default:\n", out_default)
        print("Output Exported:\n", out_exported)


if __name__ == "__main__":
    main()
