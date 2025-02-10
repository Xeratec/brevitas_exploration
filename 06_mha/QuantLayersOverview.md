# Brevitas Quant Layers: Complete Overview

Below is the complete list of the main **quantized layers** and related modules in the current Brevitas codebase. Each entry is documented with its inheritance, functionality, and key properties.

1. [QuantNonLinearActLayer](#1-quantnonlinearactlayer)
2. [QuantInputOutputLayer](#2-quantinputoutputlayer)
3. [QuantWeightBiasInputOutputLayer](#3-quantweightbiasinputoutputlayer)
4. [QuantLinear](#4-quantlinear)
5. [QuantConv1d](#5-quantconv1d)
6. [QuantConv2d](#6-quantconv2d)
7. [QuantConv3d](#7-quantconv3d)
8. [QuantMultiheadAttention](#8-quantmultiheadattention)
9. [QuantReLU](#9-quantrelu)
10. [QuantSigmoid](#10-quantsigmoid)
11. [QuantTanh](#11-quanttanh)
12. [QuantHardTanh](#12-quanthardtanh)
13. [QuantIdentity](#13-quantidentity)
14. [QuantScaleBias](#14-quantscalebias) (in the same file there is also `ScaleBias`)
15. [TruncAvgPool2d](#15-truncavgpool2d)
16. [TruncAdaptiveAvgPool2d](#16-truncadaptiveavgpool2d)
17. [\_BatchNormToQuantScaleBias](#17-_batchnormtoquantscalebias)
18. [BatchNorm1dToQuantScaleBias](#18-batchnorm1dtoquantscalebias)
19. [BatchNorm2dToQuantScaleBias](#19-batchnorm2dtoquantscalebias)
20. [QuantEltwiseAdd](#20-quanteltwiseadd)
21. [QuantCat](#21-quantcat)
22. [TruncQuantAccumulator](#22-truncquantaccumulator)
23. [ClampQuantAccumulator](#23-clampquantaccumulator)
24. [QuantUpsample](#24-quantupsample)
25. [QuantUpsamplingBilinear2d](#25-quantupsamplingbilinear2d)
26. [QuantUpsamplingNearest2d](#26-quantupsamplingnearest2d)
27. [QuantConvTranspose1d](#27-quantconvtranspose1d)
28. [QuantConvTranspose2d](#28-quantconvtranspose2d)
29. [QuantConvTranspose3d](#29-quantconvtranspose3d)
30. [QuantEmbedding](#30-quantembedding)
31. [QuantRNN](#31-quantrnn)
32. [QuantLSTM](#32-quantlstm)
33. [HadamardClassifier](#33-hadamardclassifier)
34. [EqualizedModule](#34-equalizedmodule)

---

## 1. `QuantNonLinearActLayer`

**Inheritance**

- Inherits from:
  - `QuantNonLinearActMixin`
  - `QuantInputMixin`
  - `QuantLayerMixin`
  - `Module` (PyTorch)

**What it does**

- Handles **quantization of the input** and **non-linear activation** (e.g., ReLU, Sigmoid) in a single module.
- The constructor initializes:
  - An activation implementation (`act_impl`) if needed.
  - `passthrough_act` to determine whether the activation is bypassed or not.
  - `input_quant` to quantize incoming data.
  - `act_quant` to quantize the activation output.
- In the `forward` pass:
  1. Unpacks the input if it’s a `QuantTensor`.
  2. Quantizes the input (`self.input_quant(input)`).
  3. Applies the activation quantization (`self.act_quant(...)`).
  4. Re-packs the output if needed.
- Supports exporting through `export_mode`. If `export_mode` is `True`, it calls `export_handler`.
- `channelwise_separable` is `True`, meaning it can handle channelwise operations separately.

---

## 2. `QuantInputOutputLayer`

**Inheritance**

- Inherits from:
  - `QuantOutputMixin`
  - `QuantInputMixin`
  - `QuantLayerMixin`

**What it does**

- Provides **quantization at both input and output** of a layer.
- The constructor initializes:
  - `input_quant` for incoming data quantization.
  - `output_quant` for outgoing data quantization.
  - `tie_input_output_quant` to share the same quantization parameters for input and output if desired.
  - `return_quant_tensor` to indicate whether to return a `QuantTensor` or plain `Tensor`.
- The property `requires_export_handler` checks if any input/output quantizers are active, controlling export logic.

In essence, `QuantInputOutputLayer` is a **base layer** for modules that require **both** input and output quantization but **do not** necessarily have parameters such as weights or bias.

---

## 3. `QuantWeightBiasInputOutputLayer`

**Inheritance**

- Inherits from:
  - `QuantBiasMixin`
  - `QuantWeightMixin`
  - `QuantInputOutputLayer`

**What it does**

- Extends `QuantInputOutputLayer` by **adding quantization for weights and bias**.
- Used for layers like quantized linear or convolution layers, where parameters need quantization.
- The constructor receives:
  - `weight_quant` for quantizing the layer’s weights.
  - `bias_quant` for quantizing the bias.
- Implements two key **abstract methods**:
  - `inner_forward_impl(self, x, quant_weight, quant_bias)`: the underlying operation (e.g., `matmul` or `conv`).
  - `max_acc_bit_width(self, input_bit_width, quant_weight_bit_width)`: optional logic for accumulator bit-width.

In the forward pass:

1. Unpacks the input if it’s a `QuantTensor`.
2. Quantizes the input, weights (and bias if present).
3. Executes `inner_forward_impl`.
4. Quantizes the output if `output_quant` is active.

---

## 4. `QuantLinear`

**Inheritance**

- Inherits from:
  - `QuantWeightBiasInputOutputLayer`
  - `Linear` (PyTorch’s standard linear layer)

**Core Functionality**

- A fully-connected layer with **integrated quantization** of inputs, outputs, weights, and optionally bias.
- Key overrides:
  - `forward`: Delegates to `forward_impl(input)` in `QuantWeightBiasInputOutputLayer`, which handles input unpack/quant before calling the linear operation.
  - `inner_forward_impl(x, quant_weight, quant_bias)`: Uses `torch.nn.functional.linear`.
  - `quant_output_scale_impl(inp, quant_input_scale, quant_weight_scale)`: For advanced scale calculation.

**Other Properties**

- `per_elem_ops`: `2 * self.in_features` (multiply + add).
- `output_channel_dim`: `0`, meaning output channels are along dimension 0.
- `out_channels`: `self.out_features`.
- `channelwise_separable`: `False` for a linear layer.

---

## 5. `QuantConv1d`

**Inheritance**

- Inherits from:
  - `QuantWeightBiasInputOutputLayer`
  - `Conv1d`

**Core Functionality**

- A 1D convolutional layer with **quantization** for weights, bias, input, and output.
- The `forward_impl` method:
  1. Unpacks and quantizes input, weight, bias.
  2. Uses `_conv_forward` or `conv1d_same_zeros_pad_stride` if `is_same_padded_strided`.

**Other Properties**

- `per_elem_ops`: `2 * kernel_size[0] * (in_channels // groups)`.
- `output_channel_dim`: `1` if transposed, otherwise `0`.
- `channelwise_separable`: `True` if `groups == in_channels` (depthwise 1D conv).

---

## 6. `QuantConv2d`

**Inheritance**

- Inherits from:
  - `QuantWeightBiasInputOutputLayer`
  - `Conv2d`

**Core Functionality**

- A 2D convolution with **quantized** weights, bias, input, and output.
- `forward_impl(...)` unpacks/quantizes and may use `_conv_forward` or `conv2d_same_zeros_pad_stride` for “same” padding.

**Other Properties**

- `per_elem_ops`: `2 * (kernel_height * kernel_width) * (in_channels // groups)`.
- `output_channel_dim`: `1` if transposed, otherwise `0`.
- `channelwise_separable`: `True` if `groups == in_channels`.

---

## 7. `QuantConv3d`

**Inheritance**

- Inherits from:
  - `QuantWeightBiasInputOutputLayer`
  - `Conv3d`

**Core Functionality**

- A 3D convolution with **quantized** weights, bias, input, and output.
- Uses `_conv_forward` or `conv3d_same_zeros_pad_stride` for “same” padding.

**Other Properties**

- `per_elem_ops`: `2 * (kernel_d * kernel_h * kernel_w) * (in_channels // groups)`.
- `output_channel_dim`: `1` if transposed, otherwise `0`.
- `channelwise_separable`: `True` if `groups == in_channels`.

---

## 8. `QuantMultiheadAttention`

**Inheritance**

- Inherits directly from `Module`.
- Uses multiple **Brevitas** components (e.g., `QuantLinear`, `QuantIdentity`) to handle quantizing Q, K, V, and out projections.

**Core Functionality**

- A **quantized** variant of multi-head attention, similar to PyTorch’s `nn.MultiheadAttention`.
- Can use a single “packed” in-projection (`in_proj`) or separate linears (`q_proj`, `k_proj`, `v_proj`).
- Incorporates quantization steps (scaling Q, transposing K, quantizing attention weights, etc.).
- The final out-projection (`out_proj`) is also quantized.

**Flow**

1. Checks shape with `mha_shape_check`.
2. In-projection (packed or separate).
3. Reshapes to multiple heads, does scaled dot-product with optional attention mask.
4. Applies softmax + optional dropout.
5. Multiplies by `v` and then out-projection.

---

## 9. `QuantReLU`

**Inheritance**

- Inherits from `QuantNonLinearActLayer`.
- Wraps `nn.ReLU`, with `passthrough_act=True`.
- Default `act_quant`: `Uint8ActPerTensorFloat`.

**Behavior**

- Quantizes the input if `input_quant` is set.
- Applies ReLU activation.
- Quantizes the output with `act_quant`.

---

## 10. `QuantSigmoid`

**Inheritance**

- Inherits from `QuantNonLinearActLayer`.
- Uses a custom `Sigmoid` class, `passthrough_act=False`.
- Default `act_quant`: `Uint8ActPerTensorFloat`.

**Behavior**

- Quantizes the input if `input_quant` is set.
- Applies the Sigmoid activation.
- Quantizes the resulting output if `act_quant` is enabled.

---

## 11. `QuantTanh`

**Inheritance**

- Inherits from `QuantNonLinearActLayer`.
- Uses a custom `Tanh` class, `passthrough_act=False`.
- Default `act_quant`: `Int8ActPerTensorFloat`.

**Behavior**

- Quantizes the input if provided.
- Applies Tanh activation.
- Quantizes the output if `act_quant` is active.

---

## 12. `QuantHardTanh`

**Inheritance**

- Inherits from `QuantNonLinearActLayer`.
- Wraps `nn.Hardtanh`, `passthrough_act=True`.
- Default `act_quant`: `Int8ActPerTensorFloatMinMaxInit`.

**Behavior**

- Quantizes input if set.
- Applies HardTanh activation.
- Quantizes the output if `act_quant` is active.

---

## 13. `QuantIdentity`

**Inheritance**

- Inherits from `QuantNonLinearActLayer`.
- No explicit activation (`act_impl=None`) with `passthrough_act=True`.
- Default `act_quant`: `Int8ActPerTensorFloat`.

**Behavior**

- Quantizes the input if `act_quant` is enabled, but does no additional transform.
- Functions as a placeholder for boundary quantization with no activation change.

---

## 14. `QuantScaleBias`

_(in the same file there is also a simple `ScaleBias`)_

**Inheritance**

- Inherits from:
  - `QuantWeightBiasInputOutputLayer`
  - `ScaleBias`

**What it does**

- Combines **scale-and-bias** logic with quantization of weight, bias, input, and output.
- The `ScaleBias` part has a learnable `weight` (scale) and optional `bias` for each feature:
  - `out = input * weight + bias`
- Quantization is handled by the `QuantWeightBiasInputOutputLayer` side.

**Other Properties**

- `per_elem_ops`: `2` (one multiply + one add).
- `output_channel_dim`: `0` (scaling each channel).
- `out_channels`: number of features (`num_features`).
- `channelwise_separable`: `True`.

---

## 15. `TruncAvgPool2d`

**Inheritance**

- Inherits from:
  - `TruncMixin`
  - `QuantLayerMixin`
  - `AvgPool2d`

**Core Functionality**

- A **quantized average pooling** layer that replaces division with a right-shift or truncation approach (`TruncMixin`).
- Accepts a `QuantTensor` for fully quantized behavior:
  1. Runs `AvgPool2d`.
  2. Multiplies the result by `kernel_size` (removes division).
  3. Applies truncation quant if enabled.

**Other Properties**

- `channelwise_separable`: `True`.
- `requires_export_handler`: `True` due to custom logic.

---

## 16. `TruncAdaptiveAvgPool2d`

**Inheritance**

- Inherits from:
  - `TruncMixin`
  - `QuantLayerMixin`
  - `AdaptiveAvgPool2d`

**Core Functionality**

- Similar to `TruncAvgPool2d`, but for **adaptive** average pooling.
- Uses output size to compute an effective kernel area, removing division via truncation.
- Accepts a `QuantTensor` to preserve scale.

**Other Properties**

- `channelwise_separable`: `True`.
- `requires_export_handler`: `True`.

---

## 17. `_BatchNormToQuantScaleBias`

**Inheritance**

- Inherits from:
  - `QuantScaleBias`
  - `ABC` (abstract base class in Python)

**Core Functionality**

- An **abstract** base for converting a PyTorch BatchNorm layer into a quantized **ScaleBias**.
- In `_load_from_state_dict`:
  1. Reads BN parameters (weight, bias, running_mean, running_var).
  2. Uses `mul_add_from_bn` to derive equivalent scale/bias.
  3. Assigns them to `QuantScaleBias` fields.
  4. Removes BN-specific entries (running stats) from the state dict.

---

## 18. `BatchNorm1dToQuantScaleBias`

**Inheritance**

- Inherits from `_BatchNormToQuantScaleBias`.

**Core Functionality**

- Specifically targets **1D** batch normalization.
- `runtime_shape=(1, -1, 1)` for reshaping [N, C, L]-like data.
- Passes BN parameters (eps, etc.) and transforms them into scale+offset for `QuantScaleBias`.

---

## 19. `BatchNorm2dToQuantScaleBias`

**Inheritance**

- Inherits from `_BatchNormToQuantScaleBias`.

**Core Functionality**

- Specifically targets **2D** batch normalization.
- `runtime_shape=(1, -1, 1, 1)` for [N, C, H, W].
- Transforms BN parameters to scale+offset in `QuantScaleBias`.

---

## 20. `QuantEltwiseAdd`

**Inheritance**

- Inherits from:
  - `QuantInputOutputLayer`
  - `Module` (PyTorch)

**Core Functionality**

- Performs **elementwise addition** of two tensors, both can be quantized.
- `forward` steps:
  1. Unpacks inputs.
  2. Applies input quant to each operand.
  3. Adds them elementwise.
  4. Applies output quant and re-packs if needed.

**Other Properties**

- `channelwise_separable`: `True`.
- In `export_mode`, delegates to an export handler.

---

## 21. `QuantCat`

**Inheritance**

- Inherits from:
  - `QuantInputOutputLayer`
  - `Module` (PyTorch)

**Core Functionality**

- **Concatenates** a list of tensors (or `QuantTensor`s) along a specified dimension with input/output quantization.
- The `forward` method:
  1. Unpacks each tensor.
  2. Quantizes each via `self.input_quant`.
  3. Concatenates them into a single `QuantTensor`.
  4. Applies `output_quant`.
  5. Returns the final packed output.

**Other Properties**

- `channelwise_separable`: `True`.

---

## 22. `TruncQuantAccumulator`

**Inheritance**

- Inherits from:
  - `TruncMixin`
  - `QuantLayerMixin`
  - `Module`

**Core Functionality**

- Applies **truncation quant** (e.g., right-shift logic) to an accumulator value.
- The `forward` method:
  1. Unpacks the `QuantTensor`.
  2. Applies `self.trunc_quant(...)`.
  3. Re-packs if needed.

**Other Properties**

- `channelwise_separable`: `True`.
- `requires_export_handler`: `True`.

---

## 23. `ClampQuantAccumulator`

**Inheritance**

- Inherits from:
  - `QuantClampMixin`
  - `QuantLayerMixin`
  - `Module`

**Core Functionality**

- Clamps the data to a specified range for an accumulator.
- The `forward` method:
  1. Unpacks the `QuantTensor`.
  2. Applies `self.clamp_quant(...)`.
  3. Re-packs if necessary.

**Other Properties**

- `channelwise_separable`: `True`.
- `requires_export_handler`: `True`.

---

## 24. `QuantUpsample`

**Inheritance**

- Inherits from:
  - `QuantLayerMixin`
  - `Upsample` (PyTorch)

**Core Functionality**

- Upsamples a `QuantTensor` (or Tensor) via PyTorch’s `interpolate`.
- For non-nearest mode, it **rounds** the interpolated values to the original scale factor:
  - Asserts the input is a `QuantTensor` with a defined scale.
  - Calls `round_ste(y_value / x.scale) * x.scale`.
- Keeps the same quantization parameters from the input after upsampling.

**Other Properties**

- `channelwise_separable`: `True`.
- `requires_export_handler`: `False`.

---

## 25. `QuantUpsamplingBilinear2d`

**Inheritance**

- Inherits from:
  - `QuantLayerMixin`
  - `UpsamplingBilinear2d`

**Core Functionality**

- Specialized 2D bilinear upsampling for a `QuantTensor`, rounding back to the input scale.
- Uses `round_ste(y_value / x.scale) * x.scale`.
- In `export_mode`, delegates to an `export_handler`.

**Other Properties**

- `channelwise_separable`: `True`.
- `requires_export_handler`: `False`.

---

## 26. `QuantUpsamplingNearest2d`

**Inheritance**

- Inherits from:
  - `QuantLayerMixin`
  - `UpsamplingNearest2d`

**Core Functionality**

- Specialized 2D nearest-neighbor upsampling for a `QuantTensor`.
- Unlike bilinear, it does not need rounding step because nearest-neighbor produces integral multiples at each pixel.

**Other Properties**

- `channelwise_separable`: `True`.
- `requires_export_handler`: `False`.

---

## 27. `QuantConvTranspose1d`

**Inheritance**

- Inherits from:
  - `QuantWeightBiasInputOutputLayer`
  - `ConvTranspose1d`

**Core Functionality**

- A **transposed 1D convolution** with quantized input, output, weight, and bias.
- `forward` optionally takes `output_size` for custom shape.
- The `inner_forward_impl` uses a specialized `_output_padding` logic and `conv_transpose1d_zeros_pad`.

**Other Properties**

- `per_elem_ops`: _Not implemented_ (raises `NotImplementedError`).
- `output_channel_dim`: `1` for transposed conv.
- `channelwise_separable`: `self.groups == self.out_channels`.

---

## 28. `QuantConvTranspose2d`

**Inheritance**

- Inherits from:
  - `QuantWeightBiasInputOutputLayer`
  - `ConvTranspose2d`

**Core Functionality**

- A **transposed 2D convolution** with quantization for weight, bias, input, and output.
- Similar to `QuantConvTranspose1d` but for 2D.
- `inner_forward_impl` calls `conv_transpose2d_zeros_pad` using a computed `output_padding`.

**Other Properties**

- `per_elem_ops`: _Not implemented_.
- `output_channel_dim`: `1`.
- `channelwise_separable`: `self.groups == self.out_channels`.

---

## 29. `QuantConvTranspose3d`

**Inheritance**

- Inherits from:
  - `QuantWeightBiasInputOutputLayer`
  - `ConvTranspose3d`

**Core Functionality**

- Extends **transposed convolution** to 3D with full quantization.
- Uses `conv_transpose3d_zeros_pad`.
- Supports an optional `output_size` argument.

**Other Properties**

- `per_elem_ops`: _Not implemented_.
- `output_channel_dim`: `1`.
- `channelwise_separable`: `self.groups == self.out_channels`.

---

## 30. `QuantEmbedding`

**Inheritance**

- Inherits from:
  - `QuantWeightMixin`
  - `Embedding` (PyTorch)

**Core Functionality**

- A **quantized embedding** layer.
- The weight is quantized via `QuantWeightMixin`.
- The output is a `QuantTensor` if `return_quant_tensor` is `True` **and** weight quant is enabled; otherwise a standard `Tensor`.

**Notable Steps**

1. Applies `self.quant_weight()` to get the quantized embedding weights.
2. Uses `F.embedding` to retrieve embeddings for the input indices.

**Other Properties**

- `output_channel_dim`: `0`.
- `out_channels`: `num_embeddings`.

---

## 31. `QuantRNN`

**Inheritance**

- Inherits from `QuantRecurrentStackBase` (a multi-layer utility).

**Core Functionality**

- A **quantized RNN** (with either ReLU or Tanh nonlinearity) potentially in a multi-layer, bidirectional setup.
- Each layer is an instance of `_QuantRNNLayer`, which uses:
  - `GateParams` for quantizing weights/bias.
  - `QuantIdentity` for input/output/gate accumulator quantization.
- `forward` iterates over each layer (and direction if bidirectional), concatenating outputs.

**Key Config**

- `weight_quant`, `bias_quant`, `io_quant`, `gate_acc_quant`, etc. control quantization parameters.
- `shared_input_hidden_weights` optionally shares weight parameters across layers.

---

## 32. `QuantLSTM`

**Inheritance**

- Inherits from `QuantRecurrentStackBase`.

**Core Functionality**

- A **quantized LSTM** with possible multi-layer, bidirectional arrangement.
- Each layer is `_QuantLSTMLayer`, internally using `_QuantLSTMCell` for gating logic:
  - Input, Forget, Cell, Output gates each have quantization for accumulator and activation.
  - Potential sharing of quantizers across gates (`shared_intra_layer_gate_acc_quant`, etc.)
- The final `forward` returns both the output states and hidden/cell states, with optional concatenation of the latter.

**Key Config**

- `coupled_input_forget_gates` (CIFG) merges input/forget gating logic if `True`.
- `shared_cell_state_quant` can share quantization for cell states.
- Additional parameters for controlling gate-level quantization.

---

## 33. `HadamardClassifier`

**Inheritance**

- Inherits from:
  - `QuantLayerMixin`
  - `Module` (PyTorch)

**Core Functionality**

- Implements a **Hadamard transform**-based classifier that compresses input features into a smaller dimension, multiplying by a Hadamard matrix.
- `scale` is learnable or fixed, acts as a final scaling factor.
- Expects the input to be an `IntQuantTensor` if we want to preserve quantization info.
- In the forward pass:
  1. Normalizes input by its Frobenius norm.
  2. Applies a linear transform with a partial Hadamard matrix slice (`proj`).
  3. Scales by `-self.scale`.

**Other Properties**

- `max_output_bit_width(input_bit_width)`: calculates an upper bound on the needed bit-width by considering the input range times `in_channels`.
- `proj` is stored as a buffer but not saved in `state_dict` (it’s generated from `scipy.linalg.hadamard`).

---

## 34. `EqualizedModule`

**Inheritance**

- Inherits from `Module`.

**Core Functionality**

- A **wrapper** that applies a scaling operation (`scale_module`) to an input, then passes it to another layer (`self.layer`).
- Typically used for **activation equalization**.
- The `forward(*args, **kwargs)` intercepts whichever input is recognized as the “main” (e.g., `input`, `inp`, `query`, etc.), applies `self.scale` to it, and then calls the underlying layer with the scaled input.

---
