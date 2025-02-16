# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

# Brevitexporter

A Python library for exporting Brevitas quantized neural networks.

## Installation

### Requirements

- Python 3.11 or higher
- PyTorch 2.1.2 or higher  
- Brevitas 0.11.0 or higher

### Setup Environment

First, create and activate a new conda environment:

```bash
mamba create -n brevitas_env python=3.11
mamba activate brevitas_env
```

### Install Dependencies

Install PyTorch and its related packages:

```bash
mamba install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -c pytorch
```

### Install the Package

Clone the repository and install in development mode:

```bash
cd brevitexporter
pip install -e .
```

## Running Tests

### Using Make (Recommended)

The project includes a Makefile with several testing commands:

```bash
# Run all tests with verbose output
make test

# Run only neural network test  
make test-nn

# Run only multi-head attention test
make test-mha 

# Run only CNN test
make test-cnn

# Run a specific test file
make test-single TEST=test_simple_nn.py

# Show all available make commands
make help
```

### Using pytest directly

You can also run tests using pytest commands:

```bash
# Run all tests
python -m pytest src/brevitexporter/tests -v -s

# Run a specific test file 
python -m pytest src/brevitexporter/tests/test_simple_nn.py -v -s
```

## Project Structure

```
brevitexporter/
├── Makefile
├── pyproject.toml
├── conftest.py
└── src/
    └── brevitexporter/
        ├── custom_forwards/
        │   ├── activations.py
        │   ├── linear.py
        │   └── multiheadattention.py
        ├── injects/
        │   ├── base.py
        │   ├── executor.py
        │   └── transformations.py
        ├── tests/
        │   ├── test_simple_mha.py
        │   ├── test_simple_nn.py
        │   └── test_simple_cnn.py
        ├── custom_tracer.py
        └── export_brevitas.py
```

### Key Components

- **custom_forwards/**: Contains unrolled forward implementations for:
  - Linear layers (QuantLinear, QuantConv2d)
  - Activation functions (QuantReLU, QuantSigmoid) 
  - Multi-head attention (QuantMultiheadAttention)

- **injects/**: Contains the transformation infrastructure:
  - Base transformation class and executor
  - Module-specific transformations  
  - Validation and verification logic

- **tests/**: Example tests demonstrating exporter usage with:
  - Simple neural networks (linear + activations)
  - Multi-head attention models
  - Convolutional neural networks

- **custom_tracer.py**: Specialized FX tracer for Brevitas modules
- **export_brevitas.py**: Main export API

## Usage 

### Main Function: exportBrevitas

The main function of this library is `exportBrevitas`, which exports a Brevitas-based model to an FX GraphModule with unrolled quantization steps.

```python
from brevitexporter.export_brevitas import exportBrevitas

# Initialize your Brevitas model
model = YourBrevitasModel().eval()

# Create an input with the correct shape
input = torch.randn(1, input_channels, height, width)

# Export the model (with debug information)
fx_model = exportBrevitas(model, input, debug=True)
```

Arguments:
- `model`: The Brevitas-based model to export
- `example_input`: A representative input tensor for shape tracing  
- `debug`: If True, prints transformation progress (default: False)

### Example Usage

A simple example with a quantized convolutional model:

```python
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias 
from brevitexporter.export_brevitas import exportBrevitas

class SimpleQuantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.conv = qnn.QuantConv2d(
            in_channels=3,
            out_channels=16, 
            kernel_size=3,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.conv(x)
        return x

# Export the model
model = SimpleQuantModel().eval()
dummy_input = torch.randn(1, 3, 32, 32)
fx_model = exportBrevitas(model, dummy_input, debug=True)
```

For more examples, see the test files in `src/brevitexporter/tests/`.

When `debug=True`, you'll see progress output during the transformation process:

```
✓ Linear transformation successful
✓ Activation transformation successful
All transformations completed successfully!
```

## Advanced Features

The library provides detailed control over the tracing and transformation process:

1. Custom Tracer Configuration
- Explicit leaf/non-leaf module designation
- Fine-grained control over module traversal
- Debug output for transformation steps

2. Transformation Validation
- Automatic output validation after each transformation
- Configurable numerical tolerance
- Detailed error messages on mismatch

3. Extensible Architecture
- Base classes for custom transformations
- Modular injection system
- Support for new quantized module types

## Contributing

Contributions are welcome! Please ensure your code follows our style guidelines:

1. Include the copyright header in all source files
2. Use Google-style docstrings
3. Include type hints
4. Add tests for new functionality
5. Follow PEP 8 style guidelines

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.