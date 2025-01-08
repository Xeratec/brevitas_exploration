# Brevitas Exploration

This repo contains various examples to better understand the Brevitas library.

## Installation

First, setup a new mamba/conda environment and install the required packages:
```sh
mamba create -n brevitas
mamba activate brevitas
mamba install python=3.11
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

I suggest to install some additional useful packages:
```sh
pip -r requirements.txt
```

Next install Brevitas from the latest release on PyPI:
```sh
pip install brevitas
```

Alternatively, you can install a local editable version of Brevitas with all additional dependencies:
```sh
git clone https://github.com/Xilinx/brevitas.git
cd brevitas
pip install -e ".[notebook,dev,docs,export,hadamard,test,tts,stt,llm,vision,finn_integration,ort_integration]" --config-settings editable_mode=compat
```

## Examples

You can directly use the Makefile to run the examples. The Makefile will automatically create a new log file for each example.

```sh
# List all available examples
make help

# Example
make 01_onnx
make 02_cnn
```

To stop `netron` which is automatically started after each example and used to visualize the quantized model, press `Ctrl+C` in the terminal

### 1. Simple CNN without Training
To run the example and save the output to a log file, use the following command:

```sh
python 01_onnx/main.py 2>&1 |& tee 01_log.txt
```

Next you can visualize the quantized model with the following command:

```sh
# Visualize the ONNX model in QONNX format
netron 01_quant_model_qonnx.onnx

# Visualize the ONNX model in QCDQ format
netron 01_quant_model_qcdq.onnx
```

### 2. ResNet18 on ImageNet-1k
To run the example and save the output to a log file, use the following command:

```sh
python 02_cnn/main.py 2>&1 |& tee 02_log.txt
```

You should get around 69.8% top-1 accuracy on ImageNet-1k for both the full-precision and 67.8% quantized models.
Next you can visualize the quantized model with the following command:

```sh
# Visualize the ONNX model in QONNX format
netron 02_quant_model_qonnx.onnx

# Visualize the ONNX model in QCDQ format
netron 02_quant_model_qcdq.onnx
```

