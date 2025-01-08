# Brevitas Exploration

This repo contains various examples to better understand the Brevitas library.

## Installation

First, setup a new mamba/conda environment and install the required packages:
```sh
mamba create -n brevitas
mamba activate brevitas
mamba install python=3.11
mamba install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
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

