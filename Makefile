# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Philip Wiese <wiesep@iis.ee.ethz.ch>

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Available Targets:"
	@echo " - 01_onnx: Run example one: Simple CNN without Training"
	@echo " - 02_cnn: Run example two: ResNet18 on ImageNet-1k"
	@echo " - format: Format the code"
	@echo " - clean : Remove all generated files"
	@echo " - help: Show this help message"

01_onnx:
	@echo "Running 01_onnx..."
	@python 01_onnx/main.py 2>&1 |& tee 01_log.txt
	@netron 01_quant_model_qonnx.onnx

02_cnn:
	@echo "Running 02_cnn..."
	@python 02_cnn/main.py 2>&1 |& tee 02_log.txt
	@netron 02_quant_model_qonnx.onnx

format:
	@echo "Formatting code"
	@black */*.py

clean:
	@echo "Cleaning up..."
	@rm -rf *.onnx *.pth *.log *.pt *_log.txt

.PHONY: format all 01_onnx 02_cnn help clean