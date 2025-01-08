# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Philip Wiese <wiesep@iis.ee.ethz.ch>

all: help

format:
	@echo "Formatting code"
	@black */*.py

help:
	@echo "Usage: make [format]"
	@echo "format: Format the code"

.PHONY: format all