# Copyright 2025 ETH Zurich.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Victor Jung <jungvi@iis.ee.ethz.ch>


### Torch Imports ###
import torch


class InnerForwardImplWrapperWBIOL(torch.nn.Module):
    def __init__(self, inner_forward_impl):
        super().__init__()
        self.inner_forward_impl = inner_forward_impl

    def forward(self, quant_input, quant_weight, quant_bias):
        return self.inner_forward_impl(quant_input, quant_weight, quant_bias)
