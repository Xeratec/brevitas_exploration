# Copyright 2025 ETH Zurich.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Victor Jung <jungvi@iis.ee.ethz.ch>

### Torch Imports ###
from torch.nn import Module

### Brevitas Imports ###
from brevitas.fx import Tracer
from brevitas.fx.brevitas_tracer import _is_brevitas_leaf_module, _symbolic_trace

### Local Imports ###
from moduleWrapper import InnerForwardImplWrapperWBIOL


class CustomBrevitasSymbolicTracer(Tracer):
    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        if m.__module__.startswith("brevitas.nn.quant_conv") or m.__module__.startswith("brevitas.nn.quant_linear"):
            return False
        if isinstance(m, InnerForwardImplWrapperWBIOL):
            return True
        return _is_brevitas_leaf_module(m, module_qualified_name)


def custom_brevitas_symbolic_trace(root, concrete_args=None):
    return _symbolic_trace(CustomBrevitasSymbolicTracer(), root, concrete_args)
