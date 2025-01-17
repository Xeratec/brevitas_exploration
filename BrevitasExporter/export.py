# Copyright 2025 ETH Zurich.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# 
# Victor Jung <jungvi@iis.ee.ethz.ch>

### Torch Imports ###
from sys import modules
import torch
from torch.fx.graph_module import GraphModule

### Brevitas Imports ###
from brevitas.fx import brevitas_symbolic_trace
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer

### Local Imports ###
from forwardInjector import quantWBIOL_injector
from tracer import custom_brevitas_symbolic_trace

def exportBrevitasCalibratedModel(model: torch.nn.Module) -> GraphModule :

    # Trace the model with Brevitas modules as leaf nodes
    fx_model = brevitas_symbolic_trace(model)

    # Inject export-friendly forward function
    for node in fx_model.graph.nodes:
        if node.op == 'call_module':
            target_module = getattr(fx_model, node.target)
            if isinstance(target_module, QuantWeightBiasInputOutputLayer):
                target_module = quantWBIOL_injector(target_module)

    # Dissolve brevitas layers that contain quantizers to expose them
    fx_model = custom_brevitas_symbolic_trace(fx_model)

    # At this stage the fx graph should be composed of call_modules and get_attrs only
    # get_attrs should only point at nn.Parameters (constant tensors)
    return fx_model


