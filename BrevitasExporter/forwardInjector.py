# Copyright 2025 ETH Zurich.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Victor Jung <jungvi@iis.ee.ethz.ch>

# JUNGVI: The idea here is to have replacements for the forward function of the quantizer layer modules from Brevitas
# These replacement functions expose the quantizers and filter out the usless logic unused after calibration.
# Of course each new forward function has to be tested to make sure they are functionally equivalent to the original


### Brevitas Imports ###
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer

### Local Imports ###
from moduleWrapper import InnerForwardImplWrapperWBIOL


def quantWBIOL_forward(self, inp):
    quant_input = self.input_quant(inp)
    quant_weight = self.weight_quant(self.weight)
    quant_bias = self.bias_quant(self.bias, quant_input, quant_weight)

    output = self.wrapped_inner_forward_impl(quant_input, quant_weight, quant_bias)

    quant_output = self.output_quant(output)
    return quant_output


def quantWBIOL_injector(module: QuantWeightBiasInputOutputLayer):
    assert isinstance(
        module, QuantWeightBiasInputOutputLayer
    ), f"{type(module)} is not an instance of QuantWeightBiasInputOutputLayer!"

    module.wrapped_inner_forward_impl = InnerForwardImplWrapperWBIOL(module.inner_forward_impl)
    module.forward = quantWBIOL_forward.__get__(module)
    return module
