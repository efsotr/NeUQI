import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from optimized_module.ops.quantize.quant_linear_work import QuantizedLinear
from typing import Union

def has_quantized_linear(module: nn.Module) -> bool:
    for sub_module in module.children():
        if isinstance(sub_module, QuantizedLinear):
            return True
        if not isinstance(sub_module, FSDP) and has_quantized_linear(sub_module):
            return True
    return False

def pv_tuning_update(model: Union[FSDP, nn.Module], code_update_ratio: float, code_trust_ratio: float):
    """
    Update the model parameters using the PV-tuning method.
    """
    if isinstance(model, FSDP):
        for module in model.modules():
            if isinstance(module, FSDP) and has_quantized_linear(module):
                with FSDP.summon_full_params(module, recurse=False):
                    for sub_module in module.modules():
                        if isinstance(sub_module, QuantizedLinear):
                            sub_module.pv_tuning_update(code_update_ratio, code_trust_ratio)
    else:
        for sub_module in model.modules():
            if isinstance(sub_module, QuantizedLinear):
                sub_module.pv_tuning_update(code_update_ratio, code_trust_ratio)
