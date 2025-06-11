import json
from typing import Any, Dict

import torch
from torch import nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.quantizers import HfQuantizer, register_quantization_config, register_quantizer
from transformers.utils.quantization_config import QuantizationConfigMixin
from optimized_module.ops.quantize.quant_linear_work import QuantizedLinear

@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfigMixin):
    def __init__(
        self, 
        bits=4, 
        group_size=128, 
        enable_outlier_preprocess=False, 
        linear_weights_not_to_quantize=None, 
        **kwargs
    ):
        self.quant_method = "my_quant"
        self.bits = bits
        self.group_size = group_size
        if linear_weights_not_to_quantize is None:
            linear_weights_not_to_quantize = ["lm_head.weight"]
        self.linear_weights_not_to_quantize = linear_weights_not_to_quantize
        self.enable_outlier_preprocess = enable_outlier_preprocess

    def to_dict(self) -> Dict[str, Any]:
        output = {
            "quant_method": self.quant_method,
            "bits": self.bits,
            "group_size": self.group_size,
            "enable_outlier_preprocess": self.enable_outlier_preprocess,
            "linear_weights_not_to_quantize": self.linear_weights_not_to_quantize
        }
        return output

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> Dict[str, Any]:
        config_dict = self.to_dict()

        default_config_dict = MyQuantConfig().to_dict()

        serializable_config_dict = {}

        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


def replace_with_my_quant_linear(
    model,
    quantization_config=None,
    linear_weights_not_to_quantize=None,
    current_key_name=None,
    has_been_replaced=False,
):
    if linear_weights_not_to_quantize is None:
        linear_weights_not_to_quantize = []

    from accelerate import init_empty_weights

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear):
            # Check if the current key is not in the `linear_weights_not_to_quantize`
            if ".".join(current_key_name) + ".weight" not in linear_weights_not_to_quantize:
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features

                    model._modules[name] = QuantizedLinear(
                        in_features,
                        out_features,
                        bias=module.bias is not None,
                        bits=quantization_config.bits,
                        group_size=quantization_config.group_size,
                        enable_outlier=quantization_config.enable_outlier_preprocess,
                    )
                    has_been_replaced = True

                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    # model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_my_quant_linear(
                module,
                quantization_config=quantization_config,
                linear_weights_not_to_quantize=linear_weights_not_to_quantize,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


@register_quantizer("my_quant")
class MyQuantQuantizer(HfQuantizer):
    def __init__(self, quantization_config: MyQuantConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config
        self.scale_map = {}
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = kwargs.get("torch_dtype", torch.float32)

    def _process_model_before_weight_loading(self, model, **kwargs):
        replace_with_my_quant_linear(
            model, 
            self.quantization_config, 
            self.quantization_config.linear_weights_not_to_quantize
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model, **kwargs):
        return model

    def is_serializable(self, **kwargs) -> bool:
        return True

    def is_trainable(self) -> bool:
        return True
    
    def is_qat_trainable(self) -> bool:
        return True