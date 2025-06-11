# SPDX-License-Identifier: Apache-2.0

import enum
from enum import Enum
from functools import partial
from fractions import Fraction
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter,
                                           ModelWeightParameter)
from vllm.model_executor.layers.quantization import register_quantization_config
from optimized_module.ops.quantize.quant_linear import quantized_linear_fwd

@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        enable_outlier_preprocess: bool,
        lm_head_quantized: bool = False,
    ) -> None:
        super().__init__()

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.enable_outlier_preprocess = enable_outlier_preprocess
        self.lm_head_quantized = lm_head_quantized
        self.pack_factor = Fraction(8, self.weight_bits)
        # if self.enable_outlier_preprocess:
        #     raise NotImplementedError
        if self.weight_bits not in [1, 2, 3, 4, 5, 6, 7, 8]:
            raise ValueError(
                "Currently, only 1/2/3/4/5/6/7/8-bit weight quantization is "
                f"supported for MyQuant, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"MyQuantConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"enable_outlier_preprocess={self.enable_outlier_preprocess}, "
                f"lm_head_quantized={self.lm_head_quantized}), ")

    @classmethod
    def get_name(cls) -> str:
        return "my_quant"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MyQuantConfig":

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        enable_outlier_preprocess = cls.get_from_keys_or(config, ["enable_outlier_preprocess"], default=False)
        # lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
        #                                          default=False)
        return cls(weight_bits, group_size, enable_outlier_preprocess)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["MyQuantLinearMethod"]:
        if isinstance(layer, LinearBase):
            return MyQuantLinearMethod(self)
        return None


class MyQuantLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: MyQuantConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.
        weight_loader = extra_weight_attrs.get("weight_loader")
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        if input_size_per_partition % group_size != 0 and group_size % input_size_per_partition != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        output_size_per_partition = sum(output_partition_sizes)
        # if (output_size_per_partition % self.quant_config.pack_factor.numerator
        #         != 0):
        #     raise ValueError(
        #         "The output size is not aligned with the quantized "
        #         "weight shape. This can be caused by too large "
        #         "tensor parallel size.")

        scale_and_zero_size = max(input_size_per_partition // group_size, 1)

        qweight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.uint8,
            ),
            output_dim=0,
            input_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        scale = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                scale_and_zero_size,
                dtype=params_dtype,
            ),
            output_dim=0,
            input_dim=1,
            packed_dim=1,
            packed_factor=group_size,
            weight_loader=weight_loader)
        
        zero = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                scale_and_zero_size,
                dtype=params_dtype,
            ),
            output_dim=0,
            input_dim=1,
            packed_dim=1,
            packed_factor=group_size,
            weight_loader=weight_loader)

        def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
            shard_size = max(shard_size // self.packed_factor, 1)
            shard_offset = shard_offset // self.packed_factor
            return shard_size, shard_offset
        
        scale.adjust_shard_indexes_for_packing = partial(adjust_shard_indexes_for_packing, scale)
        zero.adjust_shard_indexes_for_packing = partial(adjust_shard_indexes_for_packing, zero)

        def sparse_weight_loader(param, loaded_weight, **kwargs):
            loaded_weight = loaded_weight.to_dense()
            return weight_loader(param, loaded_weight, **kwargs)
        
        if self.quant_config.enable_outlier_preprocess:
            outlier_weight = ModelWeightParameter(
                data=torch.empty(
                    output_partition_sizes,
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                output_dim=0,
                input_dim=1,
                weight_loader=sparse_weight_loader,
            )
        else:
            outlier_weight = None

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("zero", zero)
        layer.register_parameter("scale", scale)
        layer.register_parameter("outlier_weight", outlier_weight)
        layer.group_size = self.quant_config.group_size
        layer.weight_bits = self.quant_config.weight_bits


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # for torch.compile
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.zero = Parameter(layer.zero.data, requires_grad=False)
        layer.scale = Parameter(layer.scale.data, requires_grad=False)
        if layer.outlier_weight is not None:
            layer.outlier_weight = Parameter(layer.outlier_weight.data.to_sparse_coo(), requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = quantized_linear_fwd(x, layer.qweight, layer.scale, layer.zero, bias, layer.weight_bits, layer.outlier_weight, None)
        return output
