# coding=utf-8

import gc
import logging
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from optimized_module import enable_nonsac
import optimized_module.inject.inject_quant # noqa: F401
from optimized_module.inject.inject_causal_forward import inject_causal_forward
from optimized_module.ops.quantize.quant_linear_work import QuantizedLinear, TrainingMethod

from . import get_model_args, get_training_args
from .arguments_init import set_tokenizer
from .modeling_quant_distill import combine


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def load_tokenizer():
    model_args = get_model_args()

    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
        "add_eos_token": False,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_id, **tokenizer_kwargs)

    set_tokenizer(tokenizer)
    return tokenizer

def load_model():
    model_args = get_model_args()
    training_args = get_training_args()

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id,
        torch_dtype=model_args.torch_dtype,
        use_cache=False,
        attn_implementation=model_args.attn_implementation,
    )
    del model.lm_head
    inject_causal_forward(model)

    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_args.quant_model_id,
        torch_dtype=model_args.torch_dtype,
        use_cache=False,
        attn_implementation=model_args.attn_implementation,
    )

    model = FSDP(
        model,
        cpu_offload=CPUOffload(offload_params=model_args.model_cpu_offload),
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=(type(model.model.layers[0]),)),
        use_orig_params=True
    )

    if not model_args.model_cpu_offload:
        model.cuda()

    if not model_args.quant_model_cpu_offload:
        quantized_model.cuda()

    model.eval()
    model.requires_grad_(False)
    quantized_model.get_input_embeddings().requires_grad_(False)
    quantized_model.get_output_embeddings().requires_grad_(False)
    if training_args.gradient_checkpointing:
        quantized_model.enable_input_require_grads()

    if training_args.disable_train_scale:
        for name, parameter in quantized_model.named_parameters():
            if name.endswith(".scale"):
                parameter.requires_grad_(False)

    if training_args.disable_train_zero:
        for name, parameter in quantized_model.named_parameters():
            if name.endswith(".zero"):
                parameter.requires_grad_(False)

    if training_args.train_method != TrainingMethod.QUANTIZE_PARAM_ONLY:
        enum = training_args.train_method
        for name, module in quantized_model.named_modules():
            if isinstance(module, QuantizedLinear):
                module.update_train_method(enum)
                module.dequantized()

    inject_causal_forward(quantized_model)

    enable_nonsac()

    model = torch.compile(model, dynamic=True)
    combine(model, quantized_model)
    quantized_model.get_output_embeddings().to("cpu")

    torch.cuda.empty_cache()
    gc.collect()
    return quantized_model
