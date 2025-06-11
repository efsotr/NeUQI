import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers.trainer import TrainingArguments

from optimized_module.ops.quantize.quant_linear_work import str2enum_map


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class ModelArguments:
    model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    quant_model_id: str = field(default=None)
    tokenizer_id: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer."}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Override the dtype of model.",
                  "choices" : ["float16", "bfloat16", "float32"]}
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "The attention implementation.",
                  "choices": ["eager", "flash_attention_2"]}
    )
    model_cpu_offload: str = field(default=False)
    quant_model_cpu_offload: str = field(default=False)

    def __post_init__(self):
        if self.tokenizer_id is None:
            self.tokenizer_id = self.model_id
        self.torch_dtype = getattr(torch, self.torch_dtype)

@dataclass
class DataTrainingArguments:

    dataset: str = field(default=None)
    nsamples: int = field(default=128)
    seqlen: int = field(default=2048)
    test_dataset: str = field(default=None)

@dataclass
class MaxTrainingArguments(TrainingArguments):

    disable_compile: bool = field(default=False)
    disable_train_scale: bool = field(default=False)
    disable_train_zero: bool = field(default=False)
    train_method: str = field(
        default="default",
        metadata={"choices": ["default", "ste", "pv_tuning", "pv_tuning_full"]}
    )

    learning_rate_code: float = field(default=None)
    adam_beta1_code: float = field(default=None)
    adam_beta2_code: float = field(default=None)

    code_trust_ratio: float = field(default=100)
    code_update_ratio: float = field(default=1.0)

    check_stage: str = field(
        default="no_ck",
        metadata={"choices": ["no_ck", "ck_data", "ck_ref", "ck_run"]}
    )

    training_type: str = field(
        default="quant_distill",
        metadata={"help": "training type", "choices": ["quant_distill"]}
    )

    def __post_init__(self):
        super().__post_init__()

        if self.gradient_checkpointing_kwargs is None:
            self.gradient_checkpointing_kwargs = {}
        self.gradient_checkpointing_kwargs["use_reentrant"] = True

        if self.disable_train_scale and self.disable_train_zero:
            assert self.train_method != "default"

        self.train_method = str2enum_map[self.train_method]

        if self.learning_rate_code is None:
            self.learning_rate_code = self.learning_rate
        if self.adam_beta1_code is None:
            self.adam_beta1_code = self.adam_beta1
        if self.adam_beta2_code is None:
            self.adam_beta2_code = self.adam_beta2


