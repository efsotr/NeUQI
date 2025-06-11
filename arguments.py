import os
from dataclasses import dataclass, field
from enum import Enum

import torch
from transformers import AutoConfig, HfArgumentParser


METHODS = ["RTN", "LDLQ", "GPTQ"]

tune_methods = ["best_zp", "grid_s_zp", "grid_s_best_zp", "grid2_s_best_zp"]
param_init_methods = [None, "minmax_plus", "clip_wo_shift_zp"] + tune_methods
PARAM_INIT_METHODS = Enum('DynamicEnum', {str(item): item for item in param_init_methods})

@dataclass
class QuantArgs:
    nbits: int = field(default=16, metadata={"help": "quantization bits", "choices": [1, 2, 3, 4, 5, 6, 7, 8, 16]})
    group_size: int = field(default=-1)
    method: str = field(default="RTN", metadata={"choices": METHODS})
    enable_H_reorder: bool = field(default=False) # LDLQ
    round_method: str = field(default="round", metadata={"choices": ["round", "random"]})

    fix_iters: int = field(default=0)
    scale_clip_beta: float = field(default=1)

    enable_magr: str = field(default=False)
    enable_magr_only: str = field(default=False)
    magr_alpha: float = field(default=None)

    # aux_loss = \sum_{i=1}^n H_{ii}(Q(x_i)-x_i)^2
    enable_H_diag_weight: bool = field(default=False)
    param_init_method_diag: str = field(default=None, metadata={"choices": param_init_methods}) # grid s, zp is LeanQuant
    LDLQ_fix_iters: int = field(default=0)
    grid_tot_steps: int = field(default=256)
    grid_start_steps: int = field(default=1)
    grid_mid_tot_steps: int = field(default=1)
    enable_grid_low_precision_step: bool = field(default=False)

    def __post_init__(self):
        if self.param_init_method_diag is not None and ("grid" in self.param_init_method_diag or self.param_init_method_diag == "minmax_plus"):
            self.scale_clip_beta = 1

    def to_config_dict(self):
        return {
            "quant_method": "my_quant",
            "bits": self.nbits,
            "group_size": self.group_size,
        }

    def short_name(self):
        assert self.method in METHODS
        assert self.param_init_method_diag in param_init_methods
        if self.enable_magr_only:
            assert self.enable_magr
        name = ""
        if self.enable_magr:
            name += ".MagR"
            if self.enable_magr_only:
                name += "_only"
                self.magr_alpha = 1e-3
                if self.group_size != -1:
                    self.magr_alpha = 1e-4
                self.param_init_method_diag = "clip_wo_shift_zp"
                if self.nbits == 2:
                    self.scale_clip_beta = 0.8
                elif self.nbits == 3:
                    self.scale_clip_beta = 0.9
                else:
                    self.scale_clip_beta = 1
                if self.group_size == 128:
                    self.scale_clip_beta = 0.95
                elif self.group_size == 64:
                    self.scale_clip_beta = 1
            if self.magr_alpha is not None:
                name += f".a{self.magr_alpha:.1e}"
        name += f".{self.method}.{self.nbits}bit"
        if self.group_size > -1:
            name += f".g{self.group_size}"
        if self.scale_clip_beta < 1:
            name += f'.clip{self.scale_clip_beta}'
        if self.round_method == "random":
            name += f'.{self.round_method}'
        if self.enable_H_reorder:
            name += ".Hsort"
        if self.LDLQ_fix_iters > 0 and self.method == "LDLQ":
            name += f".LDLQfix{self.LDLQ_fix_iters}iters"
        if self.param_init_method_diag:
            name += "." + self.param_init_method_diag
            if "grid" in self.param_init_method_diag:
                if self.param_init_method_diag == "grid2_s_best_zp":
                    name += f".grid{self.grid_mid_tot_steps}-{self.grid_tot_steps}"
                    assert not self.enable_grid_low_precision_step
                else:
                    name += f".grid{self.grid_tot_steps}"
                if self.grid_start_steps > 1:
                    name += f".gridstart{self.grid_start_steps}"
                if self.enable_grid_low_precision_step:
                    name += ".low"
            if self.param_init_method_diag in tune_methods:
                assert self.enable_H_diag_weight
        if self.fix_iters > 0:
            name += f".fix{self.fix_iters}iters"

        return name

abbr_mapping = {
    "Llama-2-7b-hf": "llama-2-7b",
    "Llama-2-13b-hf": "llama-2-13b",
    "Llama-2-70b-hf": "llama-2-70b",
    "Meta-Llama-3-8B": "llama-3-8b",
    "Meta-Llama-3-70B": "llama-3-70b",
    "Qwen2.5-1.5B": "qwen-2.5-1.5b",
    "Qwen2.5-3B": "qwen-2.5-3b",
    "Qwen2.5-7B": "qwen-2.5-7b",
    "Qwen2.5-14B": "qwen-2.5-14b",
    "Qwen2.5-32B": "qwen-2.5-32b",
    "Qwen2.5-72B": "qwen-2.5-72b",
}

def abbr_mapping_fn(model_path):
    model_name = model_path.split(os.sep)[-1]
    return abbr_mapping.get(model_name, model_name)

@dataclass
class Args:
    model_path: str = field(metadata={"help": "model path"})
    dataset: str = field(metadata={"help": "datasets", "choices": ["c4", "pajama", "wikitext2", "c4_new", "ptb_new"]})
    test_dataset: str = field(metadata={"help": "Test Datasets"})
    result_dir: str = field(metadata={"help": "Save path for result"})
    log_dir: str = field()
    save_dir: str = field()
    H_dir: str = field(default=None)
    tmp_dir: str = field(default="tmp")
    nsamples: int = field(default=128, metadata={"help": "Num of calibration data samples."})
    seqlen: int = field(default=2048, metadata={"help": "calibration data context length"})
    dtype: str = field(default="auto", metadata={"help": "Model dtype", "choices": ["auto", "float16", "float32", "bfloat16"]})
    force_recalc_H: bool = field(default=False)
    batch_size: int = field(default=1)
    torch_dtype: torch.dtype = field(init=False)
    result_path: str = field(init=False)
    save_path: str = field(init=False)
    H_path: str = field(init=False)
    tmp_path: str = field(init=False)


    def __post_init__(self):
        if self.dtype == "auto":
            config = AutoConfig.from_pretrained(self.model_path)
            self.torch_dtype = config.torch_dtype
        else:
            self.torch_dtype = getattr(torch, self.dtype)

        self.test_dataset = set(self.test_dataset.split(","))
        assert {"c4", "wikitext2", "ptb", "pajama", "c4_new", "ptb_new"}.issuperset(self.test_dataset)
        self.test_dataset = sorted(self.test_dataset)

def get_short_name(args: Args, quant_args: QuantArgs):
    abbr_model_name = abbr_mapping_fn(args.model_path)
    if quant_args.nbits == 16:
        short_name = f"{abbr_model_name}.{str(args.torch_dtype).split('.')[1]}"
    else:
        short_name = f"{abbr_model_name}.{args.dataset}.n{args.nsamples}.L{args.seqlen}" + quant_args.short_name() + ".seq"
    return short_name

if __name__ == "__main__":
    args, quant_args = HfArgumentParser((Args, QuantArgs)).parse_args_into_dataclasses()
    print(get_short_name(args, quant_args))
