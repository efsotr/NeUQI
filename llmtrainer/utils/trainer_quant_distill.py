import logging
import math
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.distributed.fsdp.api import MixedPrecision
from transformers.trainer_utils import EvalLoopOutput

from optimized_module.patch_compile import patch_compile_state
from optimized_module.ops.quantize.pv_tuning_update import pv_tuning_update
from optimized_module.ops.quantize.quant_linear_work import TrainingMethod
from optimized_module.ops.quantize.utils_quantize_work import pack_quantize

from .arguments_init import get_model_args
from .trainer_base import BaseTrainer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Metric:

    def __call__(self, eval_out: EvalLoopOutput):
        losses = eval_out.predictions
        return {"ppl": math.exp(losses.mean())}


class Trainer(BaseTrainer):

    step: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.accelerator.native_amp = False
        if self.args.train_method != TrainingMethod.QUANTIZE_PARAM_ONLY:
            if self.is_fsdp_enabled:
                torch_dtype = get_model_args().torch_dtype
                self.model._apply(lambda t: t.float() if t.is_floating_point() else t)
                self.model.get_input_embeddings().to(torch_dtype)
                self.model.get_output_embeddings().to(torch_dtype)
                self.accelerator.state.fsdp_plugin.mixed_precision_policy = MixedPrecision(
                    param_dtype=torch_dtype,
                    reduce_dtype=torch_dtype,
                )

            if self.accelerator.state.fsdp_plugin is not None:
                self.accelerator.state.fsdp_plugin.ignored_modules = [self.model.get_output_embeddings()]

            logger.info(f"fsdp_plugin.mixed_precision_policy {self.accelerator.state.fsdp_plugin.mixed_precision_policy}")

    def get_optimizer_grouped_parameters(self, named_parameters: Dict[str, torch.nn.Parameter], decay_parameters: List[str]):
        optimizer_grouped_parameters = dict()
        for n, p in named_parameters.items():
            if not p.requires_grad:
                continue
            lr = self.args.learning_rate
            betas = (self.args.adam_beta1, self.args.adam_beta2)
            if n.endswith(".weight"):
                scale = n.removesuffix("weight") + "scale"
                if scale in named_parameters:
                    lr = self.args.learning_rate_code
                    betas = (self.args.adam_beta1_code, self.args.adam_beta2_code)
            weight_decay = (self.args.weight_decay if n in decay_parameters else 0)
            if (lr, betas, weight_decay) not in optimizer_grouped_parameters:
                optimizer_grouped_parameters[(lr, betas, weight_decay)] = {"params": [], "lr": lr, "betas": betas, "weight_decay": weight_decay}
            optimizer_grouped_parameters[(lr, betas, weight_decay)]["params"].append(p)

        return list(optimizer_grouped_parameters.values())

    def training_init(self, model):
        logger.info(f"{model}")
        if not self.args.disable_compile:
            patch_compile_state(model, self.accelerator.state, self.args.gradient_checkpointing)
        self.step = 0
        if self.args.train_method == TrainingMethod.PV_TUNING or self.args.train_method == TrainingMethod.PV_TUNING_FULL:
            def wrap(fn):
                @wraps(fn)
                def wrapped_fn(*args, **kwargs):
                    ret = fn(*args, **kwargs)
                    pv_tuning_update(model, self.args.code_update_ratio, self.args.code_trust_ratio)
                    return ret

                return wrapped_fn

            model.zero_grad = wrap(model.zero_grad)

    def _save(self, output_dir=None, state_dict: dict[str, Tensor] | None=None):
        if state_dict is not None:
            _state_dict = {}
            remove_keys = []
            for key, val in state_dict.items():
                if key.endswith(".bits"):
                    prefix = key.removesuffix("bits")
                    remove_keys.append(prefix + "weight")
                    remove_keys.append(prefix + "bits")
                    bits = val
                    weight = state_dict[prefix + "weight"]
                    scale = state_dict[prefix + "scale"]
                    zero = state_dict[prefix + "zero"]
                    qweight = pack_quantize(weight, scale, zero, bits)
                    _state_dict[prefix + "qweight"] = qweight
            for key in remove_keys:
                state_dict.pop(key)
            
            state_dict.update(_state_dict)
            torch_dtype = get_model_args().torch_dtype
            for key, val in state_dict.items():
                if val.is_floating_point() and val.dtype != torch_dtype:
                    state_dict[key] = val.to(torch_dtype)
            
        return super()._save(output_dir, state_dict)

    def training_step_init(self):
        self.step += 1
        self.step_log = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # self.log_gpu_mem("before forward")
        output = model(**inputs)
        # self.log_gpu_mem("after forward")
        return output

    def evaluate(self, *args, **kwargs):
        lm_head = self.model.get_output_embeddings()
        is_cuda = lm_head.weight.is_cuda
        if not is_cuda:
            lm_head.to("cuda", non_blocking=True)
        out = super(BaseTrainer, self).evaluate(*args, **kwargs)
        if not is_cuda:
            lm_head.to("cpu", non_blocking=True)
        return out

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:

        inputs = self._prepare_inputs(inputs)
        model.eval()
        with torch.no_grad():
            losses = model(labels=inputs["input_ids"], **inputs)
            loss = losses.mean()

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, losses, losses)
