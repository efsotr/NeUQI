import logging
from functools import partial
from typing import Any, Callable, Dict, List, Union

import torch
from deepspeed import DeepSpeedEngine
from deepspeed.runtime import bf16_optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3, instrument_w_nvtx
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from torch import nn
from transformers.trainer import Trainer, is_sagemaker_mp_enabled

from .arguments import MaxTrainingArguments


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BaseTrainer(Trainer):
    args : MaxTrainingArguments
    wrap_fisrt = True
    engine : DeepSpeedEngine
    step_log : Dict[str, Any] = {}
    num_fns: Dict[str, Callable] = []

    def training_step_init(self):
        self.step_log = {}

    def training_init(self, model):
        self.num_fns = {}

    def log_gpu_mem(self, state = ""):
        if self.args.local_rank == 0:
            logger.info(f"{state}  "
                        f"MA  {torch.cuda.memory_allocated(self.args.device) / 1024 ** 3:.4f} GiB    "
                        f"CA  {torch.cuda.memory_reserved(self.args.device) / 1024 ** 3: .4f} GiB    "
                        f"MAX_MA  {torch.cuda.max_memory_allocated(self.args.device) / 1024 ** 3: .4f} GiB    "
                        f"MAX_CA  {torch.cuda.max_memory_reserved(self.args.device) / 1024 ** 3: .4f} GiB    ")
            torch.cuda.reset_peak_memory_stats(self.args.device)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[Dict[str, int], List[Dict[str, torch.torch.Tensor]]]], num_items_in_batch=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # structure of inputs
        if self.wrap_fisrt:
            self.wrap_fisrt = False
            if self.is_deepspeed_enabled:
                self.engine : DeepSpeedEngine = self.accelerator.deepspeed_engine_wrapped.engine
                def empty_save_checkpoint(self, save_dir, tag=None, client_state={}, save_latest=True, exclude_frozen_parameters=False):
                    return True
                self.engine.save_checkpoint = partial(empty_save_checkpoint, self.engine)
                optimizer = self.engine.optimizer
                try:
                    optimizer.loss_scaler.raise_error_at_min_scale = False
                    if self.args.local_rank == 0:
                        logger.info('set raise_error_at_min_scale to False')
                except:
                    pass
                if isinstance(optimizer, DeepSpeedZeroOptimizer):
                    def unscale_and_clip_grads(self, grad_groups_flat, total_norm):
                        # compute combined scale factor for this group
                        combined_scale = self.loss_scale
                        if self.clip_grad > 0.:
                            # norm is in fact norm*scale
                            clip = (total_norm / self.loss_scale) / self.clip_grad
                            clip = torch.clamp(clip, min=1.0)
                            combined_scale = clip * self.loss_scale
                        combined_scale = 1. / combined_scale

                        for grad in grad_groups_flat:
                            if isinstance(grad, list):
                                sub_partitions = grad
                                for g in sub_partitions:
                                    g.data.mul_(combined_scale)
                            else:
                                grad.data.mul_(combined_scale)
                    optimizer.unscale_and_clip_grads = partial(unscale_and_clip_grads, optimizer)
                    def _has_inf_or_nan(x: torch.Tensor, j=None):
                        # inf_or_nan = ~x.isfinite().all()
                        inf_or_nan = ~x.sum(dtype=torch.float).isfinite()
                        return inf_or_nan.float()
                    optimizer._has_inf_or_nan = _has_inf_or_nan
                elif isinstance(optimizer, bf16_optimizer.BF16_Optimizer):
                    from deepspeed.runtime.utils import (
                        get_accelerator,
                        get_global_norm_of_tensors,
                        graph_cache,
                        graph_process,
                    )
                    def clip_tensors_by_global_norm(input_tensors, max_norm=1.0, global_norm=None, mpu=None, eps=1e-6, use_graph=False):
                        """Clip list of tensors by global norm.
                        Args:
                            input_tensors: List of tensors to be clipped
                            global_norm (float, optional): Precomputed norm. Defaults to None.
                            mpu (optional): model parallelism unit. Defaults to None.
                            eps (float, optional): epsilon value added to grad norm. Defaults to 1e-6
                        Returns:
                            float: the global norm
                        """
                        if global_norm is None:
                            global_norm = get_global_norm_of_tensors(input_tensors, mpu=mpu, use_graph=use_graph)
                        clip_coef = global_norm / max_norm
                        if clip_coef > 1:
                            clip_coef = 1. / clip_coef
                            if use_graph:

                                def clip_tensors(_tensor_list, _clip_coef_tensor):
                                    for t in _tensor_list:
                                        t.detach().mul_(_clip_coef_tensor)

                                if 'clip_coef_tensor' not in graph_cache:
                                    # Alloc memory
                                    graph_cache['clip_coef_tensor'] = torch.tensor(clip_coef,
                                                                                dtype=torch.float32).to(get_accelerator().device_name())
                                clip_coef_tensor = graph_cache['clip_coef_tensor']
                                clip_coef_tensor.copy_(torch.tensor(clip_coef, dtype=torch.float32))
                                graph_process(False, clip_tensors, input_tensors, clip_coef_tensor)

                            else:
                                for t in input_tensors:
                                    t.detach().mul_(clip_coef)
                        return global_norm

                    bf16_optimizer.clip_tensors_by_global_norm = clip_tensors_by_global_norm
                elif isinstance(optimizer, FP16_UnfusedOptimizer):
                    def unscale_and_clip_grads(self, total_norm, apply_scale=True):
                        # compute combined scale factor for this group
                        combined_scale = self.cur_scale
                        if self.clip_grad > 0.:
                            # norm is in fact norm*scale
                            clip = (total_norm / self.cur_scale) / self.clip_grad
                            if clip > 1:
                                combined_scale = clip * self.cur_scale

                        if apply_scale:
                            for group in self.fp32_groups:
                                for param in group:
                                    if param.grad is not None:
                                        param.grad.data.mul_(1. / combined_scale)

                        return combined_scale
                    optimizer.unscale_and_clip_grads = partial(unscale_and_clip_grads, optimizer)
                elif isinstance(optimizer, DeepSpeedZeroOptimizer_Stage3):
                    @instrument_w_nvtx
                    def unscale_and_clip_grads(self, sub_group_id, total_norm):
                        # compute combined scale factor for this group
                        combined_scale = self.loss_scale
                        if self.clip_grad > 0.:
                            # norm is in fact norm*scale
                            clip = (total_norm / self.loss_scale) / self.clip_grad
                            clip = torch.clamp(clip, min=1.0)
                            combined_scale = clip * self.loss_scale

                        self.fp32_partitioned_groups_flat[sub_group_id].grad.mul_(1. / combined_scale)

                    optimizer.unscale_and_clip_grads = partial(unscale_and_clip_grads, optimizer)

            self.training_init(model)

        self.training_step_init()
        return super().training_step(model, inputs, num_items_in_batch)

    # def evaluate(self, *args, **kwargs):
    #     torch.cuda.empty_cache()
    #     return super(BaseTrainer, self).evaluate(*args, **kwargs)

    # def num_tokens(self, train_dl: DataLoader, max_steps: Optional[int] = None) -> int:
    #     """
    #     Helper to get number of tokens in a [`~torch.utils.data.DataLoader`] by enumerating dataloader.
    #     """
    #     train_tokens = 0
    #     try:
    #         for step, batchs in enumerate(train_dl):
    #             tokens = sum(batch['input_ids'].numel() for batch in batchs['batch'])
    #             train_tokens += tokens
    #         if max_steps is not None:
    #             return train_tokens / len(train_dl) * max_steps
    #         return train_tokens
    #     except KeyError:
    #         logger.warning("Cannot get num_tokens from dataloader")
    #         return train_tokens

    # def log(self, logs: Dict[str, float]) -> None:
    #     """
    #     Log `logs` on the various objects watching training.

    #     Subclass and override this method to inject custom behavior.

    #     Args:
    #         logs (`Dict[str, float]`):
    #             The values to log.
    #     """
    #     if self.state.epoch is not None:
    #         logs["epoch"] = round(self.state.epoch, 5)
    #     if self.args.include_num_input_tokens_seen:
    #         logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

    #     output = {**logs, **{"step": self.state.global_step}, **self.step_log, **self.nums}
    #     self.state.log_history.append(output)
    #     self.control = self.callback_handler.on_log(self.args, self.state, self.control, {**logs, **self.step_log, **self.nums})

    def get_optimizer_grouped_parameters(self, named_parameters: Dict[str, torch.nn.Parameter], decay_parameters: List[str]):
        optimizer_grouped_parameters = dict()
        for n, p in named_parameters.items():
            if not p.requires_grad:
                continue
            lr = self.args.learning_rate
            weight_decay = (self.args.weight_decay if n in decay_parameters else 0)
            if (lr, weight_decay) not in optimizer_grouped_parameters:
                optimizer_grouped_parameters[(lr, weight_decay)] = {"params": [], "lr": lr, weight_decay: "weight_decay"}
            optimizer_grouped_parameters[(lr, weight_decay)]["params"].append(p)

        return list(optimizer_grouped_parameters.values())

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)

            optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(dict(opt_model.named_parameters()), decay_parameters)

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        if self.args.local_rank == 0:
            logger.info(f" optimizer number of params {[len(g['params']) for g in self.optimizer.param_groups]}")
            for g in self.optimizer.param_groups:
                logger.info(f" optimizer parameters {[(k, v) for k, v in g.items() if k != 'params']}")

        return self.optimizer

