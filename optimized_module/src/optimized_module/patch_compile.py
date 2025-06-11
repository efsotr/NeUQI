import torch
from torch.distributed.fsdp import ShardingStrategy
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP
from accelerate.state import AcceleratorState
from accelerate.utils.dataclasses import DistributedType

from functools import partial

@torch.compile(dynamic=True)
def compiled_act_fn(act_fn, x, y):
    return act_fn(x) * y

def mlp_forward(self: LlamaMLP, x):
    return self.down_proj(compiled_act_fn(self.act_fn, self.gate_proj(x), self.up_proj(x)))

def patch_compile(model: LlamaForCausalLM, distributed_type = "NO", zero_stage: int = 0, gradient_checkpointing: bool = False, conservative=False):
    if not conservative and gradient_checkpointing and zero_stage < 3:
        for i in range(len(model.model.layers)):
            model.model.layers[i].forward = torch.compile(model.model.layers[i].forward, dynamic=True)

    if conservative or zero_stage == 3:
        for name, module in model.named_modules():
            if name.endswith("norm"):
                module.forward = torch.compile(module.forward, dynamic=True)
            if name.endswith("mlp") and hasattr(module, "gate_proj"):
                module.forward = partial(mlp_forward, module)

def ShardingStrategy_to_zero_stage(sharding_strategy: ShardingStrategy):
    if sharding_strategy == ShardingStrategy.FULL_SHARD:
        return 3
    elif sharding_strategy == ShardingStrategy.SHARD_GRAD_OP:
        return 2
    elif sharding_strategy == ShardingStrategy.NO_SHARD:
        return 0
    else:
        raise ValueError(f"Unsupported sharding strategy: {sharding_strategy}")

def patch_compile_state(model, state: AcceleratorState, gradient_checkpointing: bool = False):
    if state.distributed_type == DistributedType.DEEPSPEED:
        patch_compile(model, state.distributed_type, state.deepspeed_plugin.zero_stage, gradient_checkpointing)
    elif state.distributed_type == DistributedType.FSDP:
        patch_compile(model, state.distributed_type, ShardingStrategy_to_zero_stage(state.fsdp_plugin.sharding_strategy), gradient_checkpointing)
    else:
        patch_compile(model, state.distributed_type, 0, gradient_checkpointing)