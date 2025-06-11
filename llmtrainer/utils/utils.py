# coding=utf-8

import logging
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

from .arguments_init import get_tokenizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def do_package(b: Dict[str, Tensor]):
    mask = b.pop("attention_mask").bool()
    b["input_ids"] = b["input_ids"][mask][None]
    b["labels"] = b["labels"][mask][None]
    b["position_ids"] = b["position_ids"][mask][None]
    seqlens = mask.sum(1)
    b["cu_seqlens_q"] = F.pad(seqlens.cumsum(0), (1, 0)).int()
    b["cu_seqlens_k"] = b["cu_seqlens_q"]
    b["max_seqlen_q"] = seqlens.max().item()
    b["max_seqlen_k"] = b["max_seqlen_q"]
    if "start_q" in b:
        b["tree_dfs_order_start_q"] = b["start_q"][mask]
        b["tree_dfs_order_end_k"] = b["end_k"][mask]
    return b

def to_pt(x):
    if isinstance(x, list):
        return torch.tensor(x)
    if isinstance(x, Tensor):
        return x
    raise NotImplementedError

def concat_pt(x, y):
    if isinstance(x, list):
        return torch.tensor(x + y)
    if isinstance(x, Tensor):
        return torch.cat([x ,y])
    raise NotImplementedError

def logger_ids(name, ids):
    tokenizer = get_tokenizer()
    logger.info(f"<|{name}_len|>{len(ids)}")
    logger.info(f"<|{name}|>{tokenizer.decode(ids)}<|{name}|>")

def get_key(inputs: List[Dict[str, Any]], key: str):
    return [ex[key] for ex in inputs]
