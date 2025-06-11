import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional

def cross_entropy_fwd(x, l, idx, ignore_idx):
    pad_mask = idx.eq(ignore_idx)
    idx = idx.clamp(0)
    s = torch.gather(x, -1, idx)
    out = (l - s).masked_fill(pad_mask, 0)
    return out.view(-1), idx, pad_mask

def cross_entropy_bwd(x, l, dout, idx):
    return (x - l).exp_().mul_(dout).scatter_add_(-1, idx, -dout)

def cross_entropy_bwd_x_no_need(x, l, dout, idx):
    return x.sub_(l).exp_().mul_(dout).scatter_add_(-1, idx, -dout)

class EffCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, idx: Tensor, ignore_index: int = -100, reduction: str = "mean", x_no_need_in_bwd: bool = True):
        assert reduction in ["mean", "sum", "none"]
        l = torch.logsumexp(x.float(), dim=-1, keepdim=True)
        idx = idx.view(-1, 1)
        out, idx, pad_mask = cross_entropy_fwd(x, l, idx, ignore_index)
        ctx.save_for_backward(x, l, idx, pad_mask)
        ctx.x_no_need_in_bwd = x_no_need_in_bwd
        ctx.reduction = reduction
        if reduction == "mean":
            return out.sum() / (pad_mask.numel() - pad_mask.sum()).to(x.dtype)
        if reduction == "sum":
            return out.sum()
        return out

    @staticmethod
    def backward(ctx, dout):
        x, l, idx, pad_mask = ctx.saved_tensors
        reduction = ctx.reduction
        if reduction == "mean":
            dout = dout / (pad_mask.numel() - pad_mask.sum()).to(x.dtype)
        if reduction in ["mean", "sum"]:
            dout = dout.repeat(x.size(0))
        dout = dout.view(-1, 1).masked_fill(pad_mask, 0)
        # dx = cross_entropy_bwd_x_no_need(x, l, dout, idx) if ctx.x_no_need_in_bwd else cross_entropy_bwd(x, l, dout, idx)
        dx = cross_entropy_bwd(x, l, dout, idx)
        return dx, None, None, None, None

def eff_cross_entropy(x, idx, ignore_index=-100, reduction="mean", x_no_need_in_bwd=True):
    return EffCrossEntropyFn.apply(x, idx, ignore_index, reduction, x_no_need_in_bwd)


