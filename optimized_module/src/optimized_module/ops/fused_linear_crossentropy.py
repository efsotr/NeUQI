import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Optional

def fused_linear_cross_entropy_fwd(x: Tensor, weight: Tensor, bias: Optional[Tensor], idx: Tensor, ignore_index: int = -100):
    N, chunk_size = x.size()
    s_idx = torch.zeros((N,), dtype=x.dtype, device=x.device)
    for chunk_start in range(0, weight.size(0), chunk_size):
        chunk = slice(chunk_start, chunk_start + chunk_size)
        s_chunk = F.linear(x, weight[chunk], None if bias is None else bias[chunk])
        chunk_mask = (idx < chunk_start) | (idx >= chunk.stop)
        s_idx += s_chunk.gather(dim=1, 
                                index=idx.masked_fill(chunk_mask, chunk_start).unsqueeze(1) - chunk_start).squeeze(1).masked_fill(chunk_mask, 0)
        if chunk_start == 0:
            s_max = s_chunk.max(dim=1, keepdim=True).values.float()
            s_sumexp_acc = (s_chunk - s_max).exp().sum(dim=1, keepdim=True)
        else:
            s_max_old = s_max
            s_max = torch.maximum(s_max, s_chunk.max(dim=1, keepdim=True).values.float())
            s_sumexp_acc = s_sumexp_acc * (s_max_old - s_max).exp() + (s_chunk - s_max).exp().sum(dim=1, keepdim=True)
    s_logsumexp = (s_sumexp_acc.log() + s_max).view(-1) # torch.float32
    neg_log_softmax = (s_logsumexp - s_idx).masked_fill_(idx == ignore_index, 0) # torch.float32
    return neg_log_softmax, s_logsumexp


def fused_linear_cross_entropy_bwd(dout: Tensor, l: Tensor, x: Tensor, weight: Tensor, bias: Optional[Tensor], idx: Tensor, ignore_index: int = -100):
    N, chunk_size = x.size()
    dx = torch.zeros_like(x, dtype=torch.float) if x.requires_grad else None
    dw = torch.empty_like(weight) if weight.requires_grad else None
    db = torch.empty_like(bias) if bias is not None else None
    dout = dout.masked_fill(idx == ignore_index, 0).view(-1, 1) # output_dtype
    l = l.view(-1, 1) # torch.float32
    for chunk_start in range(0, weight.size(0), chunk_size):
        chunk = slice(chunk_start, chunk_start + chunk_size)
        s_chunk = F.linear(x, weight[chunk], None if bias is None else bias[chunk])
        s_chunk = (s_chunk - l).exp_().mul_(dout)
        chunk_mask = (idx < chunk_start) | (idx >= chunk.stop)
        s_chunk.scatter_add_(dim=1, 
                        index=idx.masked_fill(chunk_mask, chunk_start).unsqueeze(1) - chunk_start, 
                        src=dout.neg().to(s_chunk.dtype).masked_fill(chunk_mask.unsqueeze(1), 0))
        if dx is not None:
            dx += torch.mm(s_chunk.to(weight.dtype), weight[chunk])
        if dw is not None:
            dw[chunk] = torch.mm(s_chunk.T.to(weight.dtype), x)
        if db is not None:
            db[chunk] = s_chunk.sum(dim=0).to(bias.dtype)

    _to_dtype = lambda t, t2: None if t is None else t.to(t2.dtype)
    return _to_dtype(dx, x), dw, db


class FusedLinearCrossEntropyFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Optional[Tensor], idx: Tensor, ignore_index: int = -100):
        nll_loss, l = fused_linear_cross_entropy_fwd(x, weight, bias, idx, ignore_index)
        ctx.save_for_backward(x, weight, bias, idx, l)
        ctx.ignore_index = ignore_index
        return nll_loss
    
    @staticmethod
    def backward(ctx, dout):
        x, weight, bias, idx, l = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        dx, dw, db = fused_linear_cross_entropy_bwd(dout, l, x, weight, bias, idx, ignore_index)
        return dx, dw, db, None, None


@torch.compile(dynamic=True)
def fused_linear_cross_entropy(x, weight, bias, idx, ignore_index=-100, reduction="mean", dtype=torch.float):
    assert reduction in ["none", "sum", "mean"]
    nll_loss = FusedLinearCrossEntropyFn.apply(x, weight, bias, idx, ignore_index)
    if reduction == "none":
        return nll_loss.to(dtype)
    if reduction == "sum":
        return nll_loss.sum().to(dtype)
    return (nll_loss.sum() / (idx != ignore_index).sum()).to(dtype)

class FusedLinearCrossEntropy(nn.Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear_module):
        instance = cls(0, 0, linear_module.bias is not None)
        for attr in ["in_features", "out_features", "weight", "bias"]:
            setattr(instance, attr, getattr(linear_module, attr))
            instance.in_features = linear_module.in_features
        return instance

    def forward(self, x, idx=None, ignore_index=-100, reduction="mean", dtype=torch.float):
        if idx is None:
            return F.linear(x, self.weight, self.bias)

        return fused_linear_cross_entropy(x, self.weight, self.bias, idx, ignore_index, reduction, dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"