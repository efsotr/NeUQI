import math
from enum import Enum
import torch
from torch import nn, Tensor
from torch.nn import Parameter
import torch.nn.functional as F

from .utils_quantize_work import dequantize, unpack, pack, quantize, unpack_dequantize, pack_quantize, quantize_dequantize

class TrainingMethod(str, Enum):
    QUANTIZE_PARAM_ONLY = "default"
    STE = "ste"
    PV_TUNING = "pv_tuning"
    PV_TUNING_FULL = "pv_tuning_full"

str2enum_map = {
    "default": TrainingMethod.QUANTIZE_PARAM_ONLY,
    "ste": TrainingMethod.STE,
    "pv_tuning": TrainingMethod.PV_TUNING,
    "pv_tuning_full": TrainingMethod.PV_TUNING_FULL
}

@torch.compile
def quantized_linear_fwd(x, qweight, scale, zero, bias, bits, outlier_weight, outlier_weight_indices):
    w = dequantize(unpack(qweight, bits), scale, zero) 
    if outlier_weight_indices is not None:
        outlier_weight = torch.sparse_coo_tensor(outlier_weight_indices, outlier_weight, w.size())
    if outlier_weight is not None:
        w = w + outlier_weight
    return F.linear(x, w, bias)

@torch.compile
def quantized_linear_bwd(dout, x, qweight, scale, zero, bias, bits, outlier_weight, outlier_weight_indices):
    qw = unpack(qweight, bits)
    w = dequantize(qw, scale, zero)
    if outlier_weight_indices is not None:
        outlier_weight = torch.sparse_coo_tensor(outlier_weight_indices, outlier_weight, w.size())
    if outlier_weight is not None:
        w = w + outlier_weight
    dx = dout @ w # if x.requires_grad else None
    if zero.requires_grad or scale.requires_grad:
        dw = dout.reshape(-1, dout.size(-1)).T @ x.reshape(-1, x.size(-1))
    if zero.requires_grad:
        # dz = - dw.reshape(*zero.shape, -1).sum(dim=-1) * scale ## fast without compile
        dz = - (dw.reshape(*zero.shape, -1) * scale.unsqueeze(-1)).sum(dim=-1) ## same
    else:
        dz = None
    if scale.requires_grad:
        ds = (dw.reshape(*zero.shape, -1) * (qw.reshape(*zero.shape, -1) - zero.unsqueeze(-1))).sum(dim=-1) 
    else:
        ds = None
    db = dout.sum(dim=tuple(range(len(x.size()) - 1))) if bias is not None and bias.requires_grad else None
    return dx, None, ds, dz, db

class QuantizedLinearFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, qw, scale, zero, bias, bits, ow_v, ow_i):
        ctx.save_for_backward(x, qw, scale, zero, bias, ow_v, ow_i)
        ctx.bits = bits
        out = quantized_linear_fwd(x, qw, scale, zero, bias, bits, ow_v, ow_i)
        return out
    
    @staticmethod
    def backward(ctx, dout):
        x, qw, scale, zero, bias, ow_v, ow_i = ctx.saved_tensors
        bits = ctx.bits
        dx, dw, ds, dz, db = quantized_linear_bwd(dout, x, qw, scale, zero, bias, bits, ow_v, ow_i)
        return dx, dw, ds, dz, db, None, None, None

@torch.compile
def ste_quantized_linear_fwd(x, w, scale, zero, bias, bits, outlier_weight, outlier_weight_indices, train_method: TrainingMethod):
    # if outlier_weight_indices is not None:
    #     outlier_weight = torch.sparse_coo_tensor(outlier_weight_indices, outlier_weight, w.size())
    # if outlier_weight is not None:
    #     w = w + outlier_weight
    if train_method == TrainingMethod.PV_TUNING or train_method == TrainingMethod.PV_TUNING_FULL:
        return F.linear(x, w, bias)
    if train_method == TrainingMethod.STE:
        return F.linear(x, quantize_dequantize(w, scale, zero, bits), bias)
    raise NotImplementedError(f"train_method {train_method} not implemented")

@torch.compile
def ste_quantized_linear_bwd(dout, x, w, scale, zero, bias, bits, outlier_weight, outlier_weight_indices, train_method: TrainingMethod):
    # if outlier_weight_indices is not None:
    #     outlier_weight = torch.sparse_coo_tensor(outlier_weight_indices, outlier_weight, w.size())
    # if outlier_weight is not None:
    #     w = w + outlier_weight
    if x.requires_grad:
        if train_method == TrainingMethod.PV_TUNING or train_method == TrainingMethod.PV_TUNING_FULL:
            dx = dout @ w
        else:
            dx = dout @ quantize_dequantize(w, scale, zero, bits)
    else:
        dx = None

    if w.requires_grad or zero.requires_grad or scale.requires_grad:
        dw = dout.reshape(-1, dout.size(-1)).T @ x.reshape(-1, x.size(-1))
    if zero.requires_grad:
        # dz = - dw.reshape(*zero.shape, -1).sum(dim=-1) * scale # fast when without compile
        dz = - (dw.reshape(*zero.shape, -1) * scale.unsqueeze(-1)).sum(dim=-1) # same
    else:
        dz = None
    if scale.requires_grad:
        qw = dequantize(w, scale, zero)
        ds = (dw.reshape(*zero.shape, -1) * (qw.reshape(*zero.shape, -1) - zero.unsqueeze(-1))).sum(dim=-1) 
    else:
        ds = None
    db = dout.sum(dim=tuple(range(len(x.size()) - 1))) if bias is not None and bias.requires_grad else None
    return dx, dw, ds, dz, db

class STEQuantizedLinearFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w, scale, zero, bias, bits, ow_v, ow_i, train_method):
        ctx.save_for_backward(x, w, scale, zero, bias, ow_v, ow_i)
        ctx.bits = bits
        ctx.train_method = train_method
        out = ste_quantized_linear_fwd(x, w, scale, zero, bias, bits, ow_v, ow_i, train_method)
        return out
    
    @staticmethod
    def backward(ctx, dout):
        x, w, scale, zero, bias, ow_v, ow_i = ctx.saved_tensors
        bits, train_method = ctx.bits, ctx.train_method
        dx, dw, ds, dz, db = ste_quantized_linear_bwd(dout, x, w, scale, zero, bias, bits, ow_v, ow_i, train_method)
        return dx, dw, ds, dz, db, None, None, None, None

def quantized_linear(x, weight, scale, zero, bias=None, bits=2, outlier_weight=None, train_method=TrainingMethod.QUANTIZE_PARAM_ONLY) -> Tensor:
    if train_method == TrainingMethod.QUANTIZE_PARAM_ONLY:
        return QuantizedLinearFn.apply(x, weight, scale, zero, bias, bits, outlier_weight, None)
    else:
        return STEQuantizedLinearFn.apply(x, weight, scale, zero, bias, bits, outlier_weight, None, train_method)

class QuantizedLinear(nn.Module):

    __constants__ = ["in_features", "out_features", "group_size", "bits"]
    in_features: int
    out_features: int
    group_size: int
    bits: int 
    qweight: Tensor
    result = None
    train_method: TrainingMethod

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 4, 
        group_size: int = 128, 
        enable_outlier: bool = False,
        train_method: TrainingMethod = TrainingMethod.QUANTIZE_PARAM_ONLY,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        if group_size == -1:
            group_size = in_features
        self.group_size = group_size
        self.train_method = TrainingMethod.QUANTIZE_PARAM_ONLY
        self.qweight = Parameter(
            torch.empty((out_features, in_features * bits // 8), device=device, dtype=torch.uint8), False
        )
        self.scale = Parameter(torch.empty((out_features, in_features // group_size), **factory_kwargs))
        self.zero = Parameter(torch.empty((out_features, in_features // group_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        
        if enable_outlier:
            self.outlier_weight_sparse_indices = Parameter(torch.empty(0, dtype=torch.int64, device=device), requires_grad=False)
            self.outlier_weight_sparse_values = Parameter(torch.empty(2, 0, **factory_kwargs), requires_grad=False)
            self.outlier_weight_sparse_size = Parameter(torch.empty(2, dtype=torch.int64, device=device), requires_grad=False)
        else:
            self.register_parameter("outlier_weight_sparse_indices", None)
            self.register_parameter("outlier_weight_sparse_values", None)
            self.register_parameter("outlier_weight_sparse_size", None)

        self.update_train_method(train_method)

    def forward(self, input: Tensor) -> Tensor:
        if self.train_method == TrainingMethod.QUANTIZE_PARAM_ONLY:
            return QuantizedLinearFn.apply(input, self.qweight, self.scale, self.zero, self.bias, self.bits, self.outlier_weight_sparse_values, self.outlier_weight_sparse_indices)
        else:
            return STEQuantizedLinearFn.apply(input, self.weight, self.scale, self.zero, self.bias, self.bits, self.outlier_weight_sparse_values, self.outlier_weight_sparse_indices, self.train_method)
        
    def update_train_method(self, train_method: TrainingMethod):
        self.train_method = train_method
        if train_method == TrainingMethod.PV_TUNING:
            self.qweight.data = self.qweight.data.view(torch.float)
        else:
            self.qweight.data = self.qweight.data.view(torch.uint8)

    @torch.no_grad()
    def dequantized(self):
        assert self.train_method != TrainingMethod.QUANTIZE_PARAM_ONLY
        weight = unpack_dequantize(self.qweight.view(torch.uint8), self.scale, self.zero, self.bits)
        if getattr(self, "weight", None) is not None:
            self.weight.data.copy_(weight)
        else:
            self.weight = Parameter(weight)
        if self.train_method != TrainingMethod.PV_TUNING:
            self.register_parameter("qweight", None)

    @torch.compile
    @torch.no_grad()
    def pv_tuning_update(self, code_update_ratio: float, code_trust_ratio: float):
        if self.train_method == TrainingMethod.PV_TUNING_FULL and code_update_ratio == 1 and code_trust_ratio >= 100:
            self.weight.data.copy_(quantize_dequantize(self.weight, self.scale, self.zero, self.bits))
            return
        
        assert self.train_method == TrainingMethod.PV_TUNING

        flat_ref_weight = self.weight.view(-1, self.group_size)
        qweight = unpack(self.qweight.view(torch.uint8), self.bits)
        flat_prev_weight = dequantize(qweight, self.scale, self.zero).view(-1, self.group_size)
        code_update_num = int(math.ceil(flat_ref_weight.size(0) * code_update_ratio))

        selection = (flat_ref_weight - flat_prev_weight).norm(p=2, dim=-1).topk(k=code_update_num).indices
        flat_scale = self.scale.view(-1)[selection]
        flat_zero = self.zero.view(-1)[selection]
        flat_updated_qweight = quantize(flat_ref_weight[selection], flat_scale, flat_zero, self.bits)
        flat_updated_weight = dequantize(flat_updated_qweight, flat_scale, flat_zero)

        flat_cumsum_norm = (flat_updated_weight - flat_prev_weight[selection]).square().sum(dim=-1).cumsum(dim=0).sqrt()
        trust_norm = code_trust_ratio * flat_prev_weight.norm(p=2)
        p = torch.searchsorted(flat_cumsum_norm, trust_norm, right=True) + 1
        cond = torch.arange(code_update_num, device=p.device).less(p).unsqueeze(-1)
        flat_qweight = qweight.view(-1, self.group_size)
        flat_qweight[selection] = torch.where(cond, flat_updated_qweight, flat_qweight[selection])
        flat_prev_weight[selection] = torch.where(cond, flat_updated_weight, flat_prev_weight[selection])

        self.qweight.data.copy_(pack(qweight, self.bits).view(self.qweight.dtype))
        self.weight.data.copy_(flat_prev_weight.view(*self.weight.shape))
    
    # @torch.no_grad()
    # def quantized(self):
    #     qweight = pack_quantize(self.weight, self.scale, self.zero, self.bits)
    #     if getattr(self, "qweight", None) is not None:
    #         self.qweight.data.copy_(qweight)
    #     else:
    #         self.qweight = Parameter(qweight, requires_grad=False)
    #     self.register_parameter("weight", None)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.train_method == TrainingMethod.QUANTIZE_PARAM_ONLY:
            return 
        destination[prefix + "bits"] = self.bits
        
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, group_size={self.group_size}, bits={self.bits}"
