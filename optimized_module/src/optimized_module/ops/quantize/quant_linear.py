import torch
from torch import nn, Tensor
from torch.nn import Parameter
import torch.nn.functional as F

from .utils_quantize import dequantize, unpack

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
        # dz = - dw.reshape(*zero.shape, -1).sum(dim=-1) * scale ## fast
        dz = - (dw.reshape(*zero.shape, -1) * scale.unsqueeze(-1)).sum(dim=-1) ## same
        ds = (dw.reshape(*zero.shape, -1) * (qw.reshape(*zero.shape, -1) - zero.unsqueeze(-1))).sum(dim=-1) 
    else:
        ds, dz = None, None
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
        dx, _, ds, dz, db = quantized_linear_bwd(dout, x, qw, scale, zero, bias, bits, ow_v, ow_i)
        return dx, None, ds, dz, db, None, None, None

def quantized_linear(x, qweight, scale, zero, bias=None, bits=2, outlier_weight=None) -> Tensor:
    return QuantizedLinearFn.apply(x, qweight, scale, zero, bias, bits, outlier_weight, None)

class QuantizedLinear(nn.Module):

    __constants__ = ["in_features", "out_features", "group_size", "bits"]
    in_features: int
    out_features: int
    group_size: int
    bits: int 
    qweight: Tensor
    result = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 4, 
        group_size: int = 128, 
        enable_outlier: bool = False,
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

    def forward(self, input: Tensor) -> Tensor:
        return QuantizedLinearFn.apply(input, self.qweight, self.scale, self.zero, self.bias, self.bits, self.outlier_weight_sparse_values, self.outlier_weight_sparse_indices)
        
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, group_size={self.group_size}, bits={self.bits}"
