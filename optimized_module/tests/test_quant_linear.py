import pytest

import torch
import optimized_module.ops.quantize.quant_linear as quant_linear
from optimized_module.ops.quantize.utils_quantize import pack_2bit
from optimized_module.util import torch_assert_close

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

def forward(func, dout, kwargs, dtype=None):
    out = func(**kwargs)
    if dout is None:
        dout = torch.randn_like(out, dtype=dtype).float()
    out.backward(dout)
    grads = [arg.grad for arg in kwargs.values() if torch.is_tensor(arg) and torch.is_floating_point(arg) and arg.grad is not None]
    for arg in kwargs.values():
        if torch.is_tensor(arg) and torch.is_floating_point(arg):
            arg.grad = None
    return out.detach().float(), grads, dout


size_list = [2048, 3072, 4096, 8192]
seqlen_list = [1, 16, 230, 566, 4096, 10001]

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "in_dim,out_dim", 
    [(i, i) for i in size_list] + \
    [(i, int(i * 3.5)) for i in size_list] + \
    [(int(i * 3.5), i) for i in size_list] 
)
@pytest.mark.parametrize("G", [-1, 32, 64, 128])
@pytest.mark.parametrize("seqlen", seqlen_list)
def test_quant_2bit_linear(dtype, in_dim, out_dim, G, seqlen):
    if G == -1:
        G = in_dim

    for iter in range(16):
        a = torch.randn((seqlen, in_dim), device="cuda", dtype=dtype) * in_dim ** -0.5
        b = torch.randn((out_dim, in_dim), device="cuda", dtype=dtype) * in_dim ** -0.5
        bmax = b.view(out_dim, -1, G).max(dim=-1).values
        bmin = b.view(out_dim, -1, G).min(dim=-1).values
        scale = (bmax - bmin) / 3
        zero = - bmin / scale
        qb = torch.round(b.view(out_dim, -1, G) / scale.unsqueeze(-1) + zero.unsqueeze(-1)).view(out_dim, in_dim).clamp(0, 3).to(torch.uint8)
        qb_packed = pack_2bit(qb)

        for tensor in [a, scale, zero]:
            tensor.requires_grad_(True)

        args = {"x": a, "qweight": qb_packed, "scale": scale, "zero": zero, "bias": None, "bits": 2}
        out_ref, grad_ref, dout = forward(quant_linear.quant_linear_fwd, None, args, dtype)
        out_fast, grad_fast, dout = forward(quant_linear.quant_linear, dout, args)

        assert torch_assert_close(out_ref, out_fast, 0, 0)
        for i in range(len(grad_ref)):
            assert torch_assert_close(grad_ref[i], grad_fast[i], 1.6e-2, 1e-6)
        