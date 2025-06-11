import torch
from torch import Tensor

## uint8

def pack_2bit(x: Tensor):
    return x[:, ::4] + (x[:, 1::4] << 2) + (x[:, 2::4] << 4) + (x[:, 3::4] << 6)

def pack_4bit(x: Tensor):
    return x[:, ::2] + (x[:, 1::2] << 4)

def pack_xbit(x: Tensor, bits: int):
    bits_8 = torch.arange(8, device=x.device, dtype=x.dtype)
    return ((x.view(x.size(0), -1, 8, 1) >> bits_8[: bits]) & 1).transpose(-1, -2).bitwise_left_shift(bits_8).sum(dim=-1, dtype=x.dtype).view(x.size(0), -1)

def unpack_2bit(x: Tensor):
    return torch.stack([x & 3, (x >> 2) & 3, (x >> 4) & 3, (x >> 6)], dim=-1).view(x.size(0), -1)

def unpack_4bit(x: Tensor):
    return torch.stack([x & 15, x >> 4], dim=-1).view(x.size(0), -1)

def unpack_xbit(x: Tensor, bits: int):
    bits_8 = torch.arange(8, device=x.device, dtype=x.dtype)
    return ((x.view(x.size(0), -1, bits, 1) >> bits_8) & 1).transpose(-1, -2).bitwise_left_shift(bits_8[: bits]).sum(dim=-1, dtype=x.dtype).view(x.size(0), -1)

@torch.compile
def pack(x: Tensor, bits: int):
    if bits == 2:
        return pack_2bit(x)
    if bits == 4:
        return pack_4bit(x)
    if bits == 8:
        return x
    if bits <= 7:
        return pack_xbit(x, bits)
    raise NotImplementedError

@torch.compile
def unpack(x: Tensor, bits: int):
    if bits == 2:
        return unpack_2bit(x)
    if bits == 4:
        return unpack_4bit(x)
    if bits == 8:
        return x
    if bits <= 7:
        return unpack_xbit(x, bits)
    raise NotImplementedError

@torch.compile
def dequantize(q: Tensor, scale: Tensor, zero: Tensor):
    return ((q.view(*scale.shape, -1) - zero.unsqueeze(-1)) * scale.unsqueeze(-1)).view(q.shape)

@torch.compile
def unpack_dequantize(q: Tensor, scale: Tensor, zero: Tensor, bits: int):
    return dequantize(unpack(q, bits), scale, zero)

@torch.compile
def quantize(x: Tensor, scale: Tensor, zero: Tensor, bits: int):
    return (x.view(*scale.shape, -1) / scale.unsqueeze(-1) + zero.unsqueeze(-1)).view(x.shape).round().clamp(0, 2 ** bits - 1).to(torch.uint8)

@torch.compile
def pack_quantize(x: Tensor, scale: Tensor, zero: Tensor, bits: int):
    q = quantize(x, scale, zero, bits)
    return pack(q, bits)

@torch.compile
def quantize_dequantize(x: Tensor, scale: Tensor, zero: Tensor, bits: int):
    return dequantize(quantize(x, scale, zero, bits), scale, zero)