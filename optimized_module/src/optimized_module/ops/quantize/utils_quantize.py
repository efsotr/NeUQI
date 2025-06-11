import torch

## uint8

def pack_2bit(x):
    return x[:, ::4] + (x[:, 1::4] << 2) + (x[:, 2::4] << 4) + (x[:, 3::4] << 6)

def pack_4bit(x):
    return x[:, ::2] + (x[:, 1::2] << 4)

def pack_xbit(x, bits):
    bits_8 = torch.arange(8, device=x.device, dtype=x.dtype)
    return ((x.view(x.size(0), -1, 8, 1) >> bits_8[: bits]) & 1).transpose(-1, -2).bitwise_left_shift(bits_8).sum(dim=-1, dtype=x.dtype).view(x.size(0), -1)

def unpack_2bit(x):
    return torch.stack([x & 3, (x >> 2) & 3, (x >> 4) & 3, (x >> 6)], dim=-1).view(x.size(0), -1)

def unpack_4bit(x):
    return torch.stack([x & 15, x >> 4], dim=-1).view(x.size(0), -1)

def unpack_xbit(x, bits):
    bits_8 = torch.arange(8, device=x.device, dtype=x.dtype)
    return ((x.view(x.size(0), -1, bits, 1) >> bits_8) & 1).transpose(-1, -2).bitwise_left_shift(bits_8[: bits]).sum(dim=-1, dtype=x.dtype).view(x.size(0), -1)

def pack(x, bits):
    if bits == 2:
        return pack_2bit(x)
    if bits == 4:
        return pack_4bit(x)
    if bits == 8:
        return x
    if bits <= 7:
        return pack_xbit(x, bits)
    raise NotImplementedError

def unpack(x, bits):
    if bits == 2:
        return unpack_2bit(x)
    if bits == 4:
        return unpack_4bit(x)
    if bits == 8:
        return x
    if bits <= 7:
        return unpack_xbit(x, bits)
    raise NotImplementedError

def dequantize(dex, scale, zero):
    return ((dex.view(*scale.shape, -1) - zero.unsqueeze(-1)) * scale.unsqueeze(-1)).view(dex.shape)

def unpack_dequantize(x, scale, zero, bits):
    return dequantize(unpack(x, bits), scale, zero)
