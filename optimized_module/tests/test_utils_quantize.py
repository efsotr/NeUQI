import pytest
import torch

from optimized_module.ops.quantize.utils_quantize import pack, unpack

@pytest.mark.parametrize("bits", list(range(1, 9)))
def test_pack_and_unpack(bits):
    for iter in range(64):
        x = torch.randint(0, 2 ** bits - 1, (4096, 2048), device="cuda", dtype=torch.uint8)
        x_packed = pack(x, bits)
        x_pack_unpack = unpack(x_packed, bits)
        torch.allclose(x, x_pack_unpack)