"""CPU reference HiF4 / NVFP4 quantizers (cpu-ref)."""

from hif4_nvfp4.constants import DEVICE_TAG
from hif4_nvfp4.formats import (
    decode_e2m1,
    decode_e4m3,
    decode_e6m2,
    decode_s1p2,
    encode_e2m1,
    encode_e4m3,
    encode_e6m2,
    encode_s1p2,
)
from hif4_nvfp4.hif4 import HiF4Tensor, dequantize_hif4, fakequant_hif4, quantize_hif4
from hif4_nvfp4.nvfp4 import NVFP4Tensor, dequantize_nvfp4, fakequant_nvfp4, quantize_nvfp4
from hif4_nvfp4.pack import H2PackResult, pack_h2

__all__ = [
    "DEVICE_TAG",
    "HiF4Tensor",
    "NVFP4Tensor",
    "H2PackResult",
    "decode_e2m1",
    "decode_e4m3",
    "decode_e6m2",
    "decode_s1p2",
    "encode_e2m1",
    "encode_e4m3",
    "encode_e6m2",
    "encode_s1p2",
    "dequantize_hif4",
    "dequantize_nvfp4",
    "fakequant_hif4",
    "fakequant_nvfp4",
    "quantize_hif4",
    "quantize_nvfp4",
    "pack_h2",
]
