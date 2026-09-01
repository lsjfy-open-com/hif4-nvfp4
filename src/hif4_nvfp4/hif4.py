"""HiF4 quantize / dequantize (Algorithm 1, ``docs/00-formal-model.md`` §2.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hif4_nvfp4.constants import (
    E1_16_THRESHOLD,
    E1_8_THRESHOLD,
    E6M2_NAN_CODE,
    HIF4_GROUP,
    HIF4_INTRA_MAX,
)
from hif4_nvfp4.formats import (
    decode_e6m2,
    decode_s1p2,
    e6m2_reciprocal_bf16,
    encode_e6m2,
    encode_s1p2,
    to_bf16,
)

# Algorithm 1: SF_BF16 = Vmax * (1/7)_BF16
_ONE_SEVENTH_BF16 = float(to_bf16(np.array([1.0 / HIF4_INTRA_MAX], dtype=np.float32))[0])


@dataclass
class HiF4Tensor:
    """One HiF4 unit per last-axis group of 64.

    Shapes (``prefix`` is ``x.shape[:-1]``):
      e6m2  uint8   prefix + (n_groups,)
      e1_8  uint8   prefix + (n_groups, 8)     values in {0,1}
      e1_16 uint8   prefix + (n_groups, 16)    values in {0,1}
      s1p2  uint8   prefix + (n_groups, 64)    4-bit codes in the low nibble
    """

    e6m2: np.ndarray
    e1_8: np.ndarray
    e1_16: np.ndarray
    s1p2: np.ndarray
    orig_shape: tuple[int, ...]


def _require_group(x: np.ndarray, group: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0 or x.shape[-1] % group != 0:
        raise ValueError(
            f"last axis length {x.shape[-1] if x.ndim else 0} must be a multiple of {group}"
        )
    return x


def _groups(x: np.ndarray, group: int) -> np.ndarray:
    x = _require_group(x, group)
    return x.reshape(x.shape[:-1] + (x.shape[-1] // group, group))


def quantize_hif4(x: np.ndarray) -> HiF4Tensor:
    """BF16→HiF4 Algorithm 1 along the last axis (groups of 64)."""
    x = _require_group(np.asarray(x), HIF4_GROUP)
    orig_shape = tuple(x.shape)
    # Algorithm 1 consumes BF16
    x_bf = to_bf16(x).astype(np.float32)
    g = _groups(x_bf, HIF4_GROUP)  # (..., G, 64)
    prefix = g.shape[:-1]  # (..., G)

    abs_g = np.abs(g)
    v16 = abs_g.reshape(prefix + (16, 4)).max(axis=-1)  # (..., G, 16)
    v8 = v16.reshape(prefix + (8, 2)).max(axis=-1)  # (..., G, 8)
    vmax = v8.max(axis=-1)  # (..., G)

    nan_group = np.isnan(vmax)
    sf = to_bf16(vmax.astype(np.float32) * np.float32(_ONE_SEVENTH_BF16))
    e6m2 = encode_e6m2(sf.astype(np.float64))
    e6m2 = np.where(nan_group, np.uint8(E6M2_NAN_CODE), e6m2).astype(np.uint8)

    rec = e6m2_reciprocal_bf16(e6m2)[..., None]  # (..., G, 1)

    v8_scaled = to_bf16(v8.astype(np.float32) * rec)
    e1_8 = (v8_scaled >= np.float32(E1_8_THRESHOLD)).astype(np.uint8)
    e1_8 = np.where(nan_group[..., None], np.uint8(0), e1_8).astype(np.uint8)

    e8_for_v16 = np.repeat(e1_8, 2, axis=-1)  # (..., G, 16)
    half_e8 = np.where(e8_for_v16.astype(bool), np.float32(0.5), np.float32(1.0))
    v16_scaled = to_bf16(to_bf16(v16.astype(np.float32) * rec) * half_e8)
    e1_16 = (v16_scaled >= np.float32(E1_16_THRESHOLD)).astype(np.uint8)
    e1_16 = np.where(nan_group[..., None], np.uint8(0), e1_16).astype(np.uint8)

    e8_el = np.repeat(e1_8, 8, axis=-1)
    e16_el = np.repeat(e1_16, 4, axis=-1)
    inv_pow = np.where(e8_el.astype(bool), np.float32(0.5), np.float32(1.0)) * np.where(
        e16_el.astype(bool), np.float32(0.5), np.float32(1.0)
    )
    scaled = to_bf16(to_bf16(g.astype(np.float32) * rec) * inv_pow)
    scaled = np.where(np.isfinite(scaled), scaled, np.float32(0.0))
    s1p2 = encode_s1p2(scaled.astype(np.float64))
    s1p2 = np.where(nan_group[..., None], np.uint8(0), s1p2).astype(np.uint8)

    return HiF4Tensor(
        e6m2=e6m2,
        e1_8=e1_8,
        e1_16=e1_16,
        s1p2=s1p2,
        orig_shape=orig_shape,
    )


def dequantize_hif4(enc: HiF4Tensor) -> np.ndarray:
    """Eq. 2: V_i = E6M2 * 2^(E1_8 + E1_16) * S1P2; NaN E6M2 broadcasts to the group."""
    e6 = decode_e6m2(enc.e6m2)[..., None]  # (..., G, 1)
    e8 = np.repeat(enc.e1_8.astype(np.float64), 8, axis=-1)
    e16 = np.repeat(enc.e1_16.astype(np.float64), 4, axis=-1)
    s = decode_s1p2(enc.s1p2)
    pow2 = np.ldexp(1.0, (e8 + e16).astype(np.int32))
    out = e6 * pow2 * s
    nan = np.isnan(e6)
    out = np.where(nan, np.nan, out)
    return out.reshape(enc.orig_shape)


def fakequant_hif4(x: np.ndarray) -> np.ndarray:
    """Quantize then dequantize (cpu-ref)."""
    return dequantize_hif4(quantize_hif4(x))
