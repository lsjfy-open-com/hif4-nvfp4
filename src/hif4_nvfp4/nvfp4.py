"""NVFP4 quantize / dequantize (TE formula, ``docs/00-formal-model.md`` §1.1).

With PTS: ``s_global = global_amax / (448 * 6)`` (peak-to-2688).
Without PTS: ``s_global = 1`` (direct-cast per-block E4M3 so the block peak maps toward 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hif4_nvfp4.constants import E2M1_MAX, NVFP4_GROUP, PTS_PEAK
from hif4_nvfp4.formats import decode_e2m1, decode_e4m3, encode_e2m1, encode_e4m3


@dataclass
class NVFP4Tensor:
    """NVFP4 along the last axis (groups of 16).

    Shapes (``prefix`` is ``x.shape[:-1]``):
      e2m1     uint8    prefix + (n_blocks, 16)
      s_block  uint8    prefix + (n_blocks,)   E4M3 codes
      s_global float64  scalar (one per tensor, matching TE)
    """

    e2m1: np.ndarray
    s_block: np.ndarray
    s_global: float
    orig_shape: tuple[int, ...]
    use_pts: bool


def _require_group(x: np.ndarray, group: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0 or x.shape[-1] % group != 0:
        raise ValueError(
            f"last axis length {x.shape[-1] if x.ndim else 0} must be a multiple of {group}"
        )
    return x


def quantize_nvfp4(x: np.ndarray, *, use_pts: bool = True) -> NVFP4Tensor:
    """TE NVFP4 along the last axis.

    ``use_pts=True``: ``s_global = amax / (448*6)`` then
    ``s_block = E4M3((block_amax / 6) / s_global)``.

    ``use_pts=False``: ``s_global = 1``, ``s_block = E4M3(block_amax / 6)``.
    """
    x = _require_group(np.asarray(x, dtype=np.float64), NVFP4_GROUP)
    orig_shape = tuple(x.shape)
    blocks = x.reshape(x.shape[:-1] + (x.shape[-1] // NVFP4_GROUP, NVFP4_GROUP))

    finite = np.isfinite(x)
    if finite.any():
        global_amax = float(np.max(np.abs(x[finite])))
    else:
        global_amax = 0.0

    if use_pts:
        if global_amax == 0.0 or not np.isfinite(global_amax):
            s_global = 1.0
        else:
            s_global = global_amax / PTS_PEAK
    else:
        s_global = 1.0

    block_amax = np.max(np.abs(blocks), axis=-1)
    # TE: s_block = (block_amax / 6) / s_global
    raw_scale = (block_amax / E2M1_MAX) / s_global
    raw_scale = np.where(np.isfinite(raw_scale), raw_scale, 0.0)
    s_block = encode_e4m3(raw_scale)

    s_b = decode_e4m3(s_block)[..., None]
    denom = s_b * s_global
    scaled = np.divide(blocks, denom, out=np.zeros_like(blocks), where=denom != 0.0)
    e2m1 = encode_e2m1(scaled)

    return NVFP4Tensor(
        e2m1=e2m1,
        s_block=s_block,
        s_global=float(s_global),
        orig_shape=orig_shape,
        use_pts=use_pts,
    )


def dequantize_nvfp4(enc: NVFP4Tensor) -> np.ndarray:
    """x = E2M1 * s_block * s_global."""
    elem = decode_e2m1(enc.e2m1)
    scale = decode_e4m3(enc.s_block)[..., None]
    out = elem * scale * enc.s_global
    return out.reshape(enc.orig_shape)


def fakequant_nvfp4(x: np.ndarray, *, use_pts: bool = True) -> np.ndarray:
    """Quantize then dequantize (cpu-ref)."""
    return dequantize_nvfp4(quantize_nvfp4(x, use_pts=use_pts))
