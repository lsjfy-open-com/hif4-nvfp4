"""Encode/decode for E2M1, E4M3, E6M2, S1P2, and BF16 rounding.

Grids and special values follow ``docs/00-formal-model.md`` only.
"""

from __future__ import annotations

import numpy as np

from hif4_nvfp4.constants import (
    E2M1_MAX,
    E2M1_POS_GRID,
    E4M3_BIAS,
    E4M3_MAX,
    E6M2_BIAS,
    E6M2_NAN_CODE,
    S1P2_POS_GRID,
)

E2M1_POS = np.asarray(E2M1_POS_GRID, dtype=np.float64)
S1P2_POS = np.asarray(S1P2_POS_GRID, dtype=np.float64)


def to_bf16(x: np.ndarray | float) -> np.ndarray:
    """Round to bfloat16 (round-half-to-even); return float32 with low 16 bits cleared."""
    arr = np.asarray(x, dtype=np.float32)
    bits = arr.view(np.uint32)
    exp = (bits >> np.uint32(23)) & np.uint32(0xFF)
    special = exp == np.uint32(0xFF)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + (np.uint32(0x7FFF) + lsb)
    out_bits = np.where(special, bits, rounded) & np.uint32(0xFFFF0000)
    return out_bits.view(np.float32)


def round_half_to_even(x: np.ndarray) -> np.ndarray:
    """IEEE round-half-to-even (numpy / Algorithm 1 default)."""
    return np.rint(x)


# ---------------------------------------------------------------------------
# E2M1 (NVFP4 4-bit element): 1 sign + 2 exp + 1 mantissa, max ±6
# Codes stored in the low nibble: S | EE | M
# ---------------------------------------------------------------------------


def encode_e2m1(x: np.ndarray | float) -> np.ndarray:
    """Round to the E2M1 grid; clamp to ±6. Ties break to even 3-bit magnitude code."""
    arr = np.asarray(x, dtype=np.float64)
    sign = np.signbit(arr)
    ax = np.abs(arr)
    ax = np.where(np.isnan(arr), 0.0, ax)
    ax = np.minimum(ax, E2M1_MAX)

    dist = np.abs(E2M1_POS - ax[..., None])
    min_d = dist.min(axis=-1, keepdims=True)
    tied = dist == min_d
    even = (np.arange(E2M1_POS.size) % 2) == 0
    even_tied = tied & even
    has_even_tie = even_tied.any(axis=-1)
    idx_nearest = dist.argmin(axis=-1)
    idx_even = even_tied.astype(np.uint8).argmax(axis=-1)
    multi = tied.sum(axis=-1) > 1
    idx = np.where(has_even_tie & multi, idx_even, idx_nearest).astype(np.uint8)

    code = (sign.astype(np.uint8) << np.uint8(3)) | idx.astype(np.uint8)
    return np.where(np.isnan(np.asarray(x, dtype=np.float64)), np.uint8(0), code).astype(
        np.uint8
    )


def decode_e2m1(code: np.ndarray | int) -> np.ndarray:
    code = np.asarray(code, dtype=np.uint8)
    sign = (code >> np.uint8(3)) & np.uint8(1)
    mag = code & np.uint8(7)
    val = E2M1_POS[mag]
    return np.where(sign.astype(bool), -val, val)


# ---------------------------------------------------------------------------
# S1P2 ≡ E1M2 (HiF4 4-bit element): sign-magnitude, max ±1.75, min nonzero ±0.25
# Codes in the low nibble: S | integer.fraction (3 magnitude bits in units of 0.25)
# ---------------------------------------------------------------------------


def encode_s1p2(x: np.ndarray | float) -> np.ndarray:
    """Round to the S1P2 grid (half-to-even on 0.25 ulp), then clamp to ±1.75."""
    arr = np.asarray(x, dtype=np.float64)
    nan = ~np.isfinite(arr)
    sign = np.signbit(np.where(nan, 0.0, arr))
    ax = np.abs(np.where(nan, 0.0, arr))
    units = round_half_to_even(ax / 0.25)
    units = np.clip(units, 0.0, 7.0)
    mag = units.astype(np.uint8)
    code = (sign.astype(np.uint8) << np.uint8(3)) | mag
    return np.where(nan, np.uint8(0), code).astype(np.uint8)


def decode_s1p2(code: np.ndarray | int) -> np.ndarray:
    code = np.asarray(code, dtype=np.uint8)
    sign = (code >> np.uint8(3)) & np.uint8(1)
    mag = (code & np.uint8(7)).astype(np.float64)
    val = mag * 0.25
    return np.where(sign.astype(bool), -val, val)


# ---------------------------------------------------------------------------
# E6M2 (HiF4 unsigned 8-bit scale): 6-bit exp bias 48, 2-bit mantissa, hidden 1
# Only normals. No Inf, no 0. NaN = 111111_11. X = 2^E * 1.M
# ---------------------------------------------------------------------------


def decode_e6m2(code: np.ndarray | int) -> np.ndarray:
    code = np.asarray(code, dtype=np.uint8)
    stored_e = (code >> np.uint8(2)).astype(np.int32)
    m = (code & np.uint8(3)).astype(np.int32)
    is_nan = (stored_e == 63) & (m == 3)
    unbiased = stored_e - E6M2_BIAS
    val = np.ldexp(1.0 + m.astype(np.float64) / 4.0, unbiased)
    return np.where(is_nan, np.nan, val)


def encode_e6m2(x: np.ndarray | float) -> np.ndarray:
    """Quantize a nonnegative scale to E6M2 (half-to-even). 0 → min (no zero)."""
    arr = np.asarray(x, dtype=np.float64)
    is_nan = np.isnan(arr)
    is_neg = arr < 0.0
    ax = np.where(is_neg | (arr == 0.0), 0.0, np.abs(arr))

    min_v = np.ldexp(1.0, -E6M2_BIAS)  # 2^{-48} * 1.00
    max_v = np.ldexp(1.5, 15)  # 2^{15} * 1.50

    too_small = (ax < min_v) | is_neg | (arr == 0.0)
    too_big = ax > max_v

    safe = np.where((ax == 0.0) | is_nan, 1.0, ax)
    mant, exp = np.frexp(safe)
    unbiased = exp.astype(np.int32) - 1
    sig = mant * 2.0
    frac = (sig - 1.0) * 4.0
    m = round_half_to_even(frac).astype(np.int32)

    overflow_m = m >= 4
    unbiased = np.where(overflow_m, unbiased + 1, unbiased)
    m = np.where(overflow_m, 0, m)

    at_max_exp = unbiased > 15
    nan_slot = (unbiased == 15) & (m > 2)
    clamp_max = too_big | at_max_exp | nan_slot
    unbiased = np.where(clamp_max, 15, unbiased)
    m = np.where(clamp_max, 2, m)

    unbiased = np.where(too_small, -E6M2_BIAS, unbiased)
    m = np.where(too_small, 0, m)

    stored_e = unbiased + E6M2_BIAS
    stored_e = np.clip(stored_e, 0, 63)
    code = ((stored_e.astype(np.int32) << 2) | m).astype(np.uint8)
    return np.where(is_nan, np.uint8(E6M2_NAN_CODE), code).astype(np.uint8)


def e6m2_reciprocal_bf16(code: np.ndarray | int) -> np.ndarray:
    """E6M2^{-1} as BF16 via the 4-entry mantissa LUT (Algorithm 1)."""
    code = np.asarray(code, dtype=np.uint8)
    stored_e = (code >> np.uint8(2)).astype(np.int32)
    m = (code & np.uint8(3)).astype(np.int32)
    is_nan = (stored_e == 63) & (m == 3)
    # 1 / 1.M for M in {00,01,10,11}
    lut = np.array([1.0 / 1.00, 1.0 / 1.25, 1.0 / 1.50, 1.0 / 1.75], dtype=np.float64)
    rec = np.ldexp(lut[m], -(stored_e - E6M2_BIAS))
    out = to_bf16(rec)
    nan32 = np.float32("nan")
    return np.where(is_nan, nan32, out).astype(np.float32)


# ---------------------------------------------------------------------------
# E4M3 (NVFP4 block scale): 1 sign + 4 exp + 3 mantissa, bias 7
# No Inf. NaN when exp=15 and mant=7. Max finite 448.
# ---------------------------------------------------------------------------


def decode_e4m3(code: np.ndarray | int) -> np.ndarray:
    code = np.asarray(code, dtype=np.uint8)
    sign = (code >> np.uint8(7)) & np.uint8(1)
    stored_e = (code >> np.uint8(3)) & np.uint8(0xF)
    m = (code & np.uint8(7)).astype(np.int32)
    is_nan = (stored_e == 15) & (m == 7)
    e = stored_e.astype(np.int32)
    sub = e == 0
    # subnormal: 2^{-6} * (m/8); normal: 2^{e-7} * (1 + m/8)
    val = np.empty(code.shape, dtype=np.float64)
    val = np.where(
        sub,
        np.ldexp(m.astype(np.float64) / 8.0, 1 - E4M3_BIAS),
        np.ldexp(1.0 + m.astype(np.float64) / 8.0, e - E4M3_BIAS),
    )
    val = np.where(is_nan, np.nan, val)
    return np.where(sign.astype(bool), -val, val)


def encode_e4m3(x: np.ndarray | float) -> np.ndarray:
    """Quantize to E4M3 (half-to-even). Saturate at ±448. NaN stays NaN."""
    arr = np.asarray(x, dtype=np.float64)
    is_nan = np.isnan(arr)
    sign = np.signbit(arr)
    ax = np.abs(arr)
    ax = np.where(is_nan, 0.0, ax)

    min_normal = np.ldexp(1.0, 1 - E4M3_BIAS)  # 2^{-6}
    max_v = E4M3_MAX
    sub_ulp = np.ldexp(1.0, 1 - E4M3_BIAS - 3)  # 2^{-9}

    too_big = ax > max_v
    is_zero = ax == 0.0
    is_sub = (~is_zero) & (ax < min_normal)

    sub_units = round_half_to_even(ax / sub_ulp)
    sub_m = np.clip(sub_units, 0.0, 8.0).astype(np.int32)
    sub_to_normal = is_sub & (sub_m >= 8)
    sub_m = np.where(sub_m >= 8, 0, sub_m)

    safe = np.where(is_zero | is_nan | (ax == 0.0), 1.0, ax)
    mant, exp = np.frexp(safe)
    unbiased = exp.astype(np.int32) - 1
    sig = mant * 2.0
    frac = (sig - 1.0) * 8.0
    m = round_half_to_even(frac).astype(np.int32)
    overflow_m = m >= 8
    unbiased = np.where(overflow_m, unbiased + 1, unbiased)
    m = np.where(overflow_m, 0, m)

    # exp=15, mant=7 is NaN — max finite is exp=15, mant=6 (448)
    clamp_max = too_big | (unbiased > 8) | ((unbiased == 8) & (m > 6))
    stored_e = unbiased + E4M3_BIAS
    stored_e = np.where(clamp_max, 15, stored_e)
    m = np.where(clamp_max, 6, m)

    stored_e = np.where(is_sub & ~sub_to_normal, 0, stored_e)
    m = np.where(is_sub & ~sub_to_normal, sub_m, m)
    stored_e = np.where(sub_to_normal, 1, stored_e)
    m = np.where(sub_to_normal, 0, m)

    stored_e = np.where(is_zero, 0, stored_e)
    m = np.where(is_zero, 0, m)

    stored_e = np.clip(stored_e, 0, 15)
    m = np.clip(m, 0, 7)
    code = (
        (sign.astype(np.uint8) << np.uint8(7))
        | (stored_e.astype(np.uint8) << np.uint8(3))
        | m.astype(np.uint8)
    )
    nan_code = np.uint8(0x7F)
    return np.where(is_nan, nan_code, code).astype(np.uint8)
