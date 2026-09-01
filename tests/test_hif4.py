"""Algorithm 1 thresholds, NaN broadcast, and HiF4 dequant (Eq. 2)."""

import numpy as np

from hif4_nvfp4.constants import E6M2_NAN_CODE
from hif4_nvfp4.formats import decode_e6m2, decode_s1p2, e6m2_reciprocal_bf16, to_bf16
from hif4_nvfp4.hif4 import dequantize_hif4, fakequant_hif4, quantize_hif4


def test_e6m2_nan_broadcasts_to_all_64():
    v = np.zeros(64, dtype=np.float32)
    v[3] = np.nan
    y = fakequant_hif4(v)
    assert y.shape == (64,)
    assert np.all(np.isnan(y))


def test_e6m2_nan_code_broadcasts_on_dequant():
    enc = quantize_hif4(np.ones(64, dtype=np.float32))
    enc.e6m2[...] = E6M2_NAN_CODE
    y = dequantize_hif4(enc)
    assert np.all(np.isnan(y))


def _v8(v: np.ndarray) -> np.ndarray:
    return np.abs(v).reshape(8, 8).max(axis=-1)


def _v16(v: np.ndarray) -> np.ndarray:
    return np.abs(v).reshape(16, 4).max(axis=-1)


def test_algorithm1_e1_8_threshold_4():
    """E1_8 = (V8 * E6M2^{-1} >= 4) ? 1 : 0."""
    v = np.zeros(64, dtype=np.float32)
    v[0] = 3.0  # V8[0] = 3
    v[8] = 7.0  # V8[1] = 7 = Vmax
    enc = quantize_hif4(v)
    rec = float(e6m2_reciprocal_bf16(enc.e6m2)[0])
    v8 = _v8(v)
    expected = (to_bf16(v8.astype(np.float32) * np.float32(rec)) >= 4.0).astype(np.uint8)
    np.testing.assert_array_equal(enc.e1_8[0], expected)
    assert enc.e1_8[0, 0] == 0
    assert enc.e1_8[0, 1] == 1


def test_algorithm1_e1_16_threshold_2():
    """E1_16[i] = (V16[i] * E6M2^{-1} * 2^{-E1_8[⌈i/2⌉]} >= 2) ? 1 : 0."""
    v = np.zeros(64, dtype=np.float32)
    v[0] = 1.0  # V16[0]=1.0, V8[0] will be 2.5
    v[4] = 2.5  # V16[1]=2.5 → e8=0, 2.5>=2 → e16=1
    v[8] = 7.0  # global peak
    v[16] = 3.5  # V16[4]=3.5
    v[20] = 5.0  # V16[5]=5.0, V8[2]=5 → e8=1
    enc = quantize_hif4(v)
    rec = float(e6m2_reciprocal_bf16(enc.e6m2)[0])
    v16 = _v16(v)
    e8_for = np.repeat(enc.e1_8[0], 2)
    half = np.where(e8_for.astype(bool), np.float32(0.5), np.float32(1.0))
    scaled = to_bf16(to_bf16(v16.astype(np.float32) * np.float32(rec)) * half)
    expected = (scaled >= 2.0).astype(np.uint8)
    np.testing.assert_array_equal(enc.e1_16[0], expected)
    assert enc.e1_8[0, 0] == 0
    assert enc.e1_16[0, 0] == 0
    assert enc.e1_16[0, 1] == 1
    assert enc.e1_8[0, 2] == 1
    assert enc.e1_16[0, 4] == 0
    assert enc.e1_16[0, 5] == 1


def test_eq2_dequant_matches_metadata():
    v = np.linspace(-3.0, 5.0, 64).astype(np.float32)
    enc = quantize_hif4(v)
    y = dequantize_hif4(enc)
    e6 = float(decode_e6m2(enc.e6m2)[0])
    s = decode_s1p2(enc.s1p2[0])
    for i in range(64):
        e8 = int(enc.e1_8[0, i // 8])
        e16 = int(enc.e1_16[0, i // 4])
        expect = e6 * (2.0 ** (e8 + e16)) * s[i]
        np.testing.assert_allclose(y[i], expect, rtol=1e-6, atol=1e-12)


def test_constant_seven_is_exact():
    v = np.full(64, 7.0, dtype=np.float32)
    y = fakequant_hif4(v)
    np.testing.assert_allclose(y, 7.0, rtol=1e-5, atol=1e-5)


def test_all_zeros():
    v = np.zeros(64, dtype=np.float32)
    y = fakequant_hif4(v)
    np.testing.assert_allclose(y, 0.0, atol=1e-20)
