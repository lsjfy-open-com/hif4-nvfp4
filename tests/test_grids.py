"""Encode/decode grids from docs/00-formal-model.md."""

import numpy as np
import pytest

from hif4_nvfp4.constants import (
    E2M1_POS_GRID,
    E6M2_NAN_CODE,
    S1P2_POS_GRID,
)
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


@pytest.mark.parametrize("v", E2M1_POS_GRID)
def test_e2m1_positive_roundtrip(v):
    code = encode_e2m1(np.array([v]))
    got = decode_e2m1(code)
    np.testing.assert_allclose(got, [v], atol=0, rtol=0)


@pytest.mark.parametrize("v", E2M1_POS_GRID)
def test_e2m1_negative_roundtrip(v):
    if v == 0.0:
        code = encode_e2m1(np.array([-0.0]))
        got = decode_e2m1(code)
        assert got[0] == 0.0 or got[0] == -0.0
        return
    code = encode_e2m1(np.array([-v]))
    got = decode_e2m1(code)
    np.testing.assert_allclose(got, [-v], atol=0, rtol=0)


def test_e2m1_codes_are_nibble():
    for i, v in enumerate(E2M1_POS_GRID):
        code = int(encode_e2m1(np.array([v]))[0])
        assert (code & 7) == i
        assert (code >> 3) == 0


@pytest.mark.parametrize("v", S1P2_POS_GRID)
def test_s1p2_positive_roundtrip(v):
    code = encode_s1p2(np.array([v]))
    got = decode_s1p2(code)
    np.testing.assert_allclose(got, [v], atol=0, rtol=0)


@pytest.mark.parametrize("v", S1P2_POS_GRID)
def test_s1p2_negative_roundtrip(v):
    code = encode_s1p2(np.array([-v]))
    got = decode_s1p2(code)
    np.testing.assert_allclose(got, [-v], atol=0, rtol=0)


def test_s1p2_zero_encodes_s000():
    pos = int(encode_s1p2(np.array([0.0]))[0])
    assert (pos & 7) == 0


def test_s1p2_clamp_preserves_sign():
    hi = decode_s1p2(encode_s1p2(np.array([10.0, -10.0])))
    np.testing.assert_allclose(hi, [1.75, -1.75])


def test_e6m2_min_max_nan_encodings():
    # 000000_00 = 2^{-48} * 1.00 ; 111111_10 = 2^{15} * 1.50 ; 111111_11 = NaN
    np.testing.assert_allclose(decode_e6m2(np.uint8(0b00000000)), np.ldexp(1.0, -48))
    np.testing.assert_allclose(decode_e6m2(np.uint8(0b11111110)), np.ldexp(1.5, 15))
    assert np.isnan(decode_e6m2(np.uint8(E6M2_NAN_CODE)))
    assert int(encode_e6m2(np.array([np.nan]))[0]) == E6M2_NAN_CODE


def test_e6m2_no_zero():
    code = encode_e6m2(np.array([0.0]))
    val = decode_e6m2(code)
    assert val[0] == np.ldexp(1.0, -48)


def test_e6m2_one_roundtrip():
    code = encode_e6m2(np.array([1.0]))
    np.testing.assert_allclose(decode_e6m2(code), [1.0])


def test_e4m3_max_is_448():
    np.testing.assert_allclose(decode_e4m3(encode_e4m3(np.array([448.0]))), [448.0])
    np.testing.assert_allclose(decode_e4m3(encode_e4m3(np.array([1000.0]))), [448.0])


def test_e4m3_zero_and_one():
    np.testing.assert_allclose(decode_e4m3(encode_e4m3(np.array([0.0]))), [0.0])
    np.testing.assert_allclose(decode_e4m3(encode_e4m3(np.array([1.0]))), [1.0])
