"""NVFP4 TE formula with and without PTS."""

import numpy as np

from hif4_nvfp4.constants import E4M3_MAX, PTS_PEAK
from hif4_nvfp4.nvfp4 import dequantize_nvfp4, fakequant_nvfp4, quantize_nvfp4


def test_e2m1_grid_with_unit_scale():
    # Values already on the E2M1 grid, amax=6 → s_block should map peak to 6.
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
    x = np.tile(grid, 2)  # 16
    y = fakequant_nvfp4(x, use_pts=True)
    np.testing.assert_allclose(y, x, atol=1e-6)


def test_pts_s_global_is_amax_over_2688():
    x = np.full(16, 2 * PTS_PEAK, dtype=np.float64)
    enc = quantize_nvfp4(x, use_pts=True)
    np.testing.assert_allclose(enc.s_global, 2.0)
    y = dequantize_nvfp4(enc)
    np.testing.assert_allclose(y, x, rtol=1e-5)


def test_without_pts_s_global_is_one():
    x = np.linspace(-3.0, 3.0, 32)
    enc = quantize_nvfp4(x, use_pts=False)
    assert enc.s_global == 1.0


def test_without_pts_saturates_e4m3_on_huge_amax():
    x = np.full(16, 1.0e6, dtype=np.float64)
    enc = quantize_nvfp4(x, use_pts=False)
    from hif4_nvfp4.formats import decode_e4m3

    np.testing.assert_allclose(decode_e4m3(enc.s_block), E4M3_MAX)
    y = dequantize_nvfp4(enc)
    # decode max = 448 * 6 = 2688, cannot represent 1e6
    np.testing.assert_allclose(y, PTS_PEAK, rtol=1e-5)
    assert y[0] < 1.0e6 / 10


def test_pts_covers_huge_amax():
    x = np.full(16, 1.0e6, dtype=np.float64)
    y = fakequant_nvfp4(x, use_pts=True)
    np.testing.assert_allclose(y, x, rtol=1e-5)
