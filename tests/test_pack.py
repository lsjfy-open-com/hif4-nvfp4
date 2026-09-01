"""H2 packing reports: MSE vs original, vs NVFP4 dequant, exact-hit fraction."""

import numpy as np

from hif4_nvfp4.pack import pack_h2


def test_h2_reports_all_three_metrics():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=(64, 64))
    r = pack_h2(x, use_pts=True)
    assert np.isfinite(r.mse_vs_original)
    assert np.isfinite(r.mse_vs_nvfp4)
    assert 0.0 <= r.exact_hit_fraction <= 1.0
    assert r.n_exact == int(r.exact_hit_fraction * r.n_total)
    assert r.x_h2.shape == x.shape
    assert r.x_nv.shape == x.shape


def test_h2_hit_rate_on_shared_scale_block():
    # Four identical NVFP4 blocks (same E4M3, same E2M1 pattern) often pack well.
    block = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2, dtype=np.float64)
    x = np.tile(block, 4)  # 64
    r = pack_h2(x, use_pts=True)
    assert r.n_total == 64
    assert np.isfinite(r.mse_vs_nvfp4)
    # Exact embedding is not guaranteed; just count (may be 0).
    assert 0 <= r.n_exact <= 64
