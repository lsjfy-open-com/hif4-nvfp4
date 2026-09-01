"""Gaussian MSE protocol checks (eval plan §1). Not an oracle for 1:1.32:1.89."""

import numpy as np
import pytest

from hif4_nvfp4.gaussian_mse import run_protocol


# Interior of NVFP4's E4M3×E2M1 box: σ = 0.01·2^x with x in [4, 12].
INTERIOR = list(range(4, 13))
# Sweep endpoints: underflow of E4M3 (x=0) and overflow of 448×6 (x=17).
EXTREMES = [0, 17]


@pytest.fixture(scope="module")
def mse_rows():
    # Smaller than 1024² so the unit test stays quick; same σ schedule and axes.
    return {
        r["x"]: r
        for r in run_protocol(size=128, seed=0, x_values=range(18), with_h2=True)
    }


def test_all_mse_finite(mse_rows):
    for x, r in mse_rows.items():
        assert np.isfinite(r["mse_hif4"]), f"HiF4 MSE not finite at x={x}"
        assert np.isfinite(r["mse_nvfp4"]), f"NVFP4 MSE not finite at x={x}"
        assert np.isfinite(r["mse_nvfp4_pts"]), f"NVFP4+PTS MSE not finite at x={x}"
        assert np.isfinite(r["h2_mse_orig"])
        assert np.isfinite(r["h2_mse_nv"])
        assert 0.0 <= r["h2_hit"] <= 1.0


def test_hif4_beats_nvfp4_on_interior_sigmas(mse_rows):
    for x in INTERIOR:
        r = mse_rows[x]
        assert r["mse_hif4"] < r["mse_nvfp4"], (
            f"x={x}: HiF4 MSE {r['mse_hif4']} >= NVFP4 {r['mse_nvfp4']}"
        )
        assert r["mse_hif4"] < r["mse_nvfp4_pts"], (
            f"x={x}: HiF4 MSE {r['mse_hif4']} >= NVFP4+PTS {r['mse_nvfp4_pts']}"
        )


def test_nvfp4_without_pts_blows_up_at_range_extremes(mse_rows):
    interior_nop_over_pts = np.median(
        [mse_rows[i]["mse_nvfp4"] / mse_rows[i]["mse_nvfp4_pts"] for i in INTERIOR]
    )
    for x in EXTREMES:
        r = mse_rows[x]
        nop = r["mse_nvfp4"]
        pts = r["mse_nvfp4_pts"]
        ratio_vs_pts = nop / pts if pts > 0 else np.inf
        # Interior: no-PTS ≈ PTS. Extremes: direct-cast error rises (under/overflow).
        assert ratio_vs_pts > 1.15 * interior_nop_over_pts, (
            f"x={x} σ={r['sigma']}: NVFP4 no-PTS did not blow up vs PTS "
            f"(mse={nop}, pts={pts}, ratio={ratio_vs_pts}, interior={interior_nop_over_pts})"
        )
        assert nop > pts


def test_h2_hit_is_a_fraction(mse_rows):
    # Exact embedding is not generally possible; any count in [0,1] is valid.
    hits = [mse_rows[x]["h2_hit"] for x in mse_rows]
    assert min(hits) >= 0.0
    assert max(hits) <= 1.0
