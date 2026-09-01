"""Gaussian 1024×1024 MSE harness (eval plan §1). Device tag: cpu-ref.

Literature ratio HiF4:NVFP4:MXFP4 = 1:1.32:1.89 is an *external* check, not an
oracle hardcoded here. This script prints absolute MSE and ratios vs HiF4.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from hif4_nvfp4.constants import DEVICE_TAG
from hif4_nvfp4.hif4 import fakequant_hif4
from hif4_nvfp4.nvfp4 import fakequant_nvfp4
from hif4_nvfp4.pack import pack_h2


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    finite = np.isfinite(d)
    if not finite.any():
        return float("nan")
    return float(np.mean(d[finite] ** 2))


def gaussian_matrix(size: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=sigma, size=(size, size)).astype(np.float64)


def run_protocol(
    *,
    size: int = 1024,
    seed: int = 0,
    x_values: range | list[int] | None = None,
    with_h2: bool = True,
) -> list[dict]:
    """σ = 0.01 * 2^x for x in 0..17 (eval plan §1 / formal model §2.4)."""
    if x_values is None:
        x_values = range(18)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for xv in x_values:
        sigma = 0.01 * (2.0**xv)
        mat = gaussian_matrix(size, sigma, rng)
        y_h = fakequant_hif4(mat)
        y_n = fakequant_nvfp4(mat, use_pts=False)
        y_p = fakequant_nvfp4(mat, use_pts=True)
        mse_h = mse(y_h, mat)
        mse_n = mse(y_n, mat)
        mse_p = mse(y_p, mat)
        row: dict = {
            "x": int(xv),
            "sigma": sigma,
            "mse_hif4": mse_h,
            "mse_nvfp4": mse_n,
            "mse_nvfp4_pts": mse_p,
            "ratio_nvfp4": (mse_n / mse_h) if mse_h > 0 else float("nan"),
            "ratio_nvfp4_pts": (mse_p / mse_h) if mse_h > 0 else float("nan"),
        }
        if with_h2:
            packed = pack_h2(mat, use_pts=True)
            row["h2_mse_orig"] = packed.mse_vs_original
            row["h2_mse_nv"] = packed.mse_vs_nvfp4
            row["h2_hit"] = packed.exact_hit_fraction
        rows.append(row)
    return rows


def format_table(rows: list[dict], *, with_h2: bool) -> str:
    lines = [
        f"device: {DEVICE_TAG}",
        "protocol: Gaussian 1024×1024 MSE (eval plan §1); σ = 0.01·2^x, x=0..17",
        "MXFP4: skipped",
        "",
    ]
    if with_h2:
        hdr = (
            f"{'x':>3} {'sigma':>12} {'HiF4':>12} {'NVFP4':>12} {'NVFP4+PTS':>12} "
            f"{'r_NV':>8} {'r_PTS':>8} {'H2 vs orig':>12} {'H2 vs NV':>12} {'H2 hit':>8}"
        )
    else:
        hdr = (
            f"{'x':>3} {'sigma':>12} {'HiF4':>12} {'NVFP4':>12} {'NVFP4+PTS':>12} "
            f"{'r_NV':>8} {'r_PTS':>8}"
        )
    lines.append(hdr)
    for r in rows:
        base = (
            f"{r['x']:3d} {r['sigma']:12.6g} {r['mse_hif4']:12.6e} {r['mse_nvfp4']:12.6e} "
            f"{r['mse_nvfp4_pts']:12.6e} {r['ratio_nvfp4']:8.3f} {r['ratio_nvfp4_pts']:8.3f}"
        )
        if with_h2:
            base += (
                f" {r['h2_mse_orig']:12.6e} {r['h2_mse_nv']:12.6e} {r['h2_hit']:8.4f}"
            )
        lines.append(base)
    lines.append("")
    lines.append(
        "ratios vs HiF4 are mse(format)/mse(HiF4). Literature 1:1.32:1.89 is an "
        "external check (arXiv:2602.11287), not an oracle."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="cpu-ref Gaussian MSE for HiF4 / NVFP4 / H2")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-h2", action="store_true")
    args = p.parse_args(argv)
    rows = run_protocol(size=args.size, seed=args.seed, with_h2=not args.no_h2)
    print(format_table(rows, with_h2=not args.no_h2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
