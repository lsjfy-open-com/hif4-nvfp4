"""H2 packing: four NVFP4-16 blocks into one HiF4-64 group (formal model §4.5).

Exact embedding is not generally possible (E4M3 is not a power of two; 4-tuple
S1P2 micro-exponents are shared). This path is best-effort: NVFP4-quantize,
then run Algorithm 1 on the NVFP4 reconstruction so Cube sees a legal HiF4 unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hif4_nvfp4.hif4 import HiF4Tensor, dequantize_hif4, quantize_hif4
from hif4_nvfp4.nvfp4 import NVFP4Tensor, dequantize_nvfp4, quantize_nvfp4


@dataclass
class H2PackResult:
    hif4: HiF4Tensor
    nvfp4: NVFP4Tensor
    x_h2: np.ndarray
    x_nv: np.ndarray
    mse_vs_original: float
    mse_vs_nvfp4: float
    exact_hit_fraction: float
    n_exact: int
    n_total: int


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    finite = np.isfinite(d)
    if not finite.any():
        return float("nan")
    return float(np.mean(d[finite] ** 2))


def exact_hit_mask(x_h2: np.ndarray, x_nv: np.ndarray) -> np.ndarray:
    """Elementwise reconstruction equality in float64 (not metadata bit-identity)."""
    a = np.asarray(x_h2, dtype=np.float64)
    b = np.asarray(x_nv, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)
    close = np.abs(a - b) <= (1e-9 * scale + 1e-15)
    return both_nan | close


def pack_h2(x: np.ndarray, *, use_pts: bool = True) -> H2PackResult:
    """Constructive H2 (§4.5): NVFP4 recipe → Algorithm 1 on the decoded targets."""
    x = np.asarray(x, dtype=np.float64)
    nv = quantize_nvfp4(x, use_pts=use_pts)
    x_nv = dequantize_nvfp4(nv)
    hif4 = quantize_hif4(x_nv)
    x_h2 = dequantize_hif4(hif4)
    hits = exact_hit_mask(x_h2, x_nv)
    n_total = int(hits.size)
    n_exact = int(np.count_nonzero(hits))
    return H2PackResult(
        hif4=hif4,
        nvfp4=nv,
        x_h2=x_h2,
        x_nv=x_nv,
        mse_vs_original=_mse(x_h2, x),
        mse_vs_nvfp4=_mse(x_h2, x_nv),
        exact_hit_fraction=(n_exact / n_total) if n_total else float("nan"),
        n_exact=n_exact,
        n_total=n_total,
    )
