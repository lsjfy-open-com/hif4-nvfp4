"""Last-axis fake-quant using Algorithm 1 / TE formulas (eval plan §0, §5).

Quantization always uses the numpy cpu-ref kernels. ``cuda-sim`` only moves the
subsequent GEMM to CUDA; it does not invent a Tensor-Core 4-bit path.
"""

from __future__ import annotations

import numpy as np

from hif4_nvfp4.constants import HIF4_GROUP, NVFP4_GROUP
from hif4_nvfp4.hif4 import fakequant_hif4
from hif4_nvfp4.nvfp4 import fakequant_nvfp4

FORMATS = ("hif4", "nvfp4_pts", "bf16")
QUANTIZER_TAG = "cpu-ref"  # formulas; distinct from run-level device tag
DEVICE_TAGS = ("cpu-ref", "cuda-sim")


def group_size(fmt: str) -> int | None:
    if fmt == "hif4":
        return HIF4_GROUP
    if fmt == "nvfp4_pts":
        return NVFP4_GROUP
    if fmt == "bf16":
        return None
    raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")


def pad_to_group(x: np.ndarray, group: int) -> tuple[np.ndarray, int]:
    """Pad the last axis with zeros so its length is a multiple of ``group``."""
    x = np.asarray(x)
    if x.ndim == 0:
        raise ValueError("scalar has no last axis to pad")
    k = int(x.shape[-1])
    rem = k % group
    if rem == 0:
        return x, 0
    pad = group - rem
    pads = [(0, 0)] * (x.ndim - 1) + [(0, pad)]
    return np.pad(x, pads, mode="constant"), pad


def fakequant_numpy(x: np.ndarray, fmt: str) -> np.ndarray:
    """Fake-quant along the last axis (GEMM K / head dim)."""
    if fmt == "bf16":
        return np.asarray(x)
    group = group_size(fmt)
    assert group is not None
    padded, extra = pad_to_group(np.asarray(x), group)
    if fmt == "hif4":
        y = fakequant_hif4(padded)
    elif fmt == "nvfp4_pts":
        y = fakequant_nvfp4(padded, use_pts=True)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    if extra:
        y = y[..., :-extra]
    return y


def fakequant_torch(t: object, fmt: str) -> object:
    """Torch bridge: cpu-ref numpy quantize, restore device/dtype.

    Imported lazily so ``import hif4_nvfp4`` stays numpy-only.
    """
    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(t)}")
    if fmt == "bf16":
        return t
    x = t.detach().to(dtype=torch.float32).cpu().numpy()
    y = fakequant_numpy(x, fmt)
    out = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    return out.to(device=t.device, dtype=t.dtype)


def resolve_device(tag: str) -> tuple[str, object]:
    """Return ``(device_tag, torch.device)``. Never relabel a missing GPU as cpu-ref."""
    import torch

    if tag not in DEVICE_TAGS:
        raise ValueError(f"device tag must be one of {DEVICE_TAGS}, got {tag!r}")
    if tag == "cpu-ref":
        return tag, torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "device=cuda-sim requested but torch.cuda.is_available() is False; "
            "not falling back (eval plan forbids unlabeled / fake device tags)"
        )
    return tag, torch.device("cuda")
