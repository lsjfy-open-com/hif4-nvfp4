"""Local Mini-Challenge LLM eval harness (cpu-ref / cuda-sim fake-quant).

ICME 2026 registration is closed. This package runs the Mini-Challenge *protocol*
on linear (+ optional QK^T) using Algorithm 1 / TE formulas already in
``hif4_nvfp4``. Embedding, lm_head, softmax, and PV stay high precision.
"""

from hif4_nvfp4.eval.fakequant import (
    FORMATS,
    QUANTIZER_TAG,
    fakequant_numpy,
    group_size,
    pad_to_group,
)
from hif4_nvfp4.eval.harness import run_eval

__all__ = [
    "FORMATS",
    "QUANTIZER_TAG",
    "fakequant_numpy",
    "group_size",
    "pad_to_group",
    "run_eval",
]
