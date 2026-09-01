#!/usr/bin/env python3
"""Local Mini-Challenge LLM eval (ICME 2026 is closed — do not register)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hif4_nvfp4.eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
