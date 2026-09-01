"""CLI for the local Mini-Challenge LLM eval harness."""

from __future__ import annotations

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Local Mini-Challenge LLM protocol (ICME 2026 registration is closed). "
            "cpu-ref HiF4 / NVFP4+PTS fake-quant on linear layers; WikiText-2 canary first."
        )
    )
    p.add_argument(
        "--model",
        default=os.environ.get("HIF4_EVAL_MODEL", "smoke"),
        help="smoke (default) | llama2-7b | HuggingFace id. Llama-2-7B is opt-in.",
    )
    p.add_argument(
        "--format",
        dest="fmt",
        choices=["hif4", "nvfp4_pts", "bf16"],
        default="hif4",
        help="hif4 = Algorithm 1; nvfp4_pts = TE + peak-to-2688; bf16 = no fake-quant",
    )
    p.add_argument("--quant", choices=["W+A", "W-only"], default="W+A")
    p.add_argument(
        "--device",
        dest="device_tag",
        choices=["cpu-ref", "cuda-sim"],
        default=os.environ.get("HIF4_EVAL_DEVICE", "cpu-ref"),
    )
    p.add_argument(
        "--quantize-qk",
        action="store_true",
        help="Quantize QK^T inputs (eval-plan +attention-score). PV stays high precision. Default off.",
    )
    p.add_argument("--config", default=None, help="YAML config (default configs/mini_challenge.yaml)")
    p.add_argument("--skip-lm-eval", action="store_true")
    p.add_argument("--no-w4a4", action="store_true", help="Do not even list Mini-Challenge W4A4 names")
    p.add_argument("--limit", type=int, default=None, help="lm-eval few-shot example cap")
    p.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Truncate WikiText-2 to N words (not the full eval-plan canary if set)",
    )
    p.add_argument("--max-seq", type=int, default=None)
    p.add_argument("--json", dest="json_path", default=None, help="Write the report JSON to this path")
    p.add_argument("--json-stdout", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from hif4_nvfp4.eval.harness import dump_json, format_report, run_eval
    except ImportError as e:
        print(
            "failed to import eval harness (install torch: pip install -e '.[eval]'): "
            f"{e}",
            file=sys.stderr,
        )
        return 2

    try:
        report = run_eval(
            model=args.model,
            fmt=args.fmt,
            quant=args.quant,
            device_tag=args.device_tag,
            quantize_qk=args.quantize_qk,
            config_path=args.config,
            skip_lm_eval=args.skip_lm_eval,
            include_w4a4=not args.no_w4a4,
            limit=args.limit,
            max_words=args.max_words,
            max_seq=args.max_seq,
        )
    except Exception as e:
        print(f"eval failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(format_report(report))
    js = dump_json(report, args.json_path)
    if args.json_stdout:
        print()
        print(js)
    if args.json_path:
        print(f"\nwrote {args.json_path}")
    return 0 if report.error is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
