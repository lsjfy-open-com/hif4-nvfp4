"""Orchestrate canary PPL then Mini-Challenge / lm-eval subset. No invented numbers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from hif4_nvfp4.eval.config import load_config
from hif4_nvfp4.eval.fakequant import FORMATS, QUANTIZER_TAG, resolve_device
from hif4_nvfp4.eval.lm_eval_run import TaskResult, run_lm_eval
from hif4_nvfp4.eval.model import QKScoreQuantContext, load_model
from hif4_nvfp4.eval.ppl import PPLResult, smoke_word_ppl, wikitext2_word_ppl

SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")


@dataclass
class EvalReport:
    protocol: str
    icme_registration: str
    device: str
    quantizer: str
    gemm: str
    fmt: str
    quant: str
    scope: str
    quantize_qk: bool
    model_id: str
    model_kind: str
    wrapped_linears: list[str]
    high_precision: list[str]
    canary: list[dict]
    tasks: list[dict]
    note: str
    timestamp: str
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "protocol": self.protocol,
            "icme_registration": self.icme_registration,
            "device": self.device,
            "quantizer": self.quantizer,
            "gemm": self.gemm,
            "format": self.fmt,
            "quant": self.quant,
            "scope": self.scope,
            "quantize_qk": self.quantize_qk,
            "model": self.model_id,
            "model_kind": self.model_kind,
            "wrapped_linears": self.wrapped_linears,
            "high_precision": self.high_precision,
            "canary": self.canary,
            "tasks": self.tasks,
            "note": self.note,
            "timestamp": self.timestamp,
        }
        if self.error:
            d["error"] = self.error
        if self.extra:
            d["extra"] = self.extra
        return d


def _scope(quantize_qk: bool) -> str:
    return "+attention-score" if quantize_qk else "linear-only"


def run_eval(
    *,
    model: str | None = "smoke",
    fmt: str = "hif4",
    quant: str = "W+A",
    device_tag: str = "cpu-ref",
    quantize_qk: bool = False,
    config_path: str | None = None,
    skip_lm_eval: bool = False,
    include_w4a4: bool = True,
    limit: int | None = None,
    max_words: int | None = None,
    max_seq: int | None = None,
) -> EvalReport:
    cfg = load_config(config_path)
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {fmt!r}")
    if quant not in ("W+A", "W-only"):
        raise ValueError(f"quant must be W+A or W-only, got {quant!r}")

    tag, torch_device = resolve_device(device_tag)
    gemm = "cuda-fp32" if tag == "cuda-sim" else "cpu-fp32"
    ts = datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S %Z")
    high_prec = list(cfg.get("high_precision") or ["embedding", "lm_head", "softmax", "pv"])
    note = (
        "IEEE ICME 2026 is closed; this is a local Mini-Challenge LLM protocol run. "
        "HiFloat4 / ICME-Demo / VBench are linked, not cloned. "
        "No scores are invented: skipped tasks carry a reason."
    )

    loaded = load_model(
        model,
        fmt=fmt,
        quant=quant,
        device=torch_device,
        quantize_qk=quantize_qk,
    )

    if loaded.error:
        skip_reason = loaded.error
        canary = [
            PPLResult(
                name="wikitext2_word_ppl",
                status="skipped",
                reason=skip_reason,
            ).to_dict(),
            PPLResult(
                name="smoke_word_ppl",
                status="skipped",
                reason="requested model failed to load; rerun with --model smoke.",
            ).to_dict(),
        ]
        w4a4 = cfg.get("mini_challenge_w4a4") or []
        subset = cfg.get("lm_eval_subset") or []
        tasks = [
            TaskResult(name=_name(e), status="skipped", reason=skip_reason, group=g).to_dict()
            for g, bucket in (("lm_eval_subset", subset), ("mini_challenge_w4a4", w4a4))
            for e in bucket
        ]
        return EvalReport(
            protocol=str(cfg.get("protocol", "mini-challenge-local")),
            icme_registration=str(cfg.get("icme_registration", "closed")),
            device=tag,
            quantizer=QUANTIZER_TAG,
            gemm=gemm,
            fmt=fmt,
            quant=quant,
            scope=_scope(quantize_qk),
            quantize_qk=quantize_qk,
            model_id=loaded.model_id,
            model_kind=loaded.kind,
            wrapped_linears=[],
            high_precision=high_prec,
            canary=canary,
            tasks=tasks,
            note=note,
            timestamp=ts,
            error=skip_reason,
        )

    seq = max_seq
    if seq is None:
        seq = int(getattr(getattr(loaded.model, "config", None), "max_seq", 64) or 64)
        if loaded.kind == "hf":
            seq = min(int(getattr(loaded.model.config, "max_position_embeddings", 2048)), 2048)

    canary_results: list[PPLResult] = []
    with QKScoreQuantContext(fmt, quantize_qk):
        # Eval plan: WikiText-2 word PPL first (may skip on smoke / missing data).
        canary_results.append(
            wikitext2_word_ppl(
                loaded.model,
                loaded.tokenizer,
                device=torch_device,
                max_seq=seq,
                max_words=max_words,
                model_kind=loaded.kind,
            )
        )
        if loaded.kind == "smoke":
            canary_results.append(
                smoke_word_ppl(
                    loaded.model,
                    loaded.tokenizer,
                    device=torch_device,
                    max_seq=seq,
                )
            )

        task_rows: list[TaskResult] = []
        subset = cfg.get("lm_eval_subset") or []
        w4a4 = cfg.get("mini_challenge_w4a4") or []
        if skip_lm_eval:
            reason = "lm-eval skipped by --skip-lm-eval"
            task_rows.extend(
                TaskResult(name=_name(e), status="skipped", reason=reason, group="lm_eval_subset")
                for e in subset
            )
            if include_w4a4:
                task_rows.extend(
                    TaskResult(
                        name=_name(e),
                        status="skipped",
                        reason=reason,
                        group="mini_challenge_w4a4",
                    )
                    for e in w4a4
                )
        else:
            task_rows.extend(
                run_lm_eval(
                    subset,
                    group="lm_eval_subset",
                    wrapped_model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    model_kind=loaded.kind,
                    device_tag=tag,
                    limit=limit,
                )
            )
            if include_w4a4:
                task_rows.extend(
                    run_lm_eval(
                        w4a4,
                        group="mini_challenge_w4a4",
                        wrapped_model=loaded.model,
                        tokenizer=loaded.tokenizer,
                        model_kind=loaded.kind,
                        device_tag=tag,
                        limit=limit,
                    )
                )

    return EvalReport(
        protocol=str(cfg.get("protocol", "mini-challenge-local")),
        icme_registration=str(cfg.get("icme_registration", "closed")),
        device=tag,
        quantizer=QUANTIZER_TAG,
        gemm=gemm,
        fmt=fmt,
        quant=quant,
        scope=_scope(quantize_qk),
        quantize_qk=quantize_qk,
        model_id=loaded.model_id,
        model_kind=loaded.kind,
        wrapped_linears=list(loaded.wrapped_linears),
        high_precision=high_prec,
        canary=[c.to_dict() for c in canary_results],
        tasks=[t.to_dict() for t in task_rows],
        note=note,
        timestamp=ts,
        extra={"n_wrapped_linears": len(loaded.wrapped_linears)},
    )


def _name(entry: dict | str) -> str:
    if isinstance(entry, str):
        return entry
    return str(entry.get("name", ""))


def format_report(report: EvalReport) -> str:
    d = report.to_dict()
    lines = [
        f"device: {d['device']}",
        f"quantizer: {d['quantizer']} (Algorithm 1 / TE; not Cube/Tensor-Core accumulate)",
        f"gemm: {d['gemm']}",
        f"format: {d['format']}  quant: {d['quant']}  scope: {d['scope']}",
        f"quantize_qk: {d['quantize_qk']}  (PV / softmax / embedding / lm_head stay high precision)",
        f"model: {d['model']} ({d['model_kind']})",
        f"wrapped_linears: {len(d['wrapped_linears'])}",
        f"protocol: {d['protocol']}  icme_registration: {d['icme_registration']}",
        f"timestamp: {d['timestamp']}",
        "",
        d["note"],
        "",
        "== canary (WikiText-2 first) ==",
    ]
    for c in d["canary"]:
        lines.append(_fmt_row(c))
    lines.append("")
    lines.append("== tasks ==")
    for t in d["tasks"]:
        lines.append(_fmt_row(t))
    if d.get("error"):
        lines.append("")
        lines.append(f"ERROR: {d['error']}")
    return "\n".join(lines)


def _fmt_row(row: dict) -> str:
    st = row.get("status", "?")
    name = row.get("name", "")
    extra = ""
    if st == "ok" and "word_ppl" in row:
        extra = f" word_ppl={row['word_ppl']:.6g} n_words={row.get('n_words')} corpus={row.get('corpus')}"
    elif st == "ok" and row.get("metrics"):
        extra = f" metrics={row['metrics']}"
    elif row.get("reason"):
        extra = f" reason={row['reason']}"
    group = row.get("group")
    prefix = f"[{group}] " if group else ""
    return f"{prefix}{name}: {st}{extra}"


def dump_json(report: EvalReport, path: str | None) -> str:
    text = json.dumps(report.to_dict(), indent=2, sort_keys=False)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    return text
