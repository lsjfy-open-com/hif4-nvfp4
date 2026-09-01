"""Optional lm-eval runner. Missing extras/datasets → skip with a reason, never fake scores."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskResult:
    name: str
    status: str  # ok | skipped | error
    lm_eval_id: str | None = None
    metrics: dict | None = None
    reason: str | None = None
    group: str = "lm_eval_subset"

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "status": self.status, "group": self.group}
        if self.lm_eval_id is not None:
            d["lm_eval_id"] = self.lm_eval_id
        if self.metrics is not None:
            d["metrics"] = self.metrics
        if self.reason is not None:
            d["reason"] = self.reason
        return d


def _candidate_ids(entry: dict | str) -> tuple[str, list[str]]:
    if isinstance(entry, str):
        return entry, [entry]
    name = str(entry.get("name", ""))
    ids = entry.get("lm_eval") or [name]
    if isinstance(ids, str):
        ids = [ids]
    return name, [str(x) for x in ids]


def list_lm_eval_tasks() -> tuple[set[str] | None, str | None]:
    try:
        from lm_eval.tasks import TaskManager
    except ImportError as e:
        return None, f"lm-eval not installed ({e}). pip install -e '.[eval-full]'."
    try:
        tm = TaskManager()
        return set(tm.all_tasks), None
    except Exception as e:
        return None, f"cannot list lm-eval tasks: {type(e).__name__}: {e}"


def resolve_task(entry: dict | str, available: set[str] | None) -> tuple[str, str | None, str | None]:
    """Return (display_name, lm_eval_id or None, skip_reason or None)."""
    name, cands = _candidate_ids(entry)
    if available is None:
        return name, None, None  # caller already has a global skip
    for cid in cands:
        if cid in available:
            return name, cid, None
    return (
        name,
        None,
        (
            f"documented Mini-Challenge / subset name {name!r}; none of lm-eval "
            f"ids {cands} are registered in this install (dataset missing or "
            "task not shipped). Not inventing a score."
        ),
    )


def skip_all(entries: list, *, group: str, reason: str) -> list[TaskResult]:
    out = []
    for e in entries:
        name, _cands = _candidate_ids(e)
        out.append(TaskResult(name=name, status="skipped", reason=reason, group=group))
    return out


def run_lm_eval(
    entries: list,
    *,
    group: str,
    wrapped_model,
    tokenizer,
    model_kind: str,
    device_tag: str,
    limit: int | None,
    batch_size: int = 1,
) -> list[TaskResult]:
    if model_kind == "smoke":
        return skip_all(
            entries,
            group=group,
            reason=(
                "lm-eval subset is skipped for the randomly-initialized smoke model "
                "(scores would not be a Mini-Challenge result). Use --model llama2-7b."
            ),
        )
    if wrapped_model is None or tokenizer is None:
        return skip_all(
            entries,
            group=group,
            reason="no loaded Hugging Face model; cannot run lm-eval.",
        )

    available, err = list_lm_eval_tasks()
    if err:
        return skip_all(entries, group=group, reason=err)

    resolved: list[tuple[str, str]] = []
    results: list[TaskResult] = []
    for e in entries:
        name, tid, skip = resolve_task(e, available)
        if skip or tid is None:
            results.append(
                TaskResult(name=name, status="skipped", reason=skip, group=group)
            )
        else:
            resolved.append((name, tid))

    if not resolved:
        return results

    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        return skip_all(entries, group=group, reason=f"lm-eval import failed: {e}")

    # Drop the already-appended skips; re-build from resolved + prior skips
    skips = [r for r in results if r.status == "skipped"]
    device = "cuda" if device_tag == "cuda-sim" else "cpu"
    try:
        hflm = HFLM(pretrained=wrapped_model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    except Exception as e:
        reason = f"failed to wrap model for lm-eval: {type(e).__name__}: {e}"
        return skips + [
            TaskResult(name=n, status="skipped", lm_eval_id=tid, reason=reason, group=group)
            for n, tid in resolved
        ]

    task_ids = [tid for _n, tid in resolved]
    try:
        out = simple_evaluate(
            model=hflm,
            tasks=task_ids,
            limit=limit,
            device=device,
            batch_size=batch_size,
            verbosity="ERROR",
        )
    except Exception as e:
        reason = f"lm-eval failed: {type(e).__name__}: {e}. Not inventing scores."
        return skips + [
            TaskResult(name=n, status="error", lm_eval_id=tid, reason=reason, group=group)
            for n, tid in resolved
        ]

    results_table = (out or {}).get("results") or {}
    done = []
    for name, tid in resolved:
        metrics = results_table.get(tid)
        if not metrics:
            done.append(
                TaskResult(
                    name=name,
                    status="skipped",
                    lm_eval_id=tid,
                    reason=f"lm-eval returned no metrics for {tid}",
                    group=group,
                )
            )
        else:
            # Strip internal keys that are not measurements
            clean = {
                k: v
                for k, v in metrics.items()
                if not str(k).startswith("alias") and v is not None
            }
            done.append(
                TaskResult(
                    name=name,
                    status="ok",
                    lm_eval_id=tid,
                    metrics=clean,
                    group=group,
                )
            )
    return skips + done
