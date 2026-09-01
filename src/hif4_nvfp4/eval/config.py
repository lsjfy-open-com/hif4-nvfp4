"""Load configs/mini_challenge.yaml (with a builtin fallback)."""

from __future__ import annotations

from pathlib import Path

BUILTIN = {
    "protocol": "mini-challenge-local",
    "icme_registration": "closed",
    "device_tags": ["cpu-ref", "cuda-sim"],
    "defaults": {
        "model": "smoke",
        "format": "hif4",
        "quant": "W+A",
        "scope": "linear-only",
        "quantize_qk": False,
    },
    "high_precision": ["embedding", "lm_head", "softmax", "pv"],
    "canary": {
        "name": "wikitext2_word_ppl",
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "metric": "word_ppl",
        "run_first": True,
    },
    "lm_eval_subset": [
        {"name": "hellaswag", "lm_eval": ["hellaswag"]},
        {"name": "arc_easy", "lm_eval": ["arc_easy"]},
        {"name": "arc_challenge", "lm_eval": ["arc_challenge"]},
        {"name": "piqa", "lm_eval": ["piqa"]},
        {"name": "mmlu", "lm_eval": ["mmlu"]},
    ],
    "mini_challenge_w4a4": [
        {"name": "SuperGPQA", "lm_eval": ["supergpqa", "super_gpqa"]},
        {"name": "IFEval", "lm_eval": ["ifeval", "leaderboard_ifeval"]},
        {"name": "AIME2025", "lm_eval": ["aime2025", "aime_2025", "leaderboard_math_aime2025"]},
        {"name": "LiveCodeBench", "lm_eval": ["livecodebench", "livecodebench_v6"]},
        {"name": "BFCL", "lm_eval": ["bfcl", "bfcl_v3", "berkeley_function_call_leaderboard"]},
    ],
    "external_repos": {
        "HiFloat4": "https://github.com/global-computing-consortium/HiFloat4",
        "ICME-Demo": "https://github.com/global-computing-consortium/ICME-Demo",
        "VBench": "https://github.com/Vchitect/VBench",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    return repo_root() / "configs" / "mini_challenge.yaml"


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else default_config_path()
    if not p.is_file():
        return dict(BUILTIN)
    try:
        import yaml
    except ImportError:
        return dict(BUILTIN)
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config {p} is not a mapping")
    return data
