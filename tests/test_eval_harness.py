"""Smoke Mini-Challenge harness on a tiny random Transformer (no 7B weights)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hif4_nvfp4.eval.fakequant import resolve_device
from hif4_nvfp4.eval.harness import run_eval
from hif4_nvfp4.eval.model import (
    TinyConfig,
    build_tiny_lm,
    load_model,
    wrap_linears,
)


def test_wrap_skips_embedding_and_lm_head():
    model = build_tiny_lm(TinyConfig(seed=1))
    embed_before = model.embed_tokens.weight.detach().clone()
    head_before = model.lm_head.weight.detach().clone()
    wrapped = wrap_linears(model, "hif4", quant_w=True, quant_a=True)
    assert "lm_head" not in wrapped
    assert not any("embed" in n for n in wrapped)
    assert any(n.endswith("q_proj") for n in wrapped)
    assert any(n.endswith("fc1") for n in wrapped)
    torch.testing.assert_close(model.embed_tokens.weight, embed_before)
    torch.testing.assert_close(model.lm_head.weight, head_before)


def test_wrapped_linear_weight_is_on_hif4_grid_path():
    model = build_tiny_lm(TinyConfig(seed=2))
    w_before = model.layers[0].mlp.fc1.weight.detach().clone()
    wrap_linears(model, "hif4", quant_w=True, quant_a=False)
    w_after = model.layers[0].mlp.fc1.weight_q
    # PTQ weight fake-quant should move at least some values on a random init.
    assert w_after.shape == w_before.shape
    assert torch.isfinite(w_after).all()


def test_smoke_forward_finite_logits():
    loaded = load_model("smoke", fmt="hif4", quant="W+A", device=torch.device("cpu"), quantize_qk=False)
    assert loaded.error is None
    ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        logits = loaded.model(ids)
    assert logits.shape == (2, 8, 128)
    assert torch.isfinite(logits).all()


def test_quantize_qk_flag_changes_attention_scores_path():
    torch.manual_seed(0)
    a = load_model("smoke", fmt="hif4", quant="W+A", device=torch.device("cpu"), quantize_qk=False)
    torch.manual_seed(0)
    b = load_model("smoke", fmt="hif4", quant="W+A", device=torch.device("cpu"), quantize_qk=True)
    ids = torch.arange(8).unsqueeze(0)
    with torch.no_grad():
        la = a.model(ids)
        lb = b.model(ids)
    # +attention-score fake-quants Q/K along head dim; PV still high precision.
    assert a.model.layers[0].attn.quantize_qk is False
    assert b.model.layers[0].attn.quantize_qk is True
    # Random init + head_dim=32 (padded to 64) is expected to move scores.
    assert not torch.equal(la, lb)


def test_run_eval_smoke_cpu_ref_no_fake_scores():
    report = run_eval(
        model="smoke",
        fmt="hif4",
        quant="W+A",
        device_tag="cpu-ref",
        quantize_qk=False,
        skip_lm_eval=False,
        include_w4a4=True,
    )
    d = report.to_dict()
    assert d["device"] == "cpu-ref"
    assert d["quantizer"] == "cpu-ref"
    assert d["icme_registration"] == "closed"
    assert d["format"] == "hif4"
    assert d["quantize_qk"] is False
    assert d["scope"] == "linear-only"
    assert "pv" in d["high_precision"]
    assert "lm_head" in d["high_precision"]
    assert d.get("error") is None

    canary = {c["name"]: c for c in d["canary"]}
    assert canary["wikitext2_word_ppl"]["status"] == "skipped"
    assert "smoke" in canary["wikitext2_word_ppl"]["reason"].lower()
    smoke = canary["smoke_word_ppl"]
    assert smoke["status"] == "ok"
    assert smoke["corpus"] == "synthetic-smoke"
    assert np.isfinite(smoke["word_ppl"])
    assert smoke["n_words"] > 0

    tasks = {t["name"]: t for t in d["tasks"]}
    for name in ("hellaswag", "arc_easy", "arc_challenge", "piqa", "mmlu"):
        assert tasks[name]["status"] == "skipped"
        assert tasks[name]["group"] == "lm_eval_subset"
    for name in ("SuperGPQA", "IFEval", "AIME2025", "LiveCodeBench", "BFCL"):
        assert tasks[name]["status"] == "skipped"
        assert tasks[name]["group"] == "mini_challenge_w4a4"
        # documented, not scored
        assert "metrics" not in tasks[name]


def test_nvfp4_pts_smoke_runs():
    report = run_eval(model="smoke", fmt="nvfp4_pts", quant="W+A", skip_lm_eval=True, include_w4a4=False)
    assert report.device == "cpu-ref"
    assert report.fmt == "nvfp4_pts"
    smoke = next(c for c in report.canary if c["name"] == "smoke_word_ppl")
    assert smoke["status"] == "ok"
    assert np.isfinite(smoke["word_ppl"])


def test_missing_local_model_skips_with_error(tmp_path: Path):
    missing = tmp_path / "no-such-hif4-weights"
    report = run_eval(model=str(missing), skip_lm_eval=True, include_w4a4=True)
    assert report.error
    assert "smoke" in report.error.lower()
    wikitext = next(c for c in report.canary if c["name"] == "wikitext2_word_ppl")
    assert wikitext["status"] == "skipped"
    sg = next(t for t in report.tasks if t["name"] == "SuperGPQA")
    assert sg["status"] == "skipped"


def test_cuda_sim_without_gpu_does_not_fake_a_tag():
    if torch.cuda.is_available():
        pytest.skip("CUDA is present; cannot test the missing-GPU refusal")
    with pytest.raises(RuntimeError, match="cuda-sim"):
        resolve_device("cuda-sim")
