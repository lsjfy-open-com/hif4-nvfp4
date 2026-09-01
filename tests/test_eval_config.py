"""YAML Mini-Challenge names stay documented even when datasets are missing."""

from hif4_nvfp4.eval.config import load_config
from hif4_nvfp4.eval.fakequant import DEVICE_TAGS, pad_to_group, fakequant_numpy
from hif4_nvfp4.eval.model import resolve_model_id

import numpy as np


def test_config_lists_w4a4_mini_challenge_names():
    cfg = load_config()
    names = [e["name"] for e in cfg["mini_challenge_w4a4"]]
    assert names == ["SuperGPQA", "IFEval", "AIME2025", "LiveCodeBench", "BFCL"]


def test_config_lm_eval_subset():
    cfg = load_config()
    names = [e["name"] for e in cfg["lm_eval_subset"]]
    assert names == ["hellaswag", "arc_easy", "arc_challenge", "piqa", "mmlu"]


def test_config_canary_is_wikitext2():
    cfg = load_config()
    assert cfg["canary"]["name"] == "wikitext2_word_ppl"
    assert cfg["canary"]["config"] == "wikitext-2-raw-v1"
    assert cfg["icme_registration"] == "closed"
    assert cfg["defaults"]["quantize_qk"] is False


def test_external_repos_are_links_not_paths():
    cfg = load_config()
    repos = cfg["external_repos"]
    for key in ("HiFloat4", "ICME-Demo", "VBench"):
        assert repos[key].startswith("https://")


def test_device_tags():
    assert DEVICE_TAGS == ("cpu-ref", "cuda-sim")


def test_llama2_is_opt_in_alias():
    import os

    os.environ.pop("HIF4_EVAL_MODEL", None)
    assert resolve_model_id("smoke") == "smoke"
    assert resolve_model_id("llama2-7b") == "meta-llama/Llama-2-7b-hf"
    assert resolve_model_id(None) == "smoke"


def test_pad_to_group_hif4():
    x = np.ones(32, dtype=np.float64)
    y, pad = pad_to_group(x, 64)
    assert pad == 32
    assert y.shape == (64,)
    np.testing.assert_allclose(y[:32], 1.0)
    np.testing.assert_allclose(y[32:], 0.0)


def test_fakequant_numpy_pads_and_unpads():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 32))
    y = fakequant_numpy(x, "hif4")
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    z = fakequant_numpy(x, "nvfp4_pts")
    assert z.shape == x.shape
    b = fakequant_numpy(x, "bf16")
    np.testing.assert_array_equal(b, x)
