"""Word-PPL formula: exp(sum NLL / n_words), not a literature table."""

import math

import pytest

torch = pytest.importorskip("torch")

from hif4_nvfp4.eval.ppl import SMOKE_CORPUS, word_count, word_ppl_from_text
from hif4_nvfp4.eval.model import TinyConfig, build_tiny_lm, SmokeTokenizer


def test_word_count_smoke_corpus():
    assert word_count(SMOKE_CORPUS) == len(SMOKE_CORPUS.split())
    assert word_count(SMOKE_CORPUS) > 10


def test_word_ppl_matches_exp_nll_over_words():
    model = build_tiny_lm(TinyConfig(seed=3))
    tok = SmokeTokenizer(128)
    model.eval()
    ppl, n_words, n_tokens, nll = word_ppl_from_text(
        model, tok, SMOKE_CORPUS, device=torch.device("cpu"), max_seq=64
    )
    assert n_words == word_count(SMOKE_CORPUS)
    assert n_tokens > 0
    assert math.isfinite(ppl)
    assert abs(ppl - math.exp(nll / n_words)) < 1e-6
