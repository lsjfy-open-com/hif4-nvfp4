"""WikiText-2 word PPL canary and a synthetic smoke corpus (eval plan §2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPLResult:
    name: str
    status: str  # ok | skipped | error
    word_ppl: float | None = None
    n_words: int | None = None
    n_tokens: int | None = None
    nll_nats: float | None = None
    corpus: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "status": self.status}
        if self.word_ppl is not None:
            d["word_ppl"] = self.word_ppl
        if self.n_words is not None:
            d["n_words"] = self.n_words
        if self.n_tokens is not None:
            d["n_tokens"] = self.n_tokens
        if self.nll_nats is not None:
            d["nll_nats"] = self.nll_nats
        if self.corpus is not None:
            d["corpus"] = self.corpus
        if self.reason is not None:
            d["reason"] = self.reason
        return d


SMOKE_CORPUS = (
    "the cat sat on the mat and the dog ran in the park "
    "where a bird sang a song about the sun and the rain"
)


def word_count(text: str) -> int:
    return len(text.split())


def _nll_from_logits(logits, labels, ignore_index: int = -100) -> tuple[float, int]:
    import torch.nn.functional as F

    # logits: [B, T, V], labels: [B, T]
    logp = F.log_softmax(logits.float(), dim=-1)
    valid = labels != ignore_index
    if not valid.any():
        return float("nan"), 0
    gathered = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    nll = float((-gathered[valid]).sum().item())
    ntok = int(valid.sum().item())
    return nll, ntok


def _forward_logits(model, input_ids):
    out = model(input_ids)
    if hasattr(out, "logits"):
        return out.logits
    return out


def word_ppl_from_text(model, tokenizer, text: str, *, device, max_seq: int) -> tuple[float, int, int, float]:
    """NLL over next-token preds; divide by whitespace word count (WikiText word PPL)."""
    import math

    import torch

    n_words = word_count(text)
    if n_words == 0:
        raise ValueError("empty corpus")

    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    if hasattr(ids, "ids"):
        ids = list(ids)

    if len(ids) < 2:
        raise ValueError(f"need at least 2 tokens, got {len(ids)}")

    total_nll = 0.0
    total_tok = 0
    # chunk to max_seq with teacher forcing
    stride = max(1, max_seq - 1)
    i = 0
    model.eval()
    with torch.no_grad():
        while i < len(ids) - 1:
            chunk = ids[i : i + max_seq]
            if len(chunk) < 2:
                break
            t = torch.tensor([chunk], dtype=torch.long, device=device)
            logits = _forward_logits(model, t)
            labels = t[:, 1:]
            pred = logits[:, :-1, :]
            nll, ntok = _nll_from_logits(pred, labels)
            total_nll += nll
            total_tok += ntok
            if i + max_seq >= len(ids):
                break
            i += stride
    if total_tok == 0 or n_words == 0:
        raise ValueError("no scored tokens")
    ppl = math.exp(total_nll / n_words)
    return ppl, n_words, total_tok, total_nll


def smoke_word_ppl(model, tokenizer, *, device, max_seq: int) -> PPLResult:
    try:
        ppl, nw, nt, nll = word_ppl_from_text(
            model, tokenizer, SMOKE_CORPUS, device=device, max_seq=max_seq
        )
    except Exception as e:
        return PPLResult(
            name="smoke_word_ppl",
            status="error",
            corpus="synthetic-smoke",
            reason=f"{type(e).__name__}: {e}",
        )
    return PPLResult(
        name="smoke_word_ppl",
        status="ok",
        word_ppl=ppl,
        n_words=nw,
        n_tokens=nt,
        nll_nats=nll,
        corpus="synthetic-smoke",
    )


def wikitext2_word_ppl(
    model,
    tokenizer,
    *,
    device,
    max_seq: int,
    max_words: int | None,
    model_kind: str,
) -> PPLResult:
    name = "wikitext2_word_ppl"
    if model_kind == "smoke":
        return PPLResult(
            name=name,
            status="skipped",
            reason=(
                "smoke tokenizer is not WikiText-capable; synthetic canary is "
                "smoke_word_ppl. Pass --model llama2-7b (opt-in) for the eval-plan canary."
            ),
        )
    try:
        from datasets import load_dataset
    except ImportError as e:
        return PPLResult(
            name=name,
            status="skipped",
            reason=f"datasets not installed ({e}). pip install -e '.[eval-full]'.",
        )
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(t for t in ds["text"] if t)
    except Exception as e:
        return PPLResult(
            name=name,
            status="skipped",
            reason=f"WikiText-2 download failed: {type(e).__name__}: {e}",
        )
    if max_words is not None:
        words = text.split()
        text = " ".join(words[: max(1, max_words)])
    try:
        ppl, nw, nt, nll = word_ppl_from_text(
            model, tokenizer, text, device=device, max_seq=max_seq
        )
    except Exception as e:
        return PPLResult(
            name=name,
            status="error",
            corpus="wikitext-2-raw-v1",
            reason=f"{type(e).__name__}: {e}",
        )
    return PPLResult(
        name=name,
        status="ok",
        word_ppl=ppl,
        n_words=nw,
        n_tokens=nt,
        nll_nats=nll,
        corpus="wikitext-2-raw-v1" + (f":first-{max_words}-words" if max_words else ""),
    )
