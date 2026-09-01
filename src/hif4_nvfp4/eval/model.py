"""Tiny smoke Transformer + linear-layer fake-quant wrap.

Linear GEMM is quantized along K (last dim of activation and of ``nn.Linear.weight``).
Embedding, lm_head, softmax, and the PV matmul stay high precision
(eval plan §2 / formal model §6). ``v_proj`` is a linear, so it *is* quantized;
the ``P @ V`` product is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SKIP_LINEAR_NAMES = ("lm_head",)
SKIP_LINEAR_SUBSTR = ("embed", "wte", "wpe")


def _quant_flags(fmt: str, quant: str) -> tuple[bool, bool]:
    """Return (quantize_weights, quantize_activations)."""
    if fmt == "bf16":
        return False, False
    if quant == "W-only":
        return True, False
    if quant == "W+A":
        return True, True
    raise ValueError(f"quant must be W+A or W-only, got {quant!r}")


@dataclass
class TinyConfig:
    vocab_size: int = 128
    n_layer: int = 2
    n_head: int = 2
    d_model: int = 64
    d_ff: int = 128
    max_seq: int = 64
    seed: int = 0


def _skip_linear(qualified_name: str) -> bool:
    n = qualified_name.lower()
    if n in SKIP_LINEAR_NAMES or n.endswith(".lm_head") or n == "lm_head":
        return True
    return any(s in n for s in SKIP_LINEAR_SUBSTR)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "torch is required for the Mini-Challenge harness: pip install -e '.[eval]'"
        ) from e
    return torch, nn, F


def make_fakequant_linear():
    torch, nn, F = _require_torch()

    class _FakeQuantLinear(nn.Module):
        def __init__(self, linear: nn.Linear, fmt: str, quant_w: bool, quant_a: bool):
            super().__init__()
            from hif4_nvfp4.eval.fakequant import fakequant_torch

            self.fmt = fmt
            self.quant_w = quant_w
            self.quant_a = quant_a
            self.in_features = linear.in_features
            self.out_features = linear.out_features
            w = linear.weight.detach().contiguous()
            if quant_w and fmt != "bf16":
                w = fakequant_torch(w, fmt)
            self.register_buffer("weight_q", w)
            self.register_buffer("weight_orig", linear.weight.detach().contiguous().clone())
            if linear.bias is not None:
                self.register_buffer("bias_q", linear.bias.detach().contiguous())
            else:
                self.bias_q = None

        def forward(self, x):
            from hif4_nvfp4.eval.fakequant import fakequant_torch

            a = fakequant_torch(x, self.fmt) if (self.quant_a and self.fmt != "bf16") else x
            return F.linear(a, self.weight_q, self.bias_q)

    return _FakeQuantLinear


def wrap_linears(model, fmt: str, *, quant_w: bool, quant_a: bool) -> list[str]:
    """Replace ``nn.Linear`` with fake-quant wrappers. Returns wrapped module names."""
    torch, nn, _F = _require_torch()
    FakeQ = make_fakequant_linear()
    names: list[str] = []
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if type(mod).__name__ == "_FakeQuantLinear":
            continue
        if _skip_linear(name):
            continue
        names.append(name)

    def _set(root, qualified: str, new_mod):
        parts = qualified.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

    def _get(root, qualified: str):
        obj = root
        for p in qualified.split("."):
            obj = getattr(obj, p)
        return obj

    wrapped: list[str] = []
    for name in names:
        old = _get(model, name)
        _set(model, name, FakeQ(old, fmt, quant_w, quant_a))
        wrapped.append(name)
    return wrapped


def set_quantize_qk(model, enabled: bool, fmt: str) -> int:
    n = 0
    for m in model.modules():
        if hasattr(m, "quantize_qk") and hasattr(m, "q_proj"):
            m.quantize_qk = bool(enabled)
            m.qk_scheme = fmt
            n += 1
    return n


class QKScoreQuantContext:
    """Patch SDPA so Q/K are fake-quantized and V/softmax are not (HF Llama path)."""

    def __init__(self, fmt: str, enabled: bool):
        self.fmt = fmt
        self.enabled = enabled
        self._orig = None

    def __enter__(self):
        if not self.enabled or self.fmt == "bf16":
            return self
        torch, _nn, F = _require_torch()
        from hif4_nvfp4.eval.fakequant import fakequant_torch

        self._orig = F.scaled_dot_product_attention
        orig = self._orig
        fmt = self.fmt

        def _wrapped(query, key, value, *args, **kwargs):
            q = fakequant_torch(query, fmt)
            k = fakequant_torch(key, fmt)
            return orig(q, k, value, *args, **kwargs)

        F.scaled_dot_product_attention = _wrapped  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            _torch, _nn, F = _require_torch()
            F.scaled_dot_product_attention = self._orig  # type: ignore[method-assign]
            self._orig = None
        return False


def build_tiny_lm(cfg: TinyConfig | None = None):
    torch, nn, F = _require_torch()
    cfg = cfg or TinyConfig()
    torch.manual_seed(cfg.seed)

    class TinyAttention(nn.Module):
        def __init__(self, c: TinyConfig):
            super().__init__()
            if c.d_model % c.n_head != 0:
                raise ValueError("d_model must be divisible by n_head")
            self.n_head = c.n_head
            self.head_dim = c.d_model // c.n_head
            self.q_proj = nn.Linear(c.d_model, c.d_model, bias=False)
            self.k_proj = nn.Linear(c.d_model, c.d_model, bias=False)
            self.v_proj = nn.Linear(c.d_model, c.d_model, bias=False)
            self.o_proj = nn.Linear(c.d_model, c.d_model, bias=False)
            self.quantize_qk = False
            self.qk_scheme = "hif4"

        def forward(self, x):
            from hif4_nvfp4.eval.fakequant import fakequant_torch

            b, t, _c = x.shape
            q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
            if self.quantize_qk:
                q = fakequant_torch(q, self.qk_scheme)
                k = fakequant_torch(k, self.qk_scheme)
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(causal, float("-inf"))
            p = torch.softmax(scores, dim=-1)  # high precision
            y = torch.matmul(p, v)  # PV high precision
            y = y.transpose(1, 2).contiguous().view(b, t, x.size(-1))
            return self.o_proj(y)

    class TinyMLP(nn.Module):
        def __init__(self, c: TinyConfig):
            super().__init__()
            self.fc1 = nn.Linear(c.d_model, c.d_ff)
            self.fc2 = nn.Linear(c.d_ff, c.d_model)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))

    class TinyBlock(nn.Module):
        def __init__(self, c: TinyConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(c.d_model)
            self.attn = TinyAttention(c)
            self.ln2 = nn.LayerNorm(c.d_model)
            self.mlp = TinyMLP(c)

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class TinyCausalLM(nn.Module):
        def __init__(self, c: TinyConfig):
            super().__init__()
            self.config = c
            self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
            self.pos_embed = nn.Embedding(c.max_seq, c.d_model)
            self.layers = nn.ModuleList([TinyBlock(c) for _ in range(c.n_layer)])
            self.ln_f = nn.LayerNorm(c.d_model)
            self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

        def forward(self, input_ids):
            b, t = input_ids.shape
            if t > self.config.max_seq:
                raise ValueError(f"sequence length {t} > max_seq {self.config.max_seq}")
            pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
            x = self.embed_tokens(input_ids) + self.pos_embed(pos)
            for layer in self.layers:
                x = layer(x)
            x = self.ln_f(x)
            return self.lm_head(x)

    return TinyCausalLM(cfg)


@dataclass
class LoadedModel:
    kind: str  # smoke | hf
    model: object
    tokenizer: object
    model_id: str
    wrapped_linears: list[str] = field(default_factory=list)
    error: str | None = None


LLAMA2_7B_ID = "meta-llama/Llama-2-7b-hf"
LLAMA2_ALIASES = {"llama2-7b", "llama-2-7b", "Llama-2-7B"}


def resolve_model_id(spec: str | None) -> str:
    import os

    raw = spec or os.environ.get("HIF4_EVAL_MODEL") or "smoke"
    raw = raw.strip()
    if raw.lower() in {"smoke", "tiny", "random"}:
        return "smoke"
    if raw in LLAMA2_ALIASES or raw.lower() in {a.lower() for a in LLAMA2_ALIASES}:
        return LLAMA2_7B_ID
    return raw


class SmokeTokenizer:
    """Whitespace tokenizer hashing words into a tiny vocab. Not WikiText-capable."""

    def __init__(self, vocab_size: int = 128):
        import hashlib

        self.vocab_size = vocab_size
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.bos_id = 3
        self._hashlib = hashlib
        self.eos_token_id = self.eos_id

    def _wid(self, word: str) -> int:
        h = int(self._hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        span = self.vocab_size - 8
        return 8 + (h % span)

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_id]
        for w in text.split():
            ids.append(self._wid(w))
        ids.append(self.eos_id)
        return ids

    def __call__(self, text: str, return_tensors: str | None = None, **_kw):
        ids = self.encode(text)
        if return_tensors == "pt":
            torch, _nn, _F = _require_torch()
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": [ids]}


def load_model(
    spec: str | None,
    *,
    fmt: str,
    quant: str,
    device,
    quantize_qk: bool,
) -> LoadedModel:
    torch, _nn, _F = _require_torch()
    model_id = resolve_model_id(spec)
    quant_w, quant_a = _quant_flags(fmt, quant)

    if model_id == "smoke":
        cfg = TinyConfig()
        model = build_tiny_lm(cfg)
        wrapped = wrap_linears(model, fmt, quant_w=quant_w, quant_a=quant_a)
        set_quantize_qk(model, quantize_qk, fmt)
        model.to(device)
        model.eval()
        return LoadedModel(
            kind="smoke",
            model=model,
            tokenizer=SmokeTokenizer(cfg.vocab_size),
            model_id="smoke",
            wrapped_linears=wrapped,
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        return LoadedModel(
            kind="hf",
            model=None,
            tokenizer=None,
            model_id=model_id,
            error=(
                f"cannot load {model_id}: transformers is not installed "
                f"({e}). pip install -e '.[eval-full]'. Smoke model is still available "
                f"(--model smoke)."
            ),
        )

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"torch_dtype": torch.float32}
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id, attn_implementation="sdpa", **kwargs
            )
        except TypeError:
            hf_model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as e:  # download / auth / missing files
        return LoadedModel(
            kind="hf",
            model=None,
            tokenizer=None,
            model_id=model_id,
            error=(
                f"cannot download or load {model_id}: {type(e).__name__}: {e}. "
                "Not inventing scores. Use --model smoke for the tiny random Transformer."
            ),
        )

    wrapped = wrap_linears(hf_model, fmt, quant_w=quant_w, quant_a=quant_a)
    set_quantize_qk(hf_model, quantize_qk, fmt)
    hf_model.to(device)
    hf_model.eval()
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return LoadedModel(
        kind="hf",
        model=hf_model,
        tokenizer=tok,
        model_id=model_id,
        wrapped_linears=wrapped,
    )
