# Qwen3 causal language model in pure MLX, for the ACE-Step 5Hz planner LM.
#
# The planner ships as three sizes (0.6B/1.7B/4B) that differ only in width and
# depth: same 217204-token vocabulary, same head_dim, same grouped-query layout,
# same rope_theta, all with tied embeddings. One implementation covers them.
#
# Nothing here is AceStep-specific -- this is stock Qwen3. What makes it the
# planner is the vocabulary, which carries 65535 ``<|audio_code_N|>`` tokens
# after the ordinary text ones; see :mod:`as15.planner`.
#
# There is no weight conversion step. The published checkpoints are bf16
# safetensors whose linear weights are already ``[out, in]`` -- MLX's own
# layout -- and whose key names are exactly the module tree below, so
# ``mx.load`` feeds ``load_weights`` directly. The DiT needs a converter
# because its patch convolutions have to be transposed and its ``decoder.*``
# subtree extracted; none of that applies here.

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import mlx.nn as nn


@dataclass(frozen=True)
class Qwen3Config:
    """The subset of a Qwen3 ``config.json`` this implementation reads."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def from_file(cls, path: Path) -> Qwen3Config:
        """Read *path*, rejecting anything this implementation would mis-run.

        The checks are not defensiveness for its own sake: every one of them is
        a field that, set otherwise, produces plausible tokens rather than an
        error. A sliding window would silently attend too little, and an
        untied head would leave the model reading logits off an embedding
        matrix that was never trained to produce them.
        """
        cfg = json.loads(path.read_text())

        model_type = cfg.get("model_type")
        if model_type != "qwen3":
            raise ValueError(
                f"{path} is a {model_type!r} model; this loader implements qwen3."
            )
        if cfg.get("use_sliding_window") or cfg.get("sliding_window"):
            raise ValueError(
                f"{path} asks for a sliding attention window, which this "
                f"implementation does not have; it would attend to too little "
                f"context and still emit tokens."
            )
        if not cfg.get("tie_word_embeddings", False):
            raise ValueError(
                f"{path} has an untied output head, which this implementation "
                f"does not load."
            )
        if cfg.get("attention_bias", False):
            raise ValueError(f"{path} asks for attention biases, which are not loaded.")

        hidden = cfg["hidden_size"]
        heads = cfg["num_attention_heads"]
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=hidden,
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=heads,
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim", hidden // heads),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=cfg.get("rope_theta", 1_000_000.0),
            tie_word_embeddings=True,
        )


class KVCache:
    """Per-layer key/value cache for one autoregressive decode.

    Grown in blocks rather than by concatenating each step: a concatenation
    reallocates and copies the whole cache per token, which on a 4B model over
    a thousand audio codes is most of the decode. The valid region is tracked
    separately from the allocation, so the buffer can run ahead of it.
    """

    STEP = 256

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Append *keys*/*values* and return the whole valid cache."""
        B, H, L, D = keys.shape
        start, end = self.offset, self.offset + L
        buf_k, buf_v = self.keys, self.values

        if buf_k is None or buf_v is None or end > buf_k.shape[2]:
            grow = ((end + self.STEP - 1) // self.STEP) * self.STEP
            fresh_k = mx.zeros((B, H, grow, D), keys.dtype)
            fresh_v = mx.zeros((B, H, grow, D), values.dtype)
            if buf_k is not None and buf_v is not None:
                fresh_k[..., :start, :] = buf_k[..., :start, :]
                fresh_v[..., :start, :] = buf_v[..., :start, :]
            buf_k, buf_v = fresh_k, fresh_v

        buf_k[..., start:end, :] = keys
        buf_v[..., start:end, :] = values
        self.keys, self.values, self.offset = buf_k, buf_v, end
        return buf_k[..., :end, :], buf_v[..., :end, :]


class Qwen3Attention(nn.Module):
    """Grouped-query attention with per-head QK-RMSNorm and rotary positions."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # `traditional=False` is the half-split rotation transformers calls
        # `rotate_half`, which is the convention Qwen3 was trained with; the
        # interleaved variant would rotate the wrong pairs of channels and
        # still produce fluent-looking text. Pinned by the parity test against
        # transformers' own Qwen3.
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(B, L, self.num_kv_heads, self.head_dim)

        # QK-norm is per head, over head_dim, and comes before RoPE.
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update(k, v)

        # K/V keep their own head count: MLX takes grouped-query attention
        # directly and asks that they not be tiled out to match Q.
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | None,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), mask, cache
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3Model(nn.Module):
    """The ``model.*`` subtree, named to match the published checkpoint."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        mask: mx.array | str | None,
        caches: list[KVCache] | None,
    ) -> mx.array:
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, None if caches is None else caches[i])
        return self.norm(h)


class MLXQwen3LM(nn.Module):
    """Qwen3 with a tied output head, over a whole prompt or one token at a time."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

    @classmethod
    def from_snapshot(cls, path: Path) -> MLXQwen3LM:
        """Build from a downloaded checkpoint directory and load its weights.

        The planners are published in two layouts. The ones with repos of their
        own carry ``model.embed_tokens.weight``; the 1.7B, which ships as a
        directory inside the shared base repo, carries a bare
        ``embed_tokens.weight``. Both are the same tensors under the same
        model, so the prefix is stripped here and the weights are loaded into
        the inner module rather than teaching the module tree to answer to two
        names.

        ``load_weights`` is strict either way, so a checkpoint this tree does
        not mirror exactly fails here rather than generating from
        partially-initialised weights.
        """
        from ..models import shard_files

        model = cls(Qwen3Config.from_file(path / "config.json"))

        weights: dict[str, mx.array] = {}
        for shard in shard_files(path):
            for key, value in mx.load(str(shard)).items():
                # The head is tied, so a checkpoint that carries one anyway is
                # carrying a copy of the embedding table; `Qwen3Config` has
                # already established the tie.
                if key == "lm_head.weight":
                    continue
                weights[key.removeprefix("model.")] = value

        model.model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        return model

    def __call__(self, input_ids: mx.array, caches: list[KVCache] | None) -> mx.array:
        """Logits for every position of *input_ids*, shaped [B, L, vocab].

        The mask is causal for a multi-token step and absent for a single-token
        one, where the lone query attends to the whole cache and a causal mask
        would be a no-op computed over the full cache width.
        """
        mask = "causal" if input_ids.shape[1] > 1 else None
        h = self.model(input_ids, mask, caches)
        return self.model.embed_tokens.as_linear(h)

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in range(self.config.num_hidden_layers)]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingParams:
    """How to draw the next token.

    Bounds are checked in :meth:`check` rather than trusted, because each of
    these silently means something else when it is out of range: a temperature
    of zero divides by zero, a ``top_p`` above 1 admits the whole distribution
    while reading like a setting, and a repetition penalty below 1 *rewards*
    repetition -- which on a run of a thousand audio codes is the difference
    between a song and a held note.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 disables
    repetition_penalty: float = 1.0
    repetition_window: int = 0  # 0 means every token generated so far

    def check(self) -> None:
        """Raise ValueError naming the field that cannot be honoured."""
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError(
                f"temperature must be finite and not negative, where 0 means "
                f"greedy; got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < 0:
            raise ValueError(f"top_k must not be negative, got {self.top_k}.")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be finite and above zero, where 1 "
                f"disables it; got {self.repetition_penalty}."
            )
        if self.repetition_window < 0:
            raise ValueError(
                f"repetition_window must not be negative, got {self.repetition_window}."
            )


def apply_repetition_penalty(
    logits: mx.array, recent: mx.array, penalty: float
) -> mx.array:
    """Divide the logits of already-drawn tokens, multiplying negative ones.

    The asymmetry is the convention the penalty was defined with (CTRL, and
    every implementation since): dividing a negative logit would raise it,
    which is the opposite of a penalty, so the sign decides the operation.
    """
    if penalty == 1.0 or recent.size == 0:
        return logits
    penalised = mx.array(logits)
    scores = penalised[0, recent]
    penalised[0, recent] = mx.where(scores < 0, scores * penalty, scores / penalty)
    return penalised


def _top_k_mask(logits: mx.array, top_k: int) -> mx.array:
    """Drop everything outside the *top_k* highest logits."""
    kth = mx.sort(logits, axis=-1)[..., -top_k, None]
    return mx.where(logits < kth, -mx.inf, logits)


def _top_p_mask(logits: mx.array, top_p: float) -> mx.array:
    """Drop the tail beyond cumulative probability *top_p*.

    The most probable token always survives, whatever *top_p* is: the mask
    keeps every position whose *exclusive* cumulative probability is below the
    threshold, so a single token carrying more than ``top_p`` of the mass is
    kept rather than leaving nothing to sample from.
    """
    order = mx.argsort(-logits, axis=-1)
    ordered = mx.take_along_axis(logits, order, axis=-1)
    probs = mx.softmax(ordered, axis=-1)
    exclusive = mx.cumsum(probs, axis=-1) - probs
    ordered = mx.where(exclusive < top_p, ordered, -mx.inf)
    # Back to vocabulary order, so the caller's ids still mean what they say.
    return mx.put_along_axis(mx.zeros_like(logits), order, ordered, axis=-1)


def sample_token(
    logits: mx.array, params: SamplingParams, key: mx.array | None
) -> mx.array:
    """Draw one token id from *logits* [1, vocab].

    A seeded draw takes an explicit key rather than going through MLX's
    implicit global PRNG, which any other ``mx.random`` call in the process --
    another generation, a caller's own draw -- would move, so the same seed
    would not reproduce the same plan. ``key=None`` is the deliberately
    unseeded draw, which has no reproducibility to preserve.
    """
    if params.temperature == 0.0:
        return mx.argmax(logits, axis=-1)

    logits = logits.astype(mx.float32) / params.temperature
    if params.top_k:
        logits = _top_k_mask(logits, min(params.top_k, logits.shape[-1]))
    if params.top_p < 1.0:
        logits = _top_p_mask(logits, params.top_p)
    return mx.random.categorical(logits, key=key)


# A hook the caller uses to constrain what may be drawn next. It is handed the
# logits for the step and the tokens drawn so far, and returns the logits to
# sample from -- masking with ``-inf`` is what makes a token unreachable.
LogitsProcessor = Callable[[mx.array, list[int]], mx.array]


def generate(
    model: MLXQwen3LM,
    prompt_ids: list[int],
    max_new_tokens: int,
    params: SamplingParams | None = None,
    seed: int | None = None,
    stop_ids: frozenset[int] = frozenset(),
    logits_processor: LogitsProcessor | None = None,
    uncond_prompt_ids: list[int] | None = None,
    guidance: float = 1.0,
) -> Iterator[int]:
    """Yield generated token ids, stopping at *max_new_tokens* or a stop id.

    A generator so the caller can show progress and stop early without the
    whole plan having to be drawn first; the stop token itself is not yielded.

    Passing *uncond_prompt_ids* turns on classifier-free guidance: the same
    continuation is tracked from a second, deliberately uninformative prompt,
    and each step samples from
    ``uncond + guidance * (cond - uncond)``. Every drawn token is fed to both
    streams, so the two stay the same continuation of different prefixes. That
    is two forward passes per token rather than one -- the two prompts are
    different lengths, so they cannot share a batch without a padding-aware
    mask this model does not have.

    Guidance is applied to the raw logits, before *logits_processor*. Combining
    after it would subtract the ``-inf`` a mask writes from itself and yield
    NaN; upstream restricts the combination to the unmasked ids for that
    reason, which on finite logits is the same arithmetic.
    """
    params = params or SamplingParams()
    params.check()
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}.")
    if not prompt_ids:
        raise ValueError("prompt_ids is empty; there is nothing to continue from.")
    if not math.isfinite(guidance) or guidance < 1.0:
        raise ValueError(
            f"guidance must be finite and at least 1.0, where 1.0 means none; "
            f"got {guidance}."
        )
    if uncond_prompt_ids is not None and not uncond_prompt_ids:
        raise ValueError("uncond_prompt_ids is empty; pass None for no guidance.")

    do_cfg = uncond_prompt_ids is not None and guidance > 1.0

    caches = model.new_caches()
    logits = model(mx.array([prompt_ids]), caches)[:, -1, :]

    uncond_caches = None
    uncond_logits = None
    if do_cfg:
        uncond_caches = model.new_caches()
        uncond_logits = model(mx.array([uncond_prompt_ids]), uncond_caches)[:, -1, :]

    # One key per step, split up front: deriving each from the base key inside
    # the loop would re-split a growing stream every token.
    step_keys: list[mx.array | None] = [None] * max_new_tokens
    if seed is not None:
        step_keys = list(mx.random.split(mx.random.key(int(seed)), max_new_tokens))
    drawn: list[int] = []

    for step in range(max_new_tokens):
        step_logits = logits
        if uncond_logits is not None:
            step_logits = uncond_logits.astype(mx.float32) + guidance * (
                step_logits.astype(mx.float32) - uncond_logits.astype(mx.float32)
            )
        if params.repetition_penalty != 1.0 and drawn:
            window = params.repetition_window or len(drawn)
            recent = mx.array(list(set(drawn[-window:])))
            step_logits = apply_repetition_penalty(
                step_logits.astype(mx.float32), recent, params.repetition_penalty
            )
        if logits_processor is not None:
            step_logits = logits_processor(step_logits, drawn)

        token = sample_token(step_logits, params, step_keys[step])
        mx.eval(token)
        token_id = int(token.item())

        if token_id in stop_ids:
            return
        drawn.append(token_id)
        yield token_id

        if step + 1 < max_new_tokens:
            nxt = mx.array([[token_id]])
            logits = model(nxt, caches)[:, -1, :]
            if uncond_caches is not None:
                uncond_logits = model(nxt, uncond_caches)[:, -1, :]
