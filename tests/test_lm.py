"""The MLX Qwen3 planner LM: parity with transformers, caching, and sampling.

The parity tests are the point of this module. Every other MLX model in this
package is checked against a reimplementation of its reference (see
``tests/reference.py``); this one is checked against the reference itself,
because transformers ships Qwen3 and is already a dependency. A reduced-width
model with random weights exercises the same code paths as the 4B planner --
RoPE convention, per-head QK-norm, grouped-query attention, the tied head --
and costs milliseconds.

Parity runs on MLX's CPU stream. Not to avoid the GPU, but because MLX's GPU
float32 matmul is not IEEE float32: it agrees with a float64 reference to about
1e-3 relative, which is fine for inference in bfloat16 and useless for deciding
whether a port is correct. On the CPU stream the same comparison lands at 1e-7,
so a real mistake cannot hide inside the tolerance.
"""

from __future__ import annotations

import json

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np
import pytest

# The reduced planner. Head dim, the grouped-query ratio and rope_theta are the
# published ones; only the widths and the vocabulary shrink.
TINY = {
    "model_type": "qwen3",
    "vocab_size": 97,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "tie_word_embeddings": True,
    "max_position_embeddings": 512,
    "attention_bias": False,
    "use_sliding_window": False,
    "sliding_window": None,
}


def _write_config(path, **overrides):
    cfg = TINY | overrides
    (path / "config.json").write_text(json.dumps(cfg))
    return cfg


@pytest.fixture
def pair(tmp_path):
    """A transformers Qwen3 and the MLX port of the very same weights."""
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from as15.mlx.lm import MLXQwen3LM

    torch.manual_seed(0)
    cfg = _write_config(tmp_path)
    reference = Qwen3ForCausalLM(Qwen3Config(**cfg)).eval()
    # The head is tied, so the checkpoint does not carry it -- and the loader
    # must not need it.
    save_file(
        {
            key: value.contiguous()
            for key, value in reference.state_dict().items()
            if key != "lm_head.weight"
        },
        str(tmp_path / "model.safetensors"),
    )
    return reference, MLXQwen3LM.from_snapshot(tmp_path)


IDS = np.arange(1, 25, dtype=np.int64)[None, :]


def _reference_logits(reference, ids):
    import torch

    with torch.no_grad():
        return reference(torch.tensor(ids)).logits.float().numpy()


# --- parity ---------------------------------------------------------------


def test_a_prompt_forward_matches_transformers_qwen3(pair):
    """The whole stack, against the implementation it was ported from.

    A wrong rotary convention, QK-norm applied to the merged heads instead of
    per head, or key/value heads tiled the wrong way round all produce a model
    that still emits fluent tokens; only a numeric comparison catches them.
    """
    reference, model = pair
    with mx.stream(mx.cpu):
        got = np.array(model(mx.array(IDS), None).astype(mx.float32))
    expected = _reference_logits(reference, IDS)
    assert np.abs(got - expected).max() < 1e-5


def test_decoding_one_token_at_a_time_matches_the_whole_prompt(pair):
    """The cache has to be transparent, not merely fast.

    Each cached step re-derives its rotary positions from the cache offset; an
    offset that reset to zero would still decode, into a model that thinks
    every token is the first one.
    """
    reference, model = pair
    expected = _reference_logits(reference, IDS)

    with mx.stream(mx.cpu):
        caches = model.new_caches()
        model(mx.array(IDS[:, :8]), caches)
        stepped = [
            np.array(model(mx.array(IDS[:, i : i + 1]), caches).astype(mx.float32))[
                :, 0
            ]
            for i in range(8, IDS.shape[1])
        ]

    assert np.abs(np.stack(stepped, axis=1) - expected[:, 8:]).max() < 1e-5


def test_the_cache_survives_growing_past_its_block(pair):
    """The buffer is grown in blocks, and the copy across has to keep the old keys."""
    from as15.mlx.lm import KVCache

    _reference, model = pair
    monkey = KVCache.STEP
    try:
        KVCache.STEP = 4  # force several regrowths over a 24-token prompt
        with mx.stream(mx.cpu):
            caches = model.new_caches()
            stepped = [
                np.array(model(mx.array(IDS[:, i : i + 1]), caches).astype(mx.float32))[
                    :, 0
                ]
                for i in range(IDS.shape[1])
            ]
    finally:
        KVCache.STEP = monkey

    expected = _reference_logits(_reference, IDS)
    assert np.abs(np.stack(stepped, axis=1) - expected).max() < 1e-5


# --- config -------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"model_type": "llama"}, "implements qwen3"),
        ({"use_sliding_window": True, "sliding_window": 128}, "sliding attention"),
        ({"tie_word_embeddings": False}, "untied output head"),
        ({"attention_bias": True}, "attention biases"),
    ],
)
def test_a_config_this_loader_would_misrun_is_rejected(tmp_path, overrides, expected):
    """Each of these loads and generates; none of them generates correctly."""
    from as15.mlx.lm import Qwen3Config

    _write_config(tmp_path, **overrides)
    with pytest.raises(ValueError, match=expected):
        Qwen3Config.from_file(tmp_path / "config.json")


# --- sampling -----------------------------------------------------------


def _logits(*values):
    return mx.array([list(values)], dtype=mx.float32)


def test_zero_temperature_is_greedy():
    from as15.mlx.lm import SamplingParams, sample_token

    drawn = sample_token(_logits(0.1, 5.0, 0.2), SamplingParams(temperature=0.0), None)
    assert int(drawn.item()) == 1


def test_top_k_leaves_only_the_k_highest_reachable():
    from as15.mlx.lm import SamplingParams, sample_token

    params = SamplingParams(temperature=1.0, top_k=2)
    logits = _logits(0.0, 9.0, 1.0, 8.5)
    drawn = {
        int(sample_token(logits, params, mx.random.key(seed)).item())
        for seed in range(64)
    }
    assert drawn <= {1, 3}


def test_top_p_keeps_the_most_probable_token_even_when_it_exceeds_the_threshold():
    """A single token holding more than *top_p* of the mass is still drawable.

    Masking on the inclusive cumulative probability instead would leave the
    whole distribution at -inf, and sampling from that draws whatever the
    softmax of all-equal-infinities happens to be.
    """
    from as15.mlx.lm import SamplingParams, sample_token

    params = SamplingParams(temperature=1.0, top_p=0.5)
    logits = _logits(0.0, 20.0, 0.0)  # token 1 carries essentially all the mass
    drawn = {
        int(sample_token(logits, params, mx.random.key(seed)).item())
        for seed in range(32)
    }
    assert drawn == {1}


def test_the_repetition_penalty_pushes_both_signs_downwards():
    """Dividing a negative logit raises it, so the sign picks the operation."""
    from as15.mlx.lm import apply_repetition_penalty

    logits = _logits(2.0, -2.0, 1.0)
    out = np.array(apply_repetition_penalty(logits, mx.array([0, 1]), 2.0))
    assert out[0, 0] == pytest.approx(1.0)  # 2.0 / 2
    assert out[0, 1] == pytest.approx(-4.0)  # -2.0 * 2
    assert out[0, 2] == pytest.approx(1.0)  # untouched


def test_the_same_seed_draws_the_same_plan_twice(pair):
    """And a different one does not, so the seed is doing the work."""
    from as15.mlx.lm import SamplingParams, generate

    _reference, model = pair
    params = SamplingParams(temperature=1.0, top_p=0.95)

    def run(seed):
        return list(generate(model, [1, 2, 3], 12, params, seed=seed))

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_a_stop_id_ends_the_run_without_being_yielded(pair):
    from as15.mlx.lm import SamplingParams, generate

    _reference, model = pair
    params = SamplingParams(temperature=1.0)
    full = list(generate(model, [1, 2, 3], 16, params, seed=0))

    # The first token that has not been drawn before, so stopping on it cannot
    # also have ended an earlier step.
    cut = next(i for i, token in enumerate(full) if token not in full[:i])
    stopped = list(
        generate(model, [1, 2, 3], 16, params, seed=0, stop_ids=frozenset({full[cut]}))
    )
    assert stopped == full[:cut]


def test_a_logits_processor_can_make_a_token_unreachable(pair):
    from as15.mlx.lm import SamplingParams, generate

    _reference, model = pair
    allowed = {4, 5}

    def only_allowed(logits, _drawn):
        mask = mx.full(logits.shape, -mx.inf, dtype=mx.float32)
        for token in allowed:
            mask[0, token] = 0.0
        return logits.astype(mx.float32) + mask

    drawn = list(
        generate(
            model,
            [1, 2, 3],
            16,
            SamplingParams(temperature=1.0),
            seed=3,
            logits_processor=only_allowed,
        )
    )
    assert drawn and set(drawn) <= allowed


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"temperature": -1.0}, "temperature"),
        ({"temperature": float("inf")}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.5}, "top_p"),
        ({"top_k": -1}, "top_k"),
        ({"repetition_penalty": 0.0}, "repetition_penalty"),
        ({"repetition_window": -2}, "repetition_window"),
    ],
)
def test_sampling_settings_that_mean_something_else_are_rejected(kwargs, expected):
    from as15.mlx.lm import SamplingParams

    with pytest.raises(ValueError, match=expected):
        SamplingParams(**kwargs).check()


# --- guidance -------------------------------------------------------------


def test_guidance_moves_the_draw_and_leaves_it_alone_at_one(pair):
    """A second stream is only worth two forward passes if it changes something.

    At 1.0 the formula is the identity, so the guided path must reproduce the
    unguided draw exactly -- otherwise the extra pass is being run and its
    result quietly discarded, or worse, not discarded.
    """
    from as15.mlx.lm import SamplingParams, generate

    _reference, model = pair
    params = SamplingParams(temperature=1.0)

    def run(**kwargs):
        return list(generate(model, [1, 2, 3], 10, params, seed=4, **kwargs))

    plain = run()
    assert run(uncond_prompt_ids=[9], guidance=1.0) == plain
    assert run(uncond_prompt_ids=[9], guidance=3.0) != plain


def test_guidance_is_applied_before_the_mask_so_it_cannot_make_nan(pair):
    """Combining after a mask subtracts -inf from itself.

    ``-inf + g * (-inf - -inf)`` is NaN, and a NaN logit poisons the whole
    softmax -- the draw stops being a draw. Upstream restricts its combination
    to the unmasked ids for exactly this reason.
    """
    from as15.mlx.lm import SamplingParams, generate

    _reference, model = pair
    allowed = {4, 5, 6}

    seen: list[bool] = []

    def only_allowed(logits, _drawn):
        # What the processor receives must still be finite: that is the
        # ordering under test.
        seen.append(bool(mx.all(mx.isfinite(logits)).item()))
        mask = mx.full(logits.shape, -mx.inf, dtype=mx.float32)
        for token in allowed:
            mask[0, token] = 0.0
        return logits.astype(mx.float32) + mask

    drawn = list(
        generate(
            model,
            [1, 2, 3],
            8,
            SamplingParams(temperature=1.0),
            seed=2,
            logits_processor=only_allowed,
            uncond_prompt_ids=[7, 8],
            guidance=2.0,
        )
    )
    assert all(seen), "guidance ran after the mask and produced -inf arithmetic"
    assert drawn and set(drawn) <= allowed


@pytest.mark.parametrize("guidance", [0.5, -1.0, float("nan"), float("inf")])
def test_a_guidance_that_does_not_mean_what_it_says_is_rejected(pair, guidance):
    from as15.mlx.lm import generate

    _reference, model = pair
    with pytest.raises(ValueError, match="guidance"):
        list(generate(model, [1, 2], 4, uncond_prompt_ids=[3], guidance=guidance))


def test_an_empty_unconditional_prompt_is_rejected(pair):
    """``None`` is how you ask for no guidance; an empty list is a mistake."""
    from as15.mlx.lm import generate

    _reference, model = pair
    with pytest.raises(ValueError, match="uncond_prompt_ids"):
        list(generate(model, [1, 2], 4, uncond_prompt_ids=[], guidance=2.0))


# --- checkpoint layouts ---------------------------------------------------


@pytest.mark.parametrize("prefix", ["", "model."])
def test_both_published_key_layouts_load(tmp_path, prefix):
    """The planners ship under two names for the same tensors.

    The ones with repos of their own carry ``model.embed_tokens.weight``; the
    1.7B, published as a directory inside the shared base repo, carries a bare
    ``embed_tokens.weight``. Loading only one of them leaves `--planner 1.7b`
    failing on a strict load with a wall of missing keys.
    """
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from as15.mlx.lm import MLXQwen3LM

    torch.manual_seed(0)
    cfg = _write_config(tmp_path)
    reference = Qwen3ForCausalLM(Qwen3Config(**cfg)).eval()
    save_file(
        {
            prefix + key.removeprefix("model."): value.contiguous()
            for key, value in reference.state_dict().items()
            if key != "lm_head.weight"
        },
        str(tmp_path / "model.safetensors"),
    )

    model = MLXQwen3LM.from_snapshot(tmp_path)
    with mx.stream(mx.cpu):
        got = np.array(model(mx.array(IDS), None).astype(mx.float32))
    assert np.abs(got - _reference_logits(reference, IDS)).max() < 1e-5


def test_a_checkpoint_carrying_a_tied_head_anyway_still_loads(tmp_path):
    """It is a copy of the embedding table, and a strict load would reject it."""
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from as15.mlx.lm import MLXQwen3LM

    torch.manual_seed(0)
    cfg = _write_config(tmp_path)
    reference = Qwen3ForCausalLM(Qwen3Config(**cfg)).eval()
    # Cloned, because safetensors refuses to write two names for one
    # storage -- which is exactly why a tied head is normally left out.
    state = {k: v.clone() for k, v in reference.state_dict().items()}
    state["lm_head.weight"] = reference.get_input_embeddings().weight.clone()
    save_file(state, str(tmp_path / "model.safetensors"))

    MLXQwen3LM.from_snapshot(tmp_path)
