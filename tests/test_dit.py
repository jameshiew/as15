"""The MLX DiT: attention, the cross-attention cache and dtype propagation.

Sized down from the XL config throughout -- none of what is under test here
depends on the widths, and a 4.17 B forward is not a unit test.
"""

from __future__ import annotations

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np

from helpers import STEPS, cfg_kwargs, run_sampler, small_decoder

# --- attention ------------------------------------------------------------


def test_grouped_query_attention_is_not_tiled_out_by_hand(monkeypatch):
    """K/V reach the kernel with their own head count.

    They used to be broadcast up to the query head count first, which on the
    XL config (32 query heads over 8 key/value heads) materialised a fourfold
    copy of every key and value, on both attentions of every layer. MLX's fast
    kernel groups the heads itself, and does it bit-for-bit identically -- so
    the copies bought nothing.
    """
    from as15.mlx import dit

    captured: list[tuple] = []
    real_sdpa = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, *, scale, mask=None):
        captured.append((q, k, v))
        return real_sdpa(q, k, v, scale=scale, mask=mask)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)

    attn = dit.MLXAttention(
        hidden_size=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        attention_bias=False,
        layer_idx=0,
    )
    out = attn(mx.random.normal((1, 6, 16), key=mx.random.key(0)))

    (q, k, v), *rest = captured
    assert not rest
    assert q.shape[1] == 8
    assert k.shape[1] == v.shape[1] == 2

    def _tile(x, n_rep):
        b, h, length, d = x.shape
        return mx.broadcast_to(x[:, :, None], (b, h, n_rep, length, d)).reshape(
            b, h * n_rep, length, d
        )

    tiled = real_sdpa(q, _tile(k, 4), _tile(v, 4), scale=attn.scale, mask=None)
    reference = attn.o_proj(tiled.transpose(0, 2, 1, 3).reshape(1, 6, -1))
    assert np.array_equal(np.array(out), np.array(reference))


def test_the_conditioning_is_projected_once_for_the_whole_run(monkeypatch):
    """Cross-attention K/V come from the conditioning and nothing else.

    No noisy latent and no timestep reaches them, so one cache entry per layer
    survives every step, both of Heun's evaluations and CFG's doubled batch.
    Caching was switched off whenever CFG or Heun was on -- between them, the
    whole default SFT path -- so the conditioning went back through every
    layer's ``k_proj`` and ``v_proj`` on all 2N evaluations.
    """
    from as15.mlx import dit

    updated: list[int] = []
    real_update = dit.MLXCrossAttentionCache.update

    def spy(self, key, value, layer_idx):
        updated.append(layer_idx)
        return real_update(self, key, value, layer_idx)

    monkeypatch.setattr(dit.MLXCrossAttentionCache, "update", spy)

    layers = 2
    forwards = []
    inner = small_decoder(layers)

    def counting_decoder(**kwargs):
        forwards.append(kwargs["timestep"])
        return inner(**kwargs)

    run_sampler(counting_decoder, sampler_mode="heun", **cfg_kwargs())

    assert len(forwards) == 2 * STEPS
    assert updated == list(range(layers))


# --- compute dtype --------------------------------------------------------


def test_rope_tables_are_handed_out_in_the_activation_dtype():
    """float32 cos/sin would promote every query and key they multiply.

    transformers' Qwen3RotaryEmbedding -- which upstream uses -- ends its
    forward with ``cos.to(dtype=x.dtype)`` for the same reason.
    """
    from as15.mlx.dit import MLXRotaryEmbedding

    rope = MLXRotaryEmbedding(head_dim=8, max_len=16, base=1e6)
    cos, sin = rope(4, mx.bfloat16)
    assert cos.dtype == sin.dtype == mx.bfloat16
    assert cos.shape == sin.shape == (1, 1, 4, 8)

    # The tables themselves stay float32: the angles need the headroom.
    exact_cos, _ = rope(4, mx.float32)
    assert exact_cos.dtype == mx.float32
    assert np.allclose(np.array(cos.astype(mx.float32)), np.array(exact_cos), atol=1e-2)


def test_a_bf16_decoder_forward_stays_bf16_end_to_end():
    """One float32 input is enough to widen the whole stack.

    Sized down from the XL config; the dtype plumbing is shape-independent.
    """
    from as15.mlx.dit import MLXDiTDecoder

    decoder = MLXDiTDecoder(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        in_channels=12,
        audio_acoustic_hidden_dim=4,
        patch_size=2,
        sliding_window=4,
        max_position_embeddings=64,
        encoder_hidden_size=6,
    )
    decoder.set_dtype(mx.bfloat16)

    def _forward(timestep):
        out, _ = decoder(
            hidden_states=mx.zeros((1, 8, 4), dtype=mx.bfloat16),
            timestep=timestep,
            timestep_r=timestep,
            encoder_hidden_states=mx.zeros((1, 3, 6), dtype=mx.bfloat16),
            context_latents=mx.zeros((1, 8, 8), dtype=mx.bfloat16),
        )
        return out

    assert _forward(mx.full((1,), 0.75, dtype=mx.bfloat16)).dtype == mx.bfloat16
    # Pin the promotion the sampler used to trigger, so the fix is not silently
    # undone by dropping the dtype at the one call site that builds it.
    assert _forward(mx.full((1,), 0.75)).dtype == mx.float32
