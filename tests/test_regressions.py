"""Regressions for the two bugs that produced garbled audio.

Both were silent: the model still emitted plausible-looking audio, so only
listening (or spectral flatness) revealed them.
"""

from __future__ import annotations

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np
import pytest

from as15 import conditioning, convert
from as15.models import MODELS


def test_chunk_mask_is_one_not_two():
    """Upstream's `chunk_masks_tensor[i] = 2.0` lands in a *bool* tensor.

    It therefore saturates to True and reaches the DiT as 1.0. Copying the
    literal 2.0 into the context channel pushes it outside the trained range
    and garbles the output.
    """
    assert conditioning.CHUNK_MASK_FULL == 1.0


def test_bool_assignment_saturates():
    """Pin the torch semantics the constant above is derived from."""
    torch = pytest.importorskip("torch")
    mask = torch.stack([torch.ones(4, dtype=torch.bool)])
    mask[0] = 2.0
    assert mask.to(torch.float32).max().item() == 1.0


@pytest.mark.parametrize(
    ("key", "dcw", "shift"),
    [("xl-sft", False, 1.0), ("xl-turbo", True, 3.0)],
)
def test_dcw_and_shift_defaults_are_model_aware(key, dcw, shift):
    """DCW was tuned for the distilled models only (upstream issue #1259).

    Leaving it on for xl-sft/xl-base is the documented cause of mushy,
    distorted output.
    """
    spec = MODELS[key]
    assert spec.dcw is dcw
    assert spec.shift == shift


def test_turbo_does_not_use_cfg():
    assert MODELS["xl-turbo"].supports_cfg is False
    assert MODELS["xl-sft"].supports_cfg is True


# --- weight conversion layout -------------------------------------------


def test_conv_layout_swaps_in_and_kernel_axes():
    # PyTorch Conv1d [out, in, K] -> MLX [out, K, in]
    w = mx.arange(2 * 3 * 5).reshape(2, 3, 5)
    out = convert._apply_layout(w, "conv")
    assert out.shape == (2, 5, 3)
    assert out[1, 4, 2].item() == w[1, 2, 4].item()


def test_conv_transpose_layout_moves_in_channels_last():
    # PyTorch ConvTranspose1d [in, out, K] -> MLX [out, K, in]
    w = mx.arange(2 * 3 * 5).reshape(2, 3, 5)
    out = convert._apply_layout(w, "conv_t")
    assert out.shape == (3, 5, 2)
    assert out[2, 4, 1].item() == w[1, 2, 4].item()


def test_dit_key_mapping_unwraps_sequential_and_drops_rope():
    assert convert._convert_dit_key("decoder.proj_in.1.weight") == (
        "proj_in.weight",
        "conv",
    )
    assert convert._convert_dit_key("decoder.proj_out.1.weight") == (
        "proj_out.weight",
        "conv_t",
    )
    assert convert._convert_dit_key("decoder.proj_in.1.bias") == (
        "proj_in.bias",
        "plain",
    )
    # RoPE tables are recomputed by the MLX model.
    assert (
        convert._convert_dit_key("decoder.layers.0.self_attn.rotary_emb.inv_freq")
        is None
    )
    # Only the DiT subtree is converted; the FSQ codec never runs for text2music.
    assert (
        convert._convert_dit_key("encoder.lyric_encoder.layers.0.mlp.up_proj.weight")
        is None
    )
    assert convert._convert_dit_key("tokenizer.quantizer.weight") is None


def test_weight_norm_fusion_matches_definition():
    g = mx.array(np.array([[[2.0]], [[3.0]]], dtype=np.float32))
    v = mx.random.normal((2, 4, 3), key=mx.random.key(0))
    fused = np.array(convert._fuse_weight_norm(g, v))
    v_np = np.array(v)
    norms = np.linalg.norm(v_np.reshape(2, -1), axis=1).reshape(2, 1, 1)
    expected = np.array(g).reshape(2, 1, 1) * v_np / norms
    assert np.allclose(fused, expected, atol=1e-5)


# --- latent geometry -----------------------------------------------------


def test_latent_frame_rate():
    from as15.models import LATENT_FPS, SAMPLE_RATE, VAE_HOP

    # The Oobleck VAE downsamples by prod([2, 4, 4, 6, 10]).
    assert VAE_HOP == 2 * 4 * 4 * 6 * 10
    assert LATENT_FPS == SAMPLE_RATE // VAE_HOP == 25


def test_metas_block_format():
    text = conditioning.format_metas(110, "C major", 4, 30.0)
    assert text == (
        "- bpm: 110\n- timesignature: 4\n- keyscale: C major\n- duration: 30 seconds\n"
    )
    # Unset fields must render as N/A, not None or an empty string.
    assert conditioning.format_metas(None, None, None, 12.7) == (
        "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 12 seconds\n"
    )
