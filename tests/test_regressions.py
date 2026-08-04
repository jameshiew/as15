"""Regressions for the two bugs that produced garbled audio.

Both were silent: the model still emitted plausible-looking audio, so only
listening (or spectral flatness) revealed them.
"""

from __future__ import annotations

import json

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np
import pytest

from as15 import conditioning, convert, models
from as15.models import BASE_REPO, BASE_REVISION, MODELS, Snapshot


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


# --- APG guidance --------------------------------------------------------


def test_apg_momentum_decays_rather_than_accumulates():
    """``running = diff + (-0.75) * running``, per upstream's MomentumBuffer.

    Getting the sign wrong sums every past guidance difference with weight
    +1.0 instead, so the trajectory diverges from the reference after the
    first step -- on every xl-sft generation, which guides at 7.0 by default.
    """
    from as15.mlx.sampler import APG_MOMENTUM, _mlx_apg_forward

    assert APG_MOMENTUM == -0.75

    first = mx.array([[[1.0, 0.0]]])
    second = mx.array([[[0.0, 1.0]]])
    zero = mx.zeros((1, 1, 2))

    state: dict = {}
    # norm_threshold=0 keeps the running average out of the clamp, so the
    # state after each call is exactly the recurrence under test.
    _mlx_apg_forward(first, zero, 7.0, state, norm_threshold=0.0)
    assert np.allclose(np.array(state["running"]), np.array(first))

    _mlx_apg_forward(second, zero, 7.0, state, norm_threshold=0.0)
    expected = np.array(second) + APG_MOMENTUM * np.array(first)
    assert np.allclose(np.array(state["running"]), expected)


def test_apg_without_momentum_is_stateless():
    """The non-CFG path passes no state and must stay a pure function."""
    from as15.mlx.sampler import _mlx_apg_forward

    cond = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
    uncond = mx.array([[[0.5, 0.0], [1.0, 2.0]]])
    a = _mlx_apg_forward(cond, uncond, 7.0, None)
    b = _mlx_apg_forward(cond, uncond, 7.0, None)
    assert np.array_equal(np.array(a), np.array(b))


# --- sampler stepping ----------------------------------------------------


class _ConstantDecoder:
    """Stand-in for the DiT that records the timestep of every evaluation.

    Predicts a velocity of 1.0, offset by batch row so that the conditional
    and unconditional halves of a CFG batch differ.
    """

    def __init__(self) -> None:
        self.timesteps: list[float] = []

    def __call__(
        self,
        *,
        hidden_states,
        timestep,
        timestep_r,
        encoder_hidden_states,
        context_latents,
        cache,
        use_cache,
    ):
        self.timesteps.append(float(np.array(timestep)[0]))
        rows = mx.arange(hidden_states.shape[0]).reshape(-1, 1, 1)
        return mx.ones_like(hidden_states) + 0.1 * rows, cache


# shift=1.0 and infer_steps=3 give the schedule [1, 2/3, 1/3]; the interval
# that ends at t=0 is the one the loop used to special-case.
SCHEDULE = [1.0, 2 / 3, 1 / 3]
STEPS = len(SCHEDULE)
NOISE_SHAPE = (1, 4, 8)


def _run_sampler(decoder, **kwargs):
    from as15.mlx.sampler import mlx_generate_diffusion

    b, t, c = NOISE_SHAPE
    return mlx_generate_diffusion(
        decoder,
        encoder_hidden_states_np=np.zeros((b, 3, 6), dtype=np.float32),
        context_latents_np=np.zeros((b, t, c), dtype=np.float32),
        src_latents_shape=NOISE_SHAPE,
        seed=0,
        infer_steps=STEPS,
        shift=1.0,
        dcw_enabled=False,
        disable_tqdm=True,
        compute_dtype="float32",
        **kwargs,
    )


def _cfg_kwargs():
    return {
        "guidance_scale": 7.0,
        "null_condition_emb_np": np.zeros((1, 1, 6), dtype=np.float32),
    }


def test_heun_corrects_the_interval_that_ends_at_zero():
    """The last interval used to be a bare Euler hop to t=0.

    Upstream pairs ``zip(t[:-1], t[1:])`` over a schedule ending at zero, so
    every interval gets a corrector. Dropping the last one left an N-step Heun
    run at 2N-1 evaluations, first-order exactly where the trajectory lands on
    the clean sample.
    """
    decoder = _ConstantDecoder()
    _run_sampler(decoder, sampler_mode="heun")

    # Predictor at t_curr, corrector at t_next, for every interval.
    assert np.allclose(
        decoder.timesteps,
        [1.0, 2 / 3, 2 / 3, 1 / 3, 1 / 3, 0.0],
    )
    assert len(decoder.timesteps) == 2 * STEPS


def test_heun_advances_the_apg_momentum_once_per_interval(monkeypatch):
    """The corrector guides through plain CFG, not APG.

    Both evaluations used to run the stateful APG path, so the momentum
    recurrence stepped twice per interval and desynchronised from the
    reference after the first step.
    """
    from as15.mlx import sampler

    apg_calls: list[dict | None] = []
    real_apg = sampler._mlx_apg_forward
    cfg_calls: list[float] = []
    real_cfg = sampler._mlx_cfg_forward

    def spy_apg(pred_cond, pred_uncond, guidance_scale, momentum_state=None, **kw):
        apg_calls.append(momentum_state)
        return real_apg(pred_cond, pred_uncond, guidance_scale, momentum_state, **kw)

    def spy_cfg(pred_cond, pred_uncond, guidance_scale):
        cfg_calls.append(guidance_scale)
        return real_cfg(pred_cond, pred_uncond, guidance_scale)

    monkeypatch.setattr(sampler, "_mlx_apg_forward", spy_apg)
    monkeypatch.setattr(sampler, "_mlx_cfg_forward", spy_cfg)

    _run_sampler(_ConstantDecoder(), sampler_mode="heun", **_cfg_kwargs())

    assert len(apg_calls) == STEPS
    assert all(state is not None for state in apg_calls)
    assert len(cfg_calls) == STEPS


def test_the_last_interval_still_lands_on_x0():
    """Euler is unchanged by pairing the last interval with t=0.

    The step size there is ``t_last - 0``, so the update stays ``x - v*t``:
    with v == 1 the schedule telescopes to ``noise - t_0``.
    """
    decoder = _ConstantDecoder()
    result = _run_sampler(decoder, sampler_mode="euler")

    assert np.allclose(decoder.timesteps, SCHEDULE)
    noise = np.array(mx.random.normal(NOISE_SHAPE, key=mx.random.key(0)))
    assert np.allclose(result["target_latents"], noise - SCHEDULE[0], atol=1e-5)


def test_sde_draws_no_noise_for_the_interval_that_ends_at_zero(monkeypatch):
    """``t_next == 0`` scales the fresh noise away, so it must not be drawn.

    Drawing it anyway would advance MLX's implicit PRNG state once per
    generation for nothing.
    """
    real_normal = mx.random.normal
    keyless = []

    def spy(shape, *args, key=None, **kw):
        if key is None:
            keyless.append(shape)
        return real_normal(shape, *args, key=key, **kw)

    monkeypatch.setattr(mx.random, "normal", spy)

    _run_sampler(_ConstantDecoder(), infer_method="sde")

    assert len(keyless) == STEPS - 1


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


# --- converted-weight cache identity -------------------------------------


def test_every_repo_is_pinned_to_a_commit():
    """Unpinned repos make the cache -- and the tuned defaults -- unreproducible."""
    revisions = [BASE_REVISION, *(spec.revision for spec in MODELS.values())]
    for revision in revisions:
        assert len(revision) == 40
        assert set(revision) <= set("0123456789abcdef")


def test_cache_paths_separate_every_identity_component(monkeypatch, tmp_path):
    """Repo, commit, converter version and precision must not alias."""
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    paths = {
        convert.dit_cache_path("acestep-v15-xl-sft", "aaaa", "bf16"),
        convert.dit_cache_path("acestep-v15-xl-sft", "aaaa", "fp32"),
        convert.dit_cache_path("acestep-v15-xl-sft", "bbbb", "bf16"),
        convert.dit_cache_path("acestep-v15-xl-turbo", "aaaa", "bf16"),
        convert.vae_cache_path("aaaa"),
        convert.vae_cache_path("bbbb"),
    }
    assert len(paths) == 6
    # The converter version is in the filename, so a bump orphans the old file
    # rather than reusing weights in a layout the MLX models no longer expect.
    assert f"-v{convert.DIT_CONVERTER_VERSION}-" in (
        convert.dit_cache_path("m", "aaaa", "bf16").name
    )
    assert f"-v{convert.VAE_CONVERTER_VERSION}-" in convert.vae_cache_path("aaaa").name


def test_a_revision_is_flattened_into_one_path_component(monkeypatch, tmp_path):
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    path = convert.dit_cache_path("m", "refs/pr/7", "bf16")
    assert path.parent.name == "refs-pr-7"
    assert path.parent.parent.name == "m"


def _dit_manifest(snapshot: Snapshot, precision: str = "bf16") -> dict:
    return {
        "asset": "dit",
        "converter_version": convert.DIT_CONVERTER_VERSION,
        "precision": precision,
        "repo_id": snapshot.repo_id,
        "revision": snapshot.revision,
    }


def test_cache_is_reused_only_when_the_manifest_matches(monkeypatch, tmp_path):
    """A weights file with no matching manifest must not be trusted.

    Loading a DiT converted from different bytes does not fail -- it generates
    audibly worse audio -- so the check has to happen before reuse.
    """
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    snapshot = Snapshot("ACE-Step/acestep-v15-xl-sft", "a" * 40, tmp_path / "snap")
    out = convert.dit_cache_path(snapshot.cache_name, snapshot.revision, "bf16")
    out.parent.mkdir(parents=True)
    out.write_bytes(b"not really safetensors")

    manifest = _dit_manifest(snapshot)
    assert not convert._cache_hit(out, manifest)  # no manifest written yet

    convert._manifest_path(out).write_text(json.dumps(manifest))
    assert convert._cache_hit(out, manifest)

    # A manifest describing anything else is a miss, not a silent reuse.
    stale = Snapshot(snapshot.repo_id, "b" * 40, snapshot.path)
    assert not convert._cache_hit(out, _dit_manifest(stale))
    assert not convert._cache_hit(out, _dit_manifest(snapshot, precision="fp32"))
    convert._manifest_path(out).write_text("{ truncated")
    assert not convert._cache_hit(out, manifest)


def test_shared_vae_cache_is_keyed_on_the_base_repo(monkeypatch, tmp_path):
    """The VAE is shared across checkpoints, so only the base repo pins it."""
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    out = convert.vae_cache_path(BASE_REVISION)
    out.parent.mkdir(parents=True)
    out.write_bytes(b"")
    manifest = {
        "asset": "vae",
        "converter_version": convert.VAE_CONVERTER_VERSION,
        "precision": "fp32",
        "repo_id": BASE_REPO,
        "revision": BASE_REVISION,
    }
    convert._manifest_path(out).write_text(json.dumps(manifest))
    assert convert._cache_hit(out, manifest)
    assert not convert._cache_hit(convert.vae_cache_path("c" * 40), manifest)


# --- latent geometry -----------------------------------------------------


def test_latent_frame_rate():
    from as15.models import LATENT_FPS, SAMPLE_RATE, VAE_HOP

    # The Oobleck VAE downsamples by prod([2, 4, 4, 6, 10]).
    assert VAE_HOP == 2 * 4 * 4 * 6 * 10
    assert LATENT_FPS == SAMPLE_RATE // VAE_HOP == 25


# vae/config.json as published at BASE_REVISION. Not the Stable Audio Oobleck
# geometry ([2, 4, 4, 8, 8] at 44.1 kHz), which is what makes the check worth
# having.
PINNED_VAE_CONFIG = {
    "audio_channels": 2,
    "channel_multiples": [1, 2, 4, 8, 16],
    "decoder_channels": 128,
    "decoder_input_channels": 64,
    "downsampling_ratios": [2, 4, 4, 6, 10],
    "encoder_hidden_size": 128,
    "sampling_rate": 48000,
}


def test_pinned_vae_config_matches_the_latent_geometry():
    models.check_vae_geometry(PINNED_VAE_CONFIG)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("downsampling_ratios", [2, 4, 4, 8, 8], "hop 2048"),
        ("sampling_rate", 44100, "sampling rate 44100"),
        ("decoder_input_channels", 128, "latent channels 128"),
    ],
)
def test_a_checkpoint_that_moves_the_latent_geometry_is_rejected(field, value, message):
    """Conditioning sizes the latent window before the VAE is loaded.

    A checkpoint on a different hop, rate or latent width would decode to the
    wrong duration rather than fail, so the mismatch has to be caught at load.
    """
    cfg = {**PINNED_VAE_CONFIG, field: value}
    with pytest.raises(RuntimeError, match=message):
        models.check_vae_geometry(cfg)


def test_metas_block_format():
    text = conditioning.format_metas(110, "C major", 4, 30.0)
    assert text == (
        "- bpm: 110\n- timesignature: 4\n- keyscale: C major\n- duration: 30 seconds\n"
    )
    # Unset fields must render as N/A, not None or an empty string.
    assert conditioning.format_metas(None, None, None, 12.7) == (
        "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 12 seconds\n"
    )
