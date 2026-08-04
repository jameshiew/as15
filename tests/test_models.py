"""The registry, the pinned checkpoints and the latent geometry they imply.

None of this runs a model, and all of it decides what the model does.
"""

from __future__ import annotations

import pytest

from as15 import models
from as15.models import BASE_REVISION, MODELS


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


def test_an_unknown_model_is_a_value_error_not_a_process_exit():
    """``resolve`` is a library helper, and used to raise SystemExit.

    A bad name therefore tore down the process of anything embedding the
    package -- a service, a notebook, a test -- with nothing to catch.
    """
    with pytest.raises(ValueError, match="Unknown model"):
        models.resolve("xl-sft-turbo-plus")
    assert models.resolve("xl-sft") is MODELS["xl-sft"]


def test_every_repo_is_pinned_to_a_commit():
    """Unpinned repos make the cache -- and the tuned defaults -- unreproducible."""
    revisions = [BASE_REVISION, *(spec.revision for spec in MODELS.values())]
    for revision in revisions:
        assert len(revision) == 40
        assert set(revision) <= set("0123456789abcdef")


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
