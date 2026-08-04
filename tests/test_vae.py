"""The Oobleck VAE: windowed decode, and the weights it is handed.

Two claims are checked here that nothing else can check. The first is that
decoding in windows returns what decoding the whole track returns -- the
reason the windows exist is memory, and a seam is a click every twenty
seconds, not a failure. The second is that a checkpoint converted for MLX
still computes what the published weights say it does: the converter fuses
``weight_norm``, permutes three different axis orders and reshapes the Snake
parameters, and every one of those is silent when wrong.
"""

from __future__ import annotations

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np
import pytest

import reference
from as15 import convert, models, pipeline
from as15.mlx.vae import MLXAutoEncoderOobleck
from helpers import flat_parameters, randomised

# --- the reference, checked against the framework it describes -------------


def test_the_reference_convolutions_are_the_ones_pytorch_computes():
    """``tests/reference.py`` is the expectation for everything below.

    It is written from the documented semantics rather than from MLX, which
    is the point -- but that only helps if the semantics are right, so they
    are pinned against torch itself: padding, dilation, stride, and the fact
    that a transposed convolution *crops* the padding it is given rather than
    adding it.
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)

    x = rng.normal(size=(2, 11, 3))
    weight = rng.normal(size=(5, 3, 7))
    bias = rng.normal(size=(5,))
    expected = torch.nn.functional.conv1d(
        torch.tensor(x.transpose(0, 2, 1)),
        torch.tensor(weight),
        torch.tensor(bias),
        stride=2,
        padding=9,
        dilation=3,
    ).numpy()
    got = reference.conv1d(x, weight, bias, stride=2, padding=9, dilation=3)
    assert np.allclose(got, expected.transpose(0, 2, 1))

    x = rng.normal(size=(2, 6, 3))
    weight = rng.normal(size=(3, 5, 8))
    bias = rng.normal(size=(5,))
    expected = torch.nn.functional.conv_transpose1d(
        torch.tensor(x.transpose(0, 2, 1)),
        torch.tensor(weight),
        torch.tensor(bias),
        stride=4,
        padding=2,
    ).numpy()
    got = reference.conv_transpose1d(x, weight, bias, stride=4, padding=2)
    assert np.allclose(got, expected.transpose(0, 2, 1))


# --- converted weights ----------------------------------------------------

# Narrow enough that the numpy reference decodes in milliseconds. Only the
# widths are reduced: the kernel sizes, dilations and the three axis orders
# the converter has to get right are properties of the architecture, not of
# how many channels it was published with.
TINY_RATIOS = [2, 4]
TINY_LATENT_CHANNELS = 6


def _tiny_vae() -> MLXAutoEncoderOobleck:
    return MLXAutoEncoderOobleck(
        encoder_hidden_size=4,
        downsampling_ratios=TINY_RATIOS,
        channel_multiples=[2, 3],
        decoder_channels=2,
        decoder_input_channels=TINY_LATENT_CHANNELS,
        audio_channels=2,
    )


def _published_weights(
    vae: MLXAutoEncoderOobleck,
) -> tuple[dict[str, mx.array], dict[str, np.ndarray]]:
    """Invent a checkpoint in the layout the VAE is published in.

    Returns ``(checkpoint, fused)``: the first is what ``mx.load`` hands the
    converter -- ``weight_norm``'s ``g``/``v`` pair for every convolution,
    Snake parameters as ``[1, C, 1]``, weights in PyTorch's axis order -- and
    the second is the same weights fused and left in that order, which is
    what the reference decoder takes.

    The shapes come from the MLX model, and the axis orders from PyTorch's
    documented ones (``[out, in, K]`` for a convolution, ``[in, out, K]`` for
    a transposed one), so this inverts the converter without consulting it.
    """
    rng = np.random.default_rng(0)
    checkpoint: dict[str, mx.array] = {}
    fused: dict[str, np.ndarray] = {}

    for key, value in flat_parameters(vae).items():
        if key.endswith((".alpha", ".beta")):
            # Small: exp() of these is the Snake frequency and its inverse
            # amplitude, and a wide draw makes the reference comparison a test
            # of float32 headroom rather than of the conversion.
            values = rng.normal(0.0, 0.2, size=value.shape)
            checkpoint[key] = mx.array(values.reshape(1, -1, 1).astype(np.float32))
            fused[key] = values
        elif key.endswith(".bias"):
            values = rng.normal(0.0, 0.2, size=value.shape)
            checkpoint[key] = mx.array(values.astype(np.float32))
            fused[key] = values
        elif key.endswith(".weight"):
            out_channels, kernel, in_channels = value.shape  # MLX [out, K, in]
            transposed = "conv_t1" in key
            shape = (
                (in_channels, out_channels, kernel)
                if transposed
                else (out_channels, in_channels, kernel)
            )
            direction = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
            gain = rng.uniform(0.2, 0.6, size=(shape[0], 1, 1))
            checkpoint[f"{key}_v"] = mx.array(direction)
            checkpoint[f"{key}_g"] = mx.array(gain.astype(np.float32))
            # w = g * v / ||v||, the norm over every axis but the first.
            norm = np.sqrt(np.square(direction, dtype=np.float64).sum(axis=(1, 2)))
            fused[key] = gain * direction / norm.reshape(-1, 1, 1)
        else:
            raise AssertionError(f"unhandled parameter {key}")

    return checkpoint, fused


def test_the_converter_emits_exactly_the_parameters_the_model_loads():
    """``load_weights`` is strict, and that is the whole of the structural check.

    A key the converter renames, drops or invents is caught here rather than
    at the end of a 337 MB conversion -- and a permutation that changes a
    shape is caught with it.
    """
    vae = _tiny_vae()
    checkpoint, _ = _published_weights(vae)

    converted = convert._convert_vae_weights(checkpoint)

    expected = flat_parameters(vae)
    assert set(converted) == set(expected)
    assert {k: v.shape for k, v in converted.items()} == {
        k: v.shape for k, v in expected.items()
    }
    vae.load_weights(list(converted.items()))


def test_a_converted_vae_decodes_what_the_published_weights_say_it_should():
    """The end of the conversion, measured against the architecture it copies.

    Individually each half is already pinned -- the fusion arithmetic, the
    axis permutations -- but nothing joined them up: a correct permutation
    applied to the wrong tensor, or the Snake parameters squeezed off the
    wrong axis, leaves a model that loads, runs, and decodes to noise. The
    reference implements ``diffusers.AutoencoderOobleck``'s decoder from the
    published weights directly, so agreeing with it is the claim the port
    makes.
    """
    vae = _tiny_vae()
    checkpoint, fused = _published_weights(vae)
    vae.load_weights(list(convert._convert_vae_weights(checkpoint).items()))
    mx.eval(vae.parameters())

    latents = np.random.default_rng(1).normal(size=(1, 9, TINY_LATENT_CHANNELS))
    decoded = np.array(vae.decode(mx.array(latents.astype(np.float32))))
    expected = reference.oobleck_decode(latents, fused, TINY_RATIOS[::-1])

    assert decoded.shape == expected.shape
    assert decoded.shape[1] == latents.shape[1] * np.prod(TINY_RATIOS)
    # Loose for a float32 comparison, deliberately: Metal's transposed
    # convolution accumulates at roughly 1e-3 relative (measured against
    # torch), and there are two of them in this stack, so as shipped this
    # agrees to about 1e-3 of full scale. Everything the test is here to
    # catch is order-1 by comparison -- dropping the weight_norm fusion moves
    # the output by 1e8, permuting the transposed convolution's axes or
    # swapping the two Snake parameters by around 0.6.
    assert np.allclose(decoded, expected, rtol=1e-2, atol=1e-2 * np.abs(expected).max())


# --- windowed decode ------------------------------------------------------

# The published ratios, at a fraction of the width. The ratios are what set
# the decoder's receptive field, and the receptive field is what decides
# whether the overlap is enough -- so they stay exactly as shipped, which also
# keeps ``VAE_HOP`` (and with it the trimming arithmetic) the real one.
PUBLISHED_RATIOS = [2, 4, 4, 6, 10]
CHUNK = 64


@pytest.fixture(scope="module")
def decoder() -> MLXAutoEncoderOobleck:
    vae = MLXAutoEncoderOobleck(
        encoder_hidden_size=8,
        downsampling_ratios=PUBLISHED_RATIOS,
        channel_multiples=[1, 2, 2, 2, 2],
        decoder_channels=4,
        decoder_input_channels=models.LATENT_CHANNELS,
        audio_channels=2,
    )
    randomised(vae, scale=0.1)
    return vae


def _latents(frames: int, seed: int = 7) -> mx.array:
    return mx.random.normal(
        (1, frames, models.LATENT_CHANNELS), key=mx.random.key(seed)
    )


@pytest.mark.parametrize("frames", [200, 137])
def test_decoding_in_windows_returns_what_decoding_the_whole_track_returns(
    decoder, frames
):
    """Bit-identical, which is what the README promises and why this is safe.

    The decoder is fully convolutional, so a window that carries enough
    context on both sides computes the same numbers for its core as the
    un-tiled decode does. It is not an approximation to be traded off against
    memory -- if it were, the seams would be audible every ``CHUNK`` frames,
    which at the shipped size is every twenty seconds.

    The odd length is the case where the last window is short, and where the
    trimming has to stop at the end of the track rather than at a chunk
    boundary.
    """
    latents = _latents(frames)

    whole = np.array(decoder.decode(latents))
    windowed = np.array(
        pipeline.tiled_decode(
            decoder, latents, chunk_frames=CHUNK, overlap=pipeline.DECODE_OVERLAP_FRAMES
        )
    )

    assert whole.shape == (1, frames * models.VAE_HOP, 2)
    assert windowed.shape == whole.shape
    assert np.array_equal(windowed, whole)


def test_a_track_shorter_than_one_window_is_decoded_in_one_go(decoder):
    """No trimming to get wrong, and no concatenate over a single part."""
    latents = _latents(CHUNK - 1)
    windowed = pipeline.tiled_decode(decoder, latents, chunk_frames=CHUNK, overlap=16)
    assert np.array_equal(np.array(windowed), np.array(decoder.decode(latents)))


@pytest.mark.parametrize("overlap", [0, 4])
def test_an_overlap_that_does_not_cover_the_receptive_field_leaves_seams(
    decoder, overlap
):
    """The negative control for the two tests above.

    Without enough context each window's edges are convolved against zeros,
    and the error is concentrated at the joins -- which is exactly what makes
    it audible rather than measurable. Equality above is therefore a fact
    about the overlap, not about tiling being trivially safe: the shipped 64
    frames clears the ~12 this decoder needs by five times over.
    """
    latents = _latents(200)

    whole = np.array(decoder.decode(latents))
    windowed = np.array(
        pipeline.tiled_decode(decoder, latents, chunk_frames=CHUNK, overlap=overlap)
    )

    assert windowed.shape == whole.shape
    assert not np.array_equal(windowed, whole)
