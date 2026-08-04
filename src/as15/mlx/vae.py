# Pure MLX port of the decode half of diffusers' AutoencoderOobleck.
#
# Architecture mirrors the PyTorch version exactly:
#   Snake1d -> OobleckResidualUnit -> DecoderBlock -> OobleckDecoder
#   -> MLXOobleckVAE
#
# The encoder half is deliberately absent. Generation goes latents -> audio and
# never the other way: nothing in this package encodes, there is no reference
# implementation to check an encoder against (`tests/reference.py` implements
# the published decoder only), and building one meant constructing ~half the
# VAE's parameters and converting ~half its checkpoint on every run to hold
# weights no forward pass would read. Restoring it means restoring the
# `encoder.*` branch of `convert._convert_vae_weights` with it.
#
# All operations use MLX channels-last (NLC) convention internally.
# The public decode API accepts and returns NLC arrays.

import math

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Snake1d Activation
# ---------------------------------------------------------------------------


class MLXSnake1d(nn.Module):
    """Snake activation: x + (1/beta) * sin(alpha * x)^2.

    Parameters ``alpha`` and ``beta`` are stored as 1-D vectors of shape [C]
    and broadcast over (B, L) automatically.  When ``logscale=True`` (default)
    the actual scale is ``exp(alpha)`` / ``exp(beta)``.
    """

    def __init__(self, hidden_dim: int, logscale: bool = True):
        super().__init__()
        self.alpha = mx.zeros(hidden_dim)
        self.beta = mx.zeros(hidden_dim)
        self.logscale = logscale

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, L, C]  (NLC)
        #
        # ``exp`` is taken in whatever dtype the parameters arrived in, which
        # for this model is always float32: ``convert_vae`` casts the whole VAE
        # to fp32 and the decode is the one stage that does not run at the
        # request's precision. At float16 it would not be -- exp overflows
        # there at alpha > ~11 -- so a half-precision VAE would need this and
        # the two multiplies below widened, not just the weights re-cast.
        alpha = mx.exp(self.alpha) if self.logscale else self.alpha
        beta = mx.exp(self.beta) if self.logscale else self.beta
        # All ops broadcast [C] over [B, L, C]
        return x + mx.reciprocal(beta + 1e-9) * mx.power(mx.sin(alpha * x), 2)


# ---------------------------------------------------------------------------
# Residual Unit
# ---------------------------------------------------------------------------


class MLXOobleckResidualUnit(nn.Module):
    """Two weight-normalised Conv1d layers (k=7 dilated + k=1) wrapped with
    Snake1d activations and a residual skip connection."""

    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2

        self.snake1 = MLXSnake1d(dimension)
        self.conv1 = nn.Conv1d(
            dimension, dimension, kernel_size=7, dilation=dilation, padding=pad
        )
        self.snake2 = MLXSnake1d(dimension)
        self.conv2 = nn.Conv1d(dimension, dimension, kernel_size=1)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        # hidden_state: [B, L, C]
        output = self.conv1(self.snake1(hidden_state))
        output = self.conv2(self.snake2(output))

        # Safety trim (should be no-op with correct padding)
        padding = (hidden_state.shape[1] - output.shape[1]) // 2
        if padding > 0:
            hidden_state = hidden_state[:, padding:-padding, :]

        return hidden_state + output


# ---------------------------------------------------------------------------
# Decoder Block
# ---------------------------------------------------------------------------


class MLXOobleckDecoderBlock(nn.Module):
    """Snake -> strided ConvTranspose1d up -> 3 residual units (dilations 1, 3, 9)."""

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.snake1 = MLXSnake1d(input_dim)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.res_unit1 = MLXOobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = MLXOobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = MLXOobleckResidualUnit(output_dim, dilation=9)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.res_unit3(hidden_state)
        return hidden_state


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class MLXOobleckDecoder(nn.Module):
    """Oobleck Decoder: Conv1d -> N decoder blocks -> Snake -> Conv1d."""

    def __init__(
        self,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios: list[int],
        channel_multiples: list[int],
    ):
        super().__init__()
        strides = upsampling_ratios
        cm = [1, *channel_multiples]

        self.conv1 = nn.Conv1d(
            input_channels, channels * cm[-1], kernel_size=7, padding=3
        )

        self.block = []
        for i, stride in enumerate(strides):
            self.block.append(
                MLXOobleckDecoderBlock(
                    input_dim=channels * cm[len(strides) - i],
                    output_dim=channels * cm[len(strides) - i - 1],
                    stride=stride,
                )
            )

        self.snake1 = MLXSnake1d(channels)
        self.conv2 = nn.Conv1d(
            channels, audio_channels, kernel_size=7, padding=3, bias=False
        )

    def __call__(self, hidden_state: mx.array) -> mx.array:
        hidden_state = self.conv1(hidden_state)
        for layer in self.block:
            hidden_state = layer(hidden_state)
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state


# ---------------------------------------------------------------------------
# The VAE, as far as generation uses it
# ---------------------------------------------------------------------------


class MLXOobleckVAE(nn.Module):
    """The decode half of ``diffusers.AutoencoderOobleck``, in pure MLX.

    A module rather than the bare :class:`MLXOobleckDecoder` because the
    checkpoint publishes the decoder one level down, under ``decoder.*``: the
    nesting is what lets the converted cache keep the published names instead
    of renaming every key on the way in.

    Every geometry parameter is required, deliberately. ACE-Step 1.5 is not
    configured like the Stable Audio VAE it descends from -- it downsamples by
    [2, 4, 4, 6, 10] (hop 1920) at 48 kHz, where Stable Audio uses
    [2, 4, 4, 8, 8] (hop 2048) at 44.1 kHz -- so any default here would be a
    second, contradictory source of truth for the latent rate that the rest of
    the runtime pins in ``models.LATENT_FPS``. Callers pass the checkpoint's
    ``config.json`` through instead, and ``models.check_vae_geometry`` checks
    that config against those constants before construction.

    Data flows in NLC (batch, length, channels) format throughout.
    """

    def __init__(
        self,
        downsampling_ratios: list[int],
        channel_multiples: list[int],
        decoder_channels: int,
        decoder_input_channels: int,
        audio_channels: int,
    ):
        super().__init__()
        # Named for the config field, which describes the encoder: the decoder
        # walks the same ratios back up, so this is the one place they reverse.
        self.decoder = MLXOobleckDecoder(
            channels=decoder_channels,
            input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            upsampling_ratios=downsampling_ratios[::-1],
            channel_multiples=channel_multiples,
        )

    def decode(self, latents_nlc: mx.array) -> mx.array:
        """Decode latents -> audio.

        Args:
            latents_nlc: [B, L_latent, C_latent] in NLC format.

        Returns:
            audio: [B, L_audio, C_audio] in NLC format.
        """
        return self.decoder(latents_nlc)
