"""Independent numpy implementations of what the MLX modules compute.

These exist so a test can say *what the answer is* rather than *what the code
currently returns*. They are written from the definitions -- PyTorch's
documented convolution semantics, the Haar filter bank -- and not from the MLX
source, so a permutation, a padding or a band that goes the wrong way shows up
as a mismatch instead of being copied into the expectation.

Everything here works in float64 on NLC (batch, length, channels) arrays and
takes weights in *PyTorch* layout, which is the layout the checkpoints are
published in. ``tests/test_vae.py`` checks the two convolutions against
``torch.nn.functional`` itself, so the reference is pinned to the reference.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Convolutions, per torch.nn.functional
# ---------------------------------------------------------------------------


def conv1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> np.ndarray:
    """``F.conv1d`` on NLC arrays; *weight* is ``[out, in, K]``.

    Cross-correlation, not convolution: tap ``k`` reads input position
    ``i*stride + k*dilation``, with the kernel unflipped, which is what torch
    (and every other framework) means by "conv".
    """
    padded = np.pad(x, ((0, 0), (padding, padding), (0, 0)))
    kernel_span = dilation * (weight.shape[2] - 1) + 1
    length = (padded.shape[1] - kernel_span) // stride + 1
    starts = np.arange(length) * stride

    out = np.zeros((x.shape[0], length, weight.shape[0]))
    for k in range(weight.shape[2]):
        out += padded[:, starts + k * dilation, :] @ weight[:, :, k].T
    return out if bias is None else out + bias


def conv_transpose1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """``F.conv_transpose1d`` on NLC arrays; *weight* is ``[in, out, K]``.

    The transpose of :func:`conv1d`: every input position scatters the kernel
    into the output at ``i*stride``, and *padding* is then cropped off both
    ends -- the reverse of adding it, hence the name.
    """
    full = np.zeros(
        (x.shape[0], (x.shape[1] - 1) * stride + weight.shape[2], weight.shape[1])
    )
    starts = np.arange(x.shape[1]) * stride
    for k in range(weight.shape[2]):
        full[:, starts + k, :] += x @ weight[:, :, k]

    out = full[:, padding : full.shape[1] - padding, :]
    return out if bias is None else out + bias


# ---------------------------------------------------------------------------
# Oobleck decoder
# ---------------------------------------------------------------------------


def snake(x: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """``x + sin(exp(alpha) * x)^2 / exp(beta)``, per-channel.

    The 1e-9 matches the MLX module's guard against a beta of -inf; at the
    magnitudes these tests use it is far below float32 resolution.
    """
    return x + np.sin(np.exp(alpha) * x) ** 2 / (np.exp(beta) + 1e-9)


def _residual_unit(x: np.ndarray, w: dict, prefix: str, dilation: int) -> np.ndarray:
    y = conv1d(
        snake(x, w[f"{prefix}.snake1.alpha"], w[f"{prefix}.snake1.beta"]),
        w[f"{prefix}.conv1.weight"],
        w[f"{prefix}.conv1.bias"],
        padding=((7 - 1) * dilation) // 2,
        dilation=dilation,
    )
    y = conv1d(
        snake(y, w[f"{prefix}.snake2.alpha"], w[f"{prefix}.snake2.beta"]),
        w[f"{prefix}.conv2.weight"],
        w[f"{prefix}.conv2.bias"],
    )
    return x + y


def oobleck_decode(
    latents: np.ndarray, w: dict, upsampling_ratios: list[int]
) -> np.ndarray:
    """Decode as ``diffusers.AutoencoderOobleck``'s decoder does.

    *w* maps ``decoder.*`` names to fused, PyTorch-layout weights -- what the
    checkpoint holds once ``weight_norm``'s ``g``/``v`` pair is combined. The
    structure (a k=7 input conv, then per ratio a Snake, a transposed conv and
    three residual units at dilations 1/3/9, then a Snake and a bias-free k=7
    output conv) is the wiring the MLX port claims to mirror.
    """
    x = conv1d(latents, w["decoder.conv1.weight"], w["decoder.conv1.bias"], padding=3)

    for i, stride in enumerate(upsampling_ratios):
        block = f"decoder.block.{i}"
        x = snake(x, w[f"{block}.snake1.alpha"], w[f"{block}.snake1.beta"])
        x = conv_transpose1d(
            x,
            w[f"{block}.conv_t1.weight"],
            w[f"{block}.conv_t1.bias"],
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        for unit, dilation in enumerate([1, 3, 9], start=1):
            x = _residual_unit(x, w, f"{block}.res_unit{unit}", dilation)

    x = snake(x, w["decoder.snake1.alpha"], w["decoder.snake1.beta"])
    return conv1d(x, w["decoder.conv2.weight"], padding=3)


# ---------------------------------------------------------------------------
# Haar wavelet transform
# ---------------------------------------------------------------------------


def _haar_bank(length: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Analysis matrices for a single-level Haar DWT of *length* samples.

    Built from the orthonormal filter pair rather than from index arithmetic:
    the lowpass ``[1, 1]/sqrt(2)`` and highpass ``[1, -1]/sqrt(2)`` are laid
    into successive rows at stride 2, which is what "convolve then decimate by
    two" means. Odd lengths are zero-padded on the right, matching
    ``pytorch_wavelets``' ``mode='zero'``.
    """
    padded = length + (length % 2)
    lowpass = np.array([1.0, 1.0]) / math.sqrt(2.0)
    highpass = np.array([1.0, -1.0]) / math.sqrt(2.0)

    low = np.zeros((padded // 2, padded))
    high = np.zeros((padded // 2, padded))
    for row in range(padded // 2):
        low[row, 2 * row : 2 * row + 2] = lowpass
        high[row, 2 * row : 2 * row + 2] = highpass
    return low, high, padded


def haar_dwt(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-level Haar DWT along the T axis of ``[B, T, C]``."""
    low, high, padded = _haar_bank(x.shape[1])
    x = np.pad(x, ((0, 0), (0, padded - x.shape[1]), (0, 0)))
    return np.einsum("nt,btc->bnc", low, x), np.einsum("nt,btc->bnc", high, x)


def haar_idwt(low: np.ndarray, high: np.ndarray, out_length: int) -> np.ndarray:
    """Inverse of :func:`haar_dwt`, truncated to *out_length*.

    The basis is orthonormal, so synthesis is the transpose of analysis.
    """
    analysis_low, analysis_high, _ = _haar_bank(low.shape[1] * 2)
    reconstructed = np.einsum("nt,bnc->btc", analysis_low, low) + np.einsum(
        "nt,bnc->btc", analysis_high, high
    )
    return reconstructed[:, :out_length, :]
