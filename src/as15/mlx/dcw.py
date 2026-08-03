"""Differential Correction in Wavelet domain (DCW), pure MLX.

Vendored from ACE-Step 1.5 (``acestep.models.mlx.dcw_correction_mlx``) and
reduced to the native Haar path.  Upstream also offers ``db2``/``db4``/
``sym4``/``sym8``/``coif2`` via a per-step ``mx.array -> torch -> mx.array``
bridge into ``pytorch_wavelets``; that bridge is dropped here so the runtime
stays free of torch.  Haar is the upstream default and the only basis whose
MLX implementation is exact.
"""

from __future__ import annotations

import math

import mlx.core as mx

VALID_DCW_MODES = ("low", "high", "double", "pix")


def _haar_dwt_1d(x: mx.array):
    """Single-level Haar DWT along the T axis of a ``[B, T, C]`` array.

    Returns ``(low, high)``, each ``[B, T//2, C]``.  Odd ``T`` is zero-padded
    one sample on the right, mirroring ``pytorch_wavelets``' ``mode='zero'``.
    """
    T = x.shape[1]
    if T % 2 == 1:
        pad = mx.zeros((x.shape[0], 1, x.shape[2]), dtype=x.dtype)
        x = mx.concatenate([x, pad], axis=1)
    even = x[:, 0::2, :]
    odd = x[:, 1::2, :]
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return (even + odd) * inv_sqrt2, (even - odd) * inv_sqrt2


def _haar_idwt_1d(low: mx.array, high: mx.array, out_T: int) -> mx.array:
    """Inverse of :func:`_haar_dwt_1d`; returns an array of length ``out_T``."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    even = (low + high) * inv_sqrt2
    odd = (low - high) * inv_sqrt2
    stacked = mx.stack([even, odd], axis=2)  # [B, T//2, 2, C]
    reconstructed = stacked.reshape(even.shape[0], -1, even.shape[2])
    return reconstructed[:, :out_T, :]


def apply_mlx_dcw(
    x_next: mx.array,
    denoised: mx.array,
    t_curr: float,
    enabled: bool,
    mode: str = "double",
    scaler: float = 0.05,
    high_scaler: float = 0.02,
) -> mx.array:
    """Push ``x_next``'s frequency bands away from the predicted clean sample.

    The scaler decays with ``t_curr``, so this is identity at ``t=0`` and
    strongest at ``t~=1``.
    """
    if not enabled:
        return x_next

    # Per-mode schedule: low decays with t, high is complementary (weak at
    # high noise, strong near t->0), pix uses the raw scaler.
    t = float(t_curr)
    raw_low = float(scaler)
    raw_high = float(high_scaler)
    low_s = t * raw_low
    high_s = (1.0 - t) * raw_low
    double_high_s = (1.0 - t) * raw_high

    if mode == "pix":
        if raw_low == 0.0:
            return x_next
        return x_next + raw_low * (x_next - denoised)

    if mode == "low":
        if low_s == 0.0:
            return x_next
    elif mode == "high":
        if high_s == 0.0:
            return x_next
    elif mode == "double":
        if low_s == 0.0 and double_high_s == 0.0:
            return x_next
    else:
        raise ValueError(
            f"Invalid dcw_mode={mode!r}. Expected one of {VALID_DCW_MODES}."
        )

    T_out = x_next.shape[1]
    xL, xH = _haar_dwt_1d(x_next)
    yL, yH = _haar_dwt_1d(denoised)

    if mode == "low":
        xL = xL + low_s * (xL - yL)
    elif mode == "high":
        xH = xH + high_s * (xH - yH)
    else:  # double
        if low_s != 0.0:
            xL = xL + low_s * (xL - yL)
        if double_high_s != 0.0:
            xH = xH + double_high_s * (xH - yH)

    return _haar_idwt_1d(xL, xH, T_out)
