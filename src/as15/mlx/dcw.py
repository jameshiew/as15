"""Differential Correction in Wavelet domain (DCW), pure MLX.

Vendored from ACE-Step 1.5 (``acestep.models.mlx.dcw_correction_mlx``) and
reduced to the native Haar path.  Upstream also offers ``db2``/``db4``/
``sym4``/``sym8``/``coif2`` via a per-step ``mx.array -> torch -> mx.array``
bridge into ``pytorch_wavelets``; that bridge is dropped here so the per-step
path stays MLX-only.  Haar is the upstream default and the only basis whose
MLX implementation is exact.

Upstream's ``low``/``high``/``pix`` modes go with it.  ``double`` is its
default and the only one anything here ever selected, and the mode arrived as
a free-form string that was validated -- when it was validated at all -- deep
inside the diffusion loop.
"""

from __future__ import annotations

import math

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)

# Correction strengths, from upstream's defaults.  The two bands are weighted
# in opposite directions along the schedule -- see :func:`apply_mlx_dcw`.
DCW_SCALER = 0.05
DCW_HIGH_SCALER = 0.02


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


def apply_mlx_dcw(x_next: mx.array, denoised: mx.array, t_curr: float) -> mx.array:
    """Push ``x_next``'s frequency bands away from the predicted clean sample.

    The low band is weighted by ``t*DCW_SCALER`` and the high band by
    ``(1-t)*DCW_HIGH_SCALER``, so the correction moves from coarse structure at
    the top of the schedule to detail as the trajectory lands.
    """
    t = float(t_curr)
    low_s = t * DCW_SCALER
    high_s = (1.0 - t) * DCW_HIGH_SCALER

    T_out = x_next.shape[1]
    xL, xH = _haar_dwt_1d(x_next)
    yL, yH = _haar_dwt_1d(denoised)

    # Each weight is exactly zero at one end of the schedule -- the high band at
    # t=1, which is the first step of every run -- so skip the band rather than
    # add a zero to it.
    if low_s != 0.0:
        xL = xL + low_s * (xL - yL)
    if high_s != 0.0:
        xH = xH + high_s * (xH - yH)

    return _haar_idwt_1d(xL, xH, T_out)
