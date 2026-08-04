"""The wavelet-domain correction, against an independent Haar transform.

DCW runs on every step of every turbo generation and edits the latent
directly, so a band that goes the wrong way, a lost normalisation or a
swapped filter is heard rather than raised. Upstream reaches this basis
through ``pytorch_wavelets``; this port re-derives it in MLX, and nothing
outside these tests compares the two.

The reference in ``tests/reference.py`` builds the transform as a matrix out
of the Haar filter pair, so it shares no index arithmetic with the strided
slices under test.
"""

from __future__ import annotations

import math

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np
import pytest

from as15.mlx import dcw
from reference import haar_dwt, haar_idwt


def _close(got, expected) -> bool:
    """float32 through an orthonormal transform: 1e-6 is generous.

    Nothing here is an approximation -- Haar is exact in this arithmetic --
    so the tolerance is for accumulation order, not for the transform.
    """
    return bool(np.allclose(got, expected, rtol=1e-5, atol=1e-6))


def _signal(length: int, seed: int = 0, channels: int = 3, batch: int = 2):
    return (
        np.random.default_rng(seed)
        .normal(size=(batch, length, channels))
        .astype(np.float32)
    )


@pytest.mark.parametrize("length", [2, 8, 9, 1, 64, 101])
def test_the_analysis_matches_the_haar_filter_bank(length):
    """Lowpass ``[1, 1]/sqrt(2)``, highpass ``[1, -1]/sqrt(2)``, decimated by two.

    The implementation reads even and odd samples through strided slices,
    which is the same thing said differently -- but only if the stride,
    the offsets and the ``1/sqrt(2)`` all line up. Swap the two bands, or drop
    the normalisation, and the correction below is applied to the wrong
    coefficients at the wrong scale.
    """
    x = _signal(length)
    low, high = dcw._haar_dwt_1d(mx.array(x))
    reference_low, reference_high = haar_dwt(x)

    assert np.array(low).shape == reference_low.shape
    assert _close(np.array(low), reference_low)
    assert _close(np.array(high), reference_high)


@pytest.mark.parametrize("length", [2, 8, 9, 1, 64, 101])
def test_the_synthesis_inverts_the_analysis(length):
    """Perfect reconstruction, which is the property the correction rides on.

    ``apply_mlx_dcw`` transforms, edits and transforms back; whatever it did
    not mean to edit has to come back unchanged, including for an odd length,
    where the right-hand zero pad is added and then dropped again.
    """
    x = _signal(length, seed=1)
    low, high = dcw._haar_dwt_1d(mx.array(x))
    reconstructed = np.array(dcw._haar_idwt_1d(low, high, length))

    assert reconstructed.shape == x.shape
    assert _close(reconstructed, x)
    assert _close(reconstructed, haar_idwt(*haar_dwt(x), length))


def test_the_basis_is_orthonormal():
    """Energy is preserved, so the two scalers mean the same thing on both bands.

    An unnormalised Haar pair (``[1, 1]`` and ``[1, -1]``, no ``1/sqrt(2)``)
    round-trips just as perfectly while doubling the coefficients, which
    silently doubles the correction.
    """
    x = _signal(64, seed=2)
    low, high = (np.array(band) for band in dcw._haar_dwt_1d(mx.array(x)))

    assert (low**2).sum() + (high**2).sum() == pytest.approx((x**2).sum(), rel=1e-5)


def test_the_bands_are_not_swapped():
    """The coarse band is the average and the detail band the difference.

    Both are the same shape, so nothing else here notices if they trade
    places -- but the schedule then weights structure where it means detail,
    which is the whole of what DCW does.
    """
    constant = np.ones((1, 8, 1), dtype=np.float32)
    low, high = dcw._haar_dwt_1d(mx.array(constant))
    assert _close(np.array(low), math.sqrt(2.0))
    assert _close(np.array(high), 0.0)

    alternating = np.array([[[1.0], [-1.0]] * 4], dtype=np.float32)
    low, high = dcw._haar_dwt_1d(mx.array(alternating))
    assert _close(np.array(low), 0.0)
    assert _close(np.array(high), math.sqrt(2.0))


# --- the correction itself ------------------------------------------------


def test_the_scalers_are_the_ones_upstream_tuned():
    """The distilled checkpoints were distilled with these numbers."""
    assert dcw.DCW_SCALER == 0.05
    assert dcw.DCW_HIGH_SCALER == 0.02


@pytest.mark.parametrize("t", [1.0, 0.75, 0.5, 0.25, 0.0])
@pytest.mark.parametrize("length", [16, 17])
def test_the_correction_is_the_weighted_difference_put_back_through_the_basis(
    t, length
):
    """``x + W' diag(s) W (x - denoised)``, with s per band.

    The implementation says it the other way round -- it transforms both
    arrays and edits the coefficients of ``x`` in place -- but the transform
    is linear, so the two agree exactly. Writing the expectation in the form
    the docstring describes is what makes a band weighted by the wrong end of
    the schedule visible: at t the coarse band moves by ``t*0.05`` and the
    detail band by ``(1-t)*0.02``, and swapping them still corrects, just
    backwards along the whole run.
    """
    x = _signal(length, seed=3)
    denoised = _signal(length, seed=4)

    corrected = np.array(dcw.apply_mlx_dcw(mx.array(x), mx.array(denoised), t))

    low, high = haar_dwt(x - denoised)
    expected = x + haar_idwt(
        t * dcw.DCW_SCALER * low, (1.0 - t) * dcw.DCW_HIGH_SCALER * high, length
    )

    assert corrected.shape == x.shape
    assert _close(corrected, expected)


@pytest.mark.parametrize(
    ("t", "untouched"),
    [(1.0, "high"), (0.0, "low")],
)
def test_each_band_is_left_alone_at_its_end_of_the_schedule(t, untouched):
    """One weight is exactly zero at each end, and t=1 is every run's first step.

    The implementation skips the band rather than adding a zero to it, so this
    is also the branch that has to leave the coefficients bit-for-bit alone.
    """
    x = _signal(16, seed=5)
    denoised = _signal(16, seed=6)

    corrected = np.array(dcw.apply_mlx_dcw(mx.array(x), mx.array(denoised), t))
    before = haar_dwt(x)
    after = haar_dwt(corrected)

    index = {"low": 0, "high": 1}[untouched]
    assert _close(after[index], before[index])
    # And the other one did move, so this is not passing on a no-op.
    assert not _close(after[1 - index], before[1 - index])


def test_a_prediction_the_step_already_agrees_with_changes_nothing():
    """The correction is proportional to ``x - denoised``.

    Nothing else in the loop guarantees that, and a constant term slipped in
    here would bias every step of every turbo generation.
    """
    x = _signal(16, seed=7)
    corrected = np.array(dcw.apply_mlx_dcw(mx.array(x), mx.array(x), 0.5))
    assert _close(corrected, x)


def test_the_correction_pushes_away_from_the_predicted_clean_sample():
    """Both scalers are positive, which is what makes this a sharpening.

    A sign error here is the difference between DCW and a low-pass filter,
    and both produce audio.
    """
    x = np.full((1, 16, 1), 1.0, dtype=np.float32)
    denoised = np.zeros((1, 16, 1), dtype=np.float32)

    corrected = np.array(dcw.apply_mlx_dcw(mx.array(x), mx.array(denoised), 0.5))
    assert np.all(corrected > x)
