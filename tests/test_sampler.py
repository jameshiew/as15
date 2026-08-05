"""The diffusion loop: guidance, stepping, noise streams and compute dtype.

Every failure pinned here produced audio. The trajectory came out somewhere
else than the reference's, or in the wrong precision, and the run reported
success -- so these are the tests standing in for a listen.
"""

from __future__ import annotations

import itertools

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np
import pytest

from helpers import (
    NOISE_SHAPE,
    SCHEDULE,
    STEPS,
    ConstantDecoder,
    cfg_kwargs,
    run_sampler,
)

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


# --- stepping ------------------------------------------------------------


def test_heun_corrects_the_interval_that_ends_at_zero():
    """The last interval used to be a bare Euler hop to t=0.

    Upstream pairs ``zip(t[:-1], t[1:])`` over a schedule ending at zero, so
    every interval gets a corrector. Dropping the last one left an N-step Heun
    run at 2N-1 evaluations, first-order exactly where the trajectory lands on
    the clean sample.
    """
    decoder = ConstantDecoder()
    run_sampler(decoder, sampler_mode="heun")

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

    run_sampler(ConstantDecoder(), sampler_mode="heun", **cfg_kwargs())

    assert len(apg_calls) == STEPS
    assert all(state is not None for state in apg_calls)
    assert len(cfg_calls) == STEPS


def test_the_last_interval_still_lands_on_x0():
    """Euler is unchanged by pairing the last interval with t=0.

    The step size there is ``t_last - 0``, so the update stays ``x - v*t``:
    with v == 1 the schedule telescopes to ``noise - t_0``.
    """
    decoder = ConstantDecoder()
    result = run_sampler(decoder, sampler_mode="euler")

    assert np.allclose(decoder.timesteps, SCHEDULE)
    noise = np.array(mx.random.normal(NOISE_SHAPE, key=mx.random.key(0)))
    assert np.allclose(result["target_latents"], noise - SCHEDULE[0], atol=1e-5)


def _spy_on_noise_draws(monkeypatch):
    """Record whether each ``mx.random.normal`` call carried an explicit key."""
    real_normal = mx.random.normal
    keyed: list[bool] = []

    def spy(shape, *args, key=None, **kw):
        keyed.append(key is not None)
        return real_normal(shape, *args, key=key, **kw)

    monkeypatch.setattr(mx.random, "normal", spy)
    return keyed


def test_sde_draws_no_noise_for_the_interval_that_ends_at_zero(monkeypatch):
    """``t_next == 0`` scales the fresh noise away, so it must not be drawn."""
    draws = _spy_on_noise_draws(monkeypatch)

    run_sampler(ConstantDecoder(), infer_method="sde")

    # One initial draw, then one per interval except the one ending at zero.
    assert len(draws) == 1 + (STEPS - 1)


def test_every_sde_noise_draw_comes_from_the_seed(monkeypatch):
    """A seeded request must not touch MLX's implicit global PRNG state."""
    draws = _spy_on_noise_draws(monkeypatch)

    run_sampler(ConstantDecoder(), infer_method="sde")

    assert draws and all(draws)


def test_an_unrelated_draw_between_two_sde_runs_does_not_move_the_result():
    """The seed alone has to fix an SDE trajectory.

    The per-interval noise used to come off the implicit global stream, so an
    earlier generation -- or any other ``mx.random`` call in the process --
    shifted every step of the next run at the same seed.
    """
    first = run_sampler(ConstantDecoder(), infer_method="sde")["target_latents"]
    mx.eval(mx.random.normal((3, 5)))
    second = run_sampler(ConstantDecoder(), infer_method="sde")["target_latents"]

    assert np.array_equal(first, second)


@pytest.mark.parametrize("infer_method", ["ode", "sde"])
def test_a_take_is_the_same_wherever_it_falls_in_a_batch(infer_method):
    """What makes a session's takes the same takes as separate runs.

    ``GenerationSession`` generates several seeds back to back against one
    loaded DiT, in one process. That is only the same thing as running the
    command that many times if a take is a function of its seed alone -- if any
    of the draw came off the implicit global stream, the third take of a batch
    would be a different song from the third take of the same batch re-run
    after listening to the first two.
    """
    alone = run_sampler(ConstantDecoder(), seed=11, infer_method=infer_method)
    run_sampler(ConstantDecoder(), seed=10, infer_method=infer_method)
    after = run_sampler(ConstantDecoder(), seed=11, infer_method=infer_method)

    assert np.array_equal(alone["target_latents"], after["target_latents"])


# --- what the loop refuses -----------------------------------------------


@pytest.mark.parametrize("steps", [0, -1])
def test_a_step_count_below_one_is_rejected(steps):
    """It used to fall through to the fixed 8-step lookup table.

    The CLI prints the step count it was handed before starting, so a request
    for 0 steps announced 0 and then quietly ran 8. With that table gone the
    same request builds an empty schedule instead, and the loop hands back the
    initial noise as if it were a song.
    """
    from as15.mlx.sampler import get_timestep_schedule

    with pytest.raises(ValueError, match="infer_steps"):
        get_timestep_schedule(shift=1.0, infer_steps=steps)


@pytest.mark.parametrize("shift", [0.0, -1.0, -0.5, float("nan"), float("inf")])
def test_a_shift_the_timestep_map_cannot_express_is_rejected(shift):
    """``shift*t / (1+(shift-1)*t)`` divides 0 by 0 at t=1 when shift is 0.

    Negative shifts either hit that same zero denominator part-way down the
    schedule (-1 at t=0.5) or produce a non-monotonic one.
    """
    from as15.mlx.sampler import get_timestep_schedule

    with pytest.raises(ValueError, match="shift"):
        get_timestep_schedule(shift=shift, infer_steps=8)


@pytest.mark.parametrize("shift", [1.0, 2.0, 3.0, 0.5])
def test_an_accepted_shift_gives_a_descending_schedule_inside_the_unit_interval(shift):
    """The property the bound above protects."""
    from as15.mlx.sampler import get_timestep_schedule

    schedule = get_timestep_schedule(shift=shift, infer_steps=16)
    assert schedule[0] == 1.0
    assert all(0.0 < t <= 1.0 for t in schedule)
    assert all(b < a for a, b in itertools.pairwise(schedule))


def test_an_unknown_infer_method_is_rejected():
    """It used to fall through to the Euler ODE branch, silently."""
    with pytest.raises(ValueError, match="infer_method"):
        run_sampler(ConstantDecoder(), infer_method="dpm")


def test_heun_under_sde_is_rejected_rather_than_quietly_downgraded():
    """Heun corrects an ODE step, and an SDE step is not one.

    The loop used to log a warning and run Euler for the whole schedule. The
    warning is the only place that said so: the caller had already been told it
    was sampling with Heun, and the take it wrote recorded ``heun`` as the
    recipe -- which regenerates something else.
    """
    decoder = ConstantDecoder()
    with pytest.raises(ValueError, match="heun"):
        run_sampler(decoder, sampler_mode="heun", infer_method="sde")

    assert not decoder.timesteps, "the pairing was checked after the first step"

    # Each half of it is still fine on its own.
    heun = run_sampler(ConstantDecoder(), sampler_mode="heun")
    assert heun["target_latents"].shape == NOISE_SHAPE
    run_sampler(ConstantDecoder(), infer_method="sde")


def test_asking_for_cfg_without_a_null_embedding_is_rejected():
    """CFG needs something to guide against.

    The loop turned itself on only when both arrived, so a caller that passed a
    scale and no null embedding -- or the null embedding of a distilled
    checkpoint, which ships none -- got the ordinary conditional pass at the
    cost it expected of a guided one, and was told it had guided at 7.0.
    """
    decoder = ConstantDecoder()
    with pytest.raises(ValueError, match="null_condition_emb_np"):
        run_sampler(decoder, guidance_scale=7.0)

    assert not decoder.timesteps, "the request was checked after the first step"

    # And the pairing that does work is still accepted.
    run_sampler(ConstantDecoder(), **cfg_kwargs())


@pytest.mark.parametrize("guidance", [0.5, float("nan"), float("inf")])
def test_the_loop_holds_callers_to_the_same_guidance_bound_as_the_cli(guidance):
    """resolve_request is not on the path of a caller who drives the loop directly."""
    with pytest.raises(ValueError, match="guidance"):
        run_sampler(ConstantDecoder(), **{**cfg_kwargs(), "guidance_scale": guidance})


# --- compute dtype -------------------------------------------------------


class _DtypeRecordingDecoder(ConstantDecoder):
    """Records the dtype of every array the sampler hands the decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.dtypes: list = []

    def __call__(self, **kwargs):
        self.dtypes += [
            kwargs["hidden_states"].dtype,
            kwargs["timestep"].dtype,
            kwargs["timestep_r"].dtype,
        ]
        return super().__call__(**kwargs)


# Every branch of the step, which is three of the four combinations: heun+sde
# is not one the loop runs, and the Euler it used to degrade to under SDE was
# the euler+sde case below.
@pytest.mark.parametrize(
    "sampler_mode,infer_method",
    [("euler", "ode"), ("euler", "sde"), ("heun", "ode")],
)
def test_the_loop_stays_in_the_compute_dtype(sampler_mode, infer_method):
    """``mx.full`` infers float32 from a Python float.

    The timesteps and step sizes the loop broadcasts were built without a
    dtype, and one float32 operand promotes the bf16 array it touches -- so
    the latents left the compute dtype on the first step, and the timestep
    reached the embedding MLP in float32 and carried the promotion through
    AdaLN into every layer. MLX then widened the bf16 weights per op, paying
    both the bandwidth and the peak memory of a float32 4B model.
    """
    decoder = _DtypeRecordingDecoder()
    run_sampler(
        decoder,
        sampler_mode=sampler_mode,
        infer_method=infer_method,
        compute_dtype="bfloat16",
        **cfg_kwargs(),
    )

    assert decoder.dtypes
    assert set(decoder.dtypes) == {mx.bfloat16}


def test_latents_come_back_as_numpy_from_a_bf16_loop():
    """numpy has no bfloat16 and MLX refuses the buffer protocol for it.

    Only the float32 leak kept ``np.array(xt)`` working on the default path.
    """
    result = run_sampler(ConstantDecoder(), compute_dtype="bfloat16")
    assert result["target_latents"].dtype == np.float32
