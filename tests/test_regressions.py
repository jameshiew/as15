"""Regressions for the two bugs that produced garbled audio.

Both were silent: the model still emitted plausible-looking audio, so only
listening (or spectral flatness) revealed them.
"""

from __future__ import annotations

import fcntl
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

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
        self.timesteps.append(float(np.array(timestep.astype(mx.float32))[0]))
        rows = mx.arange(hidden_states.shape[0]).reshape(-1, 1, 1)
        # A real decoder answers in its own dtype; an int32 ``arange`` here
        # would promote the result and hide the dtype leaks under test.
        rows = rows.astype(hidden_states.dtype)
        return mx.ones_like(hidden_states) + 0.1 * rows, cache


# shift=1.0 and infer_steps=3 give the schedule [1, 2/3, 1/3]; the interval
# that ends at t=0 is the one the loop used to special-case.
SCHEDULE = [1.0, 2 / 3, 1 / 3]
STEPS = len(SCHEDULE)
NOISE_SHAPE = (1, 4, 8)


def _run_sampler(
    decoder,
    compute_dtype: str = "float32",
    seed: int | None = 0,
    **kwargs,
):
    from as15.mlx.sampler import mlx_generate_diffusion

    b, t, c = NOISE_SHAPE
    return mlx_generate_diffusion(
        decoder,
        encoder_hidden_states_np=np.zeros((b, 3, 6), dtype=np.float32),
        context_latents_np=np.zeros((b, t, c), dtype=np.float32),
        src_latents_shape=(b, t, c),
        seed=seed,
        infer_steps=STEPS,
        shift=1.0,
        dcw_enabled=False,
        disable_tqdm=True,
        compute_dtype=compute_dtype,
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

    _run_sampler(_ConstantDecoder(), infer_method="sde")

    # One initial draw, then one per interval except the one ending at zero.
    assert len(draws) == 1 + (STEPS - 1)


def test_every_sde_noise_draw_comes_from_the_seed(monkeypatch):
    """A seeded request must not touch MLX's implicit global PRNG state."""
    draws = _spy_on_noise_draws(monkeypatch)

    _run_sampler(_ConstantDecoder(), infer_method="sde")

    assert draws and all(draws)


def test_an_unrelated_draw_between_two_sde_runs_does_not_move_the_result():
    """The seed alone has to fix an SDE trajectory.

    The per-interval noise used to come off the implicit global stream, so an
    earlier generation -- or any other ``mx.random`` call in the process --
    shifted every step of the next run at the same seed.
    """
    first = _run_sampler(_ConstantDecoder(), infer_method="sde")["target_latents"]
    mx.eval(mx.random.normal((3, 5)))
    second = _run_sampler(_ConstantDecoder(), infer_method="sde")["target_latents"]

    assert np.array_equal(first, second)


# --- input validation ----------------------------------------------------


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
        _run_sampler(_ConstantDecoder(), infer_method="dpm")


def test_heun_under_sde_reports_the_sampler_that_ran():
    """Heun's corrector is an ODE construction, so SDE steps run Euler.

    The loop warned about the fallback and then reported ``heun`` back to the
    caller anyway, which is what ends up in the timings.
    """
    decoder = _ConstantDecoder()
    result = _run_sampler(decoder, sampler_mode="heun", infer_method="sde")

    assert result["time_costs"]["sampler_mode"] == "euler"
    assert len(decoder.timesteps) == STEPS  # one evaluation per step, not two

    ode = _run_sampler(_ConstantDecoder(), sampler_mode="heun")
    assert ode["time_costs"]["sampler_mode"] == "heun"


def test_asking_for_cfg_without_a_null_embedding_is_rejected():
    """CFG needs something to guide against.

    The loop turned itself on only when both arrived, so a caller that passed a
    scale and no null embedding -- or the null embedding of a distilled
    checkpoint, which ships none -- got the ordinary conditional pass at the
    cost it expected of a guided one, and was told it had guided at 7.0.
    """
    decoder = _ConstantDecoder()
    with pytest.raises(ValueError, match="null_condition_emb_np"):
        _run_sampler(decoder, guidance_scale=7.0)

    assert not decoder.timesteps, "the request was checked after the first step"

    # And the pairing that does work is still accepted.
    _run_sampler(_ConstantDecoder(), **_cfg_kwargs())


@pytest.mark.parametrize("guidance", [0.5, float("nan"), float("inf")])
def test_the_loop_holds_callers_to_the_same_guidance_bound_as_the_cli(guidance):
    """resolve_settings is not on the path of a caller who drives the loop directly."""
    with pytest.raises(ValueError, match="guidance"):
        _run_sampler(
            _ConstantDecoder(), **{**_cfg_kwargs(), "guidance_scale": guidance}
        )


def test_an_unknown_model_is_a_value_error_not_a_process_exit():
    """``resolve`` is a library helper, and used to raise SystemExit.

    A bad name therefore tore down the process of anything embedding the
    package -- a service, a notebook, a test -- with nothing to catch.
    """
    with pytest.raises(ValueError, match="Unknown model"):
        models.resolve("xl-sft-turbo-plus")
    assert models.resolve("xl-sft") is MODELS["xl-sft"]


def test_an_unknown_precision_is_a_value_error_not_a_key_error():
    """``as15 download --precision typo`` used to traceback out of the converter."""
    with pytest.raises(ValueError, match="Unknown precision"):
        convert.resolve_precision("typo")
    assert convert.resolve_precision("bf16") == "bfloat16"
    assert convert.resolve_precision("fp32") == "float32"


def _request(**kwargs):
    """A request that resolve_settings accepts, before *kwargs* spoils it."""
    from as15.pipeline import GenerationRequest

    return GenerationRequest(style_prompt="a song", lyrics="", **kwargs)


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1.0, 5.0, 1e9])
def test_a_duration_the_pipeline_cannot_honour_is_rejected(duration):
    """click's ``min=``/``max=`` are `<`/`>` comparisons, which NaN passes.

    A NaN duration then reached ``int(duration)`` in the metas block and
    ``round(duration * LATENT_FPS)`` in the latent window; an unbounded one
    sized a latent tensor from whatever the caller passed.
    """
    from as15.pipeline import resolve_settings

    with pytest.raises(ValueError, match="duration"):
        resolve_settings(MODELS["xl-sft"], _request(duration=duration))


@pytest.mark.parametrize("guidance", [0.5, 0.0, -10.0, float("nan"), float("inf")])
def test_guidance_that_does_not_mean_what_it_says_is_rejected(guidance):
    """The loop turns CFG on only above 1.0.

    So 0.5 and -10 ran the same conditional-only pass as 1.0 while the banner
    reported the number the caller asked for, and inf went into the guidance
    arithmetic and took the latents with it.
    """
    from as15.pipeline import resolve_settings

    with pytest.raises(ValueError, match="guidance"):
        resolve_settings(MODELS["xl-sft"], _request(guidance=guidance))


def test_guidance_of_exactly_one_is_how_cfg_is_turned_off():
    """The bound above must not reject the documented way to disable CFG."""
    from as15.pipeline import resolve_settings

    assert resolve_settings(MODELS["xl-sft"], _request(guidance=1.0)).guidance == 1.0


def test_a_distilled_checkpoint_reports_the_guidance_it_runs():
    """xl-turbo has no null branch, so CFG is dropped rather than honoured."""
    from as15.pipeline import resolve_settings

    assert resolve_settings(MODELS["xl-turbo"], _request(guidance=7.0)).guidance == 1.0


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_a_seed_outside_the_key_range_is_rejected(seed):
    """``mx.random.key`` takes a uint64 and raises TypeError outside it.

    That happened inside the diffusion loop, minutes in, without naming the
    seed.
    """
    from as15.pipeline import resolve_settings

    with pytest.raises(ValueError, match="seed"):
        resolve_settings(MODELS["xl-sft"], _request(seed=seed))

    with pytest.raises(TypeError):
        mx.random.key(seed)


@pytest.mark.parametrize("bpm", [0, -120, "0", "  "])
def test_a_bpm_that_is_not_a_tempo_is_rejected(bpm):
    """``bpm or 'N/A'`` renders 0 as *unset*; a negative one is written out."""
    from as15.pipeline import resolve_settings

    with pytest.raises(ValueError, match="bpm"):
        resolve_settings(MODELS["xl-sft"], _request(bpm=bpm))


@pytest.mark.parametrize("time_signature", [5, 0, "4/4", "common"])
def test_a_time_signature_the_metas_block_was_not_trained_on_is_rejected(
    time_signature,
):
    """--time-signature has always documented 2, 3, 4 or 6; now it means it."""
    from as15.pipeline import resolve_settings

    with pytest.raises(ValueError, match="time_signature"):
        resolve_settings(MODELS["xl-sft"], _request(time_signature=time_signature))


@pytest.mark.parametrize("time_signature", [2, 3, 4, 6, "4"])
def test_the_documented_time_signatures_are_accepted(time_signature):
    from as15.pipeline import resolve_settings

    resolve_settings(MODELS["xl-sft"], _request(time_signature=time_signature))


def test_settings_come_from_the_checkpoint_when_the_request_omits_them():
    """The CLI banner prints these, so they have to be the ones that run."""
    from as15.pipeline import resolve_settings

    for spec in MODELS.values():
        settings = resolve_settings(spec, _request())
        assert (settings.steps, settings.shift, settings.dcw) == (
            spec.steps,
            spec.shift,
            spec.dcw,
        )
        assert settings.compute_dtype == "bfloat16"


CLI_REJECTS = [
    ["--steps", "0"],
    ["--steps", "-4"],
    ["--shift", "0"],
    ["--shift=-2"],
    ["--shift", "nan"],
    ["--shift", "inf"],
    ["--precision", "typo"],
    ["--sampler", "dpm"],
    ["--duration", "nan"],
    ["--duration", "inf"],
    ["--guidance", "0.5"],
    ["--guidance=-10"],
    ["--guidance", "nan"],
    ["--seed=-1"],
    ["--model", "xl"],
    ["--bpm", "0"],
    ["--time-signature", "5"],
    ["--language", " "],
]


@pytest.mark.parametrize("argv", CLI_REJECTS, ids=lambda a: " ".join(a))
def test_the_cli_rejects_unusable_options(argv, monkeypatch):
    """Every one of these has to fail before generate() fetches ~10 GB.

    generate() is stubbed to say so rather than to let a regression here spend
    a CI run downloading a checkpoint.
    """
    from typer.testing import CliRunner

    from as15 import cli

    def unreachable(*args, **kwargs):
        raise AssertionError(f"generate() ran for {argv}")

    monkeypatch.setattr(cli, "generate", unreachable)

    result = CliRunner().invoke(cli.app, ["sing", "--prompt", "test", *argv])
    assert result.exit_code != 0


def test_the_cli_accepts_the_smallest_valid_step_count(monkeypatch, tmp_path):
    """The bound on --steps must reject 0 without rejecting 1.

    Also a positive control for the cases above: a --steps that no longer
    parses at all would fail those for the wrong reason.
    """
    from typer.testing import CliRunner

    from as15 import cli, pipeline

    seen: dict = {}

    def fake_generate(spec, request, device="auto", progress=True):
        seen["request"] = request
        return pipeline.GenerationResult(
            audio=np.zeros((4, 2), dtype=np.float32), sample_rate=48_000, seed=0
        )

    monkeypatch.setattr(cli, "generate", fake_generate)
    monkeypatch.setattr(cli, "write_audio", lambda *args: None)

    result = CliRunner().invoke(
        cli.app,
        ["sing", "-p", "x", "--steps", "1", "--shift", "2", "-o", str(tmp_path / "a")],
    )

    assert result.exit_code == 0, result.output
    assert seen["request"].steps == 1
    assert seen["request"].shift == 2.0


# --- conditioning token budget -------------------------------------------


class _CharTokenizer:
    """A tokenizer whose tokens are characters -- only the count matters.

    Records how it was called, so the tests can also assert on what was *not*
    asked for.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, text: str, **kwargs):
        self.calls.append(kwargs)
        ids = np.zeros((1, len(text)), dtype=np.int64)
        return SimpleNamespace(input_ids=ids, attention_mask=ids)


def _conditioner_with(tokenizer) -> conditioning.Conditioner:
    """A Conditioner holding a tokenizer and nothing else.

    Built with ``__new__``, so it has no models: reaching the end of the
    budget check would raise AttributeError rather than pass, which is the
    other half of what these tests pin -- input is rejected before the 1.2 B
    torch parameters are run, not after.
    """
    conditioner = conditioning.Conditioner.__new__(conditioning.Conditioner)
    conditioner.tokenizer = tokenizer
    return conditioner


def test_lyrics_the_encoder_cannot_read_are_rejected_not_truncated():
    """Upstream tokenises with ``truncation=True``.

    Lyrics over budget were cut there, and the run then succeeded: the song
    came back missing its last verses with nothing to say so, at the full cost
    of a generation. ``Conditioning.lyrics_text`` kept the whole sheet, so even
    printing what was conditioned on showed the input intact.
    """
    tokenizer = _CharTokenizer()

    with pytest.raises(conditioning.InputTooLong, match="lyric sheet"):
        _conditioner_with(tokenizer).build(
            style_prompt="dream pop", lyrics="l" * 4000, duration=30.0
        )

    assert tokenizer.calls, "the budget was checked without tokenising"
    assert all("truncation" not in kwargs for kwargs in tokenizer.calls)


def test_a_style_prompt_the_encoder_cannot_read_is_rejected():
    """The caption shares its 256 tokens with the instruction and metas lines.

    So the budget is smaller than it looks, and the message has to count what
    the encoder counts rather than what the user typed.
    """
    with pytest.raises(conditioning.InputTooLong, match="style prompt"):
        _conditioner_with(_CharTokenizer()).build(
            style_prompt="p" * 400, lyrics="", duration=30.0
        )


@pytest.mark.parametrize("tokens", [0, 1, 255, 256])
def test_input_that_fits_is_left_alone(tokens):
    """The bound must not reject input the encoder reads in full."""
    conditioning.check_token_budget("the style prompt", "x" * tokens, tokens, 256)


def test_the_rejection_says_how_much_to_cut():
    """Nobody can eyeball where 2048 tokens ends in their lyrics."""
    with pytest.raises(conditioning.InputTooLong) as exc:
        conditioning.check_token_budget("the lyrics", "y" * 4000, 2200, 2048)

    message = str(exc.value)
    assert "2200 tokens" in message
    assert "2048" in message
    assert "152 tokens" in message  # 2200 - 2048
    assert "276 characters" in message  # 152 of 2200 tokens, at 4000 characters


def test_the_budgets_are_the_lengths_upstream_tokenises_to():
    """These are trained lengths, not a limit this port chose."""
    assert conditioning.MAX_PROMPT_TOKENS == 256
    assert conditioning.MAX_LYRIC_TOKENS == 2048


def test_the_cli_reports_input_it_cannot_condition_on(monkeypatch):
    """Only the tokenizer can catch this, and it loads with the conditioner.

    So it is the one bad-input case that gets past the option checks, and it
    has to land as an error rather than as a traceback.
    """
    from typer.testing import CliRunner

    from as15 import cli

    def too_long(*args, **kwargs):
        raise conditioning.InputTooLong("the lyrics ... is 2200 tokens")

    monkeypatch.setattr(cli, "generate", too_long)
    monkeypatch.setattr(cli, "write_audio", lambda *args: None)

    result = CliRunner().invoke(cli.app, ["sing", "-p", "x", "-l", "y"])

    assert result.exit_code == 2
    assert "2200 tokens" in result.stderr
    assert result.exception is None or isinstance(result.exception, SystemExit)


# --- stage cleanup -------------------------------------------------------


class _StageFailure(RuntimeError):
    """Whatever goes wrong inside a stage: an OOM, a bad file, a bug."""


def test_leaving_the_conditioner_block_releases_it(monkeypatch):
    """The pipeline delegates the torch stage's lifetime to the with-block.

    Built with ``__new__`` so the test costs nothing: release() is stubbed and
    a conditioner that never loaded a model has nothing to release anyway.
    """
    released: list[bool] = []
    conditioner = conditioning.Conditioner.__new__(conditioning.Conditioner)
    monkeypatch.setattr(conditioner, "release", lambda: released.append(True))

    with conditioner as entered:
        assert entered is conditioner
    assert released == [True]


def _run_with_stub_stages(
    monkeypatch, log: list[str], fail_in: str | None = None
) -> None:
    """Run generate() with every stage stubbed, failing in one of them.

    *log* collects what the pipeline loaded, ran and handed back, in order, so
    that the cleanup can be asserted on -- including when generate() raises --
    without a checkpoint or 10 GB of weights.
    """
    from as15 import pipeline
    from as15.mlx import sampler

    def stage(name: str) -> None:
        log.append(name)
        if fail_in == name:
            raise _StageFailure(name)

    class FakeConditioner:
        def __init__(self, *args, **kwargs):
            log.append("conditioner loaded")

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self.release()

        def release(self):
            log.append("conditioner released")

        def build(self, **kwargs):
            stage("condition")
            return conditioning.Conditioning(
                encoder_hidden_states=np.zeros((1, 1, 8), np.float32),
                context_latents=np.zeros(
                    (1, 2, 2 * models.LATENT_CHANNELS), np.float32
                ),
                null_condition_emb=np.zeros((1, 1, 8), np.float32),
                latent_frames=2,
                text_prompt="",
                lyrics_text="",
            )

    class FakeVAE:
        def decode(self, latents):
            stage("decode")
            return mx.zeros((1, latents.shape[1] * models.VAE_HOP, 2))

    def fake_diffusion(**kwargs):
        stage("diffuse")
        return {
            "target_latents": np.zeros((1, 2, models.LATENT_CHANNELS), np.float32),
            "time_costs": {},
        }

    snapshot = Snapshot(
        repo_id="fake/repo", revision="0" * 40, path=Path("/nonexistent")
    )
    monkeypatch.setattr(pipeline, "_resolve_snapshots", lambda _: (snapshot, snapshot))
    monkeypatch.setattr(conditioning, "Conditioner", FakeConditioner)
    monkeypatch.setattr(sampler, "mlx_generate_diffusion", fake_diffusion)
    monkeypatch.setattr(pipeline, "_load_dit", lambda *args: log.append("dit loaded"))
    monkeypatch.setattr(pipeline, "_load_vae", lambda *args: FakeVAE())
    # The piece of cleanup with an observable effect: whether the buffers a
    # stage allocated go back to the OS or stay checked out in MLX's cache.
    monkeypatch.setattr(mx, "clear_cache", lambda: log.append("mlx cache cleared"))

    pipeline.generate(MODELS["xl-sft"], _request(), progress=False)


def test_every_stage_hands_its_memory_back_when_it_finishes(monkeypatch):
    """Positive control for the failure cases below.

    Stubs that stopped reaching a stage would pass those vacuously.
    """
    log: list[str] = []
    _run_with_stub_stages(monkeypatch, log)
    assert log == [
        "conditioner loaded",
        "condition",
        "conditioner released",
        "dit loaded",
        "diffuse",
        "mlx cache cleared",
        "decode",
        "mlx cache cleared",
    ]


@pytest.mark.parametrize(
    "fail_in,cleanup",
    [
        ("condition", "conditioner released"),
        ("diffuse", "mlx cache cleared"),
        ("decode", "mlx cache cleared"),
    ],
)
def test_a_stage_hands_its_memory_back_when_it_fails(monkeypatch, fail_in, cleanup):
    """A failed generation must not leave the next one short of memory.

    Cleanup used to sit on the success path only: a conditioning failure kept
    the ~2.4 GB of torch models and the MPS pool, and a failure in diffusion
    or decode left MLX's buffer cache holding the whole attempt. In-process
    callers -- a retry, a service, a test -- then hit an out-of-memory in a
    stage that had nothing to do with the original failure.
    """
    log: list[str] = []
    with pytest.raises(_StageFailure):
        _run_with_stub_stages(monkeypatch, log, fail_in=fail_in)

    assert log[-2:] == [fail_in, cleanup]


# --- compute dtype -------------------------------------------------------


class _DtypeRecordingDecoder(_ConstantDecoder):
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


@pytest.mark.parametrize("sampler_mode", ["euler", "heun"])
@pytest.mark.parametrize("infer_method", ["ode", "sde"])
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
    _run_sampler(
        decoder,
        sampler_mode=sampler_mode,
        infer_method=infer_method,
        compute_dtype="bfloat16",
        **_cfg_kwargs(),
    )

    assert decoder.dtypes
    assert set(decoder.dtypes) == {mx.bfloat16}


def test_latents_come_back_as_numpy_from_a_bf16_loop():
    """numpy has no bfloat16 and MLX refuses the buffer protocol for it.

    Only the float32 leak kept ``np.array(xt)`` working on the default path.
    """
    result = _run_sampler(_ConstantDecoder(), compute_dtype="bfloat16")
    assert result["target_latents"].dtype == np.float32


def test_rope_tables_are_handed_out_in_the_activation_dtype():
    """float32 cos/sin would promote every query and key they multiply.

    transformers' Qwen3RotaryEmbedding -- which upstream uses -- ends its
    forward with ``cos.to(dtype=x.dtype)`` for the same reason.
    """
    from as15.mlx.dit import MLXRotaryEmbedding

    rope = MLXRotaryEmbedding(head_dim=8, max_len=16, base=1e6)
    cos, sin = rope(4, mx.bfloat16)
    assert cos.dtype == sin.dtype == mx.bfloat16
    assert cos.shape == sin.shape == (1, 1, 4, 8)

    # The tables themselves stay float32: the angles need the headroom.
    exact_cos, _ = rope(4, mx.float32)
    assert exact_cos.dtype == mx.float32
    assert np.allclose(np.array(cos.astype(mx.float32)), np.array(exact_cos), atol=1e-2)


def test_a_bf16_decoder_forward_stays_bf16_end_to_end():
    """One float32 input is enough to widen the whole stack.

    Sized down from the XL config; the dtype plumbing is shape-independent.
    """
    from as15.mlx.dit import MLXDiTDecoder

    decoder = MLXDiTDecoder(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        in_channels=12,
        audio_acoustic_hidden_dim=4,
        patch_size=2,
        sliding_window=4,
        max_position_embeddings=64,
        encoder_hidden_size=6,
    )
    decoder.set_dtype(mx.bfloat16)

    def _forward(timestep):
        out, _ = decoder(
            hidden_states=mx.zeros((1, 8, 4), dtype=mx.bfloat16),
            timestep=timestep,
            timestep_r=timestep,
            encoder_hidden_states=mx.zeros((1, 3, 6), dtype=mx.bfloat16),
            context_latents=mx.zeros((1, 8, 8), dtype=mx.bfloat16),
        )
        return out

    assert _forward(mx.full((1,), 0.75, dtype=mx.bfloat16)).dtype == mx.bfloat16
    # Pin the promotion the sampler used to trigger, so the fix is not silently
    # undone by dropping the dtype at the one call site that builds it.
    assert _forward(mx.full((1,), 0.75)).dtype == mx.float32


# --- attention ------------------------------------------------------------


def _small_decoder(num_hidden_layers: int = 2):
    """A decoder sized to the shapes ``_run_sampler`` feeds it.

    ``NOISE_SHAPE`` is (1, 4, 8), so the latents carry 8 channels and the
    context another 8, and the conditioning is 6-wide.
    """
    from as15.mlx.dit import MLXDiTDecoder

    return MLXDiTDecoder(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        in_channels=16,
        audio_acoustic_hidden_dim=8,
        patch_size=2,
        sliding_window=4,
        max_position_embeddings=64,
        encoder_hidden_size=6,
    )


def test_grouped_query_attention_is_not_tiled_out_by_hand(monkeypatch):
    """K/V reach the kernel with their own head count.

    They used to be broadcast up to the query head count first, which on the
    XL config (32 query heads over 8 key/value heads) materialised a fourfold
    copy of every key and value, on both attentions of every layer. MLX's fast
    kernel groups the heads itself, and does it bit-for-bit identically -- so
    the copies bought nothing.
    """
    from as15.mlx import dit

    captured: list[tuple] = []
    real_sdpa = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, *, scale, mask=None):
        captured.append((q, k, v))
        return real_sdpa(q, k, v, scale=scale, mask=mask)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)

    attn = dit.MLXAttention(
        hidden_size=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        attention_bias=False,
        layer_idx=0,
    )
    out = attn(mx.random.normal((1, 6, 16), key=mx.random.key(0)))

    (q, k, v), *rest = captured
    assert not rest
    assert q.shape[1] == 8
    assert k.shape[1] == v.shape[1] == 2

    def _tile(x, n_rep):
        b, h, length, d = x.shape
        return mx.broadcast_to(x[:, :, None], (b, h, n_rep, length, d)).reshape(
            b, h * n_rep, length, d
        )

    tiled = real_sdpa(q, _tile(k, 4), _tile(v, 4), scale=attn.scale, mask=None)
    reference = attn.o_proj(tiled.transpose(0, 2, 1, 3).reshape(1, 6, -1))
    assert np.array_equal(np.array(out), np.array(reference))


def test_the_conditioning_is_projected_once_for_the_whole_run(monkeypatch):
    """Cross-attention K/V come from the conditioning and nothing else.

    No noisy latent and no timestep reaches them, so one cache entry per layer
    survives every step, both of Heun's evaluations and CFG's doubled batch.
    Caching was switched off whenever CFG or Heun was on -- between them, the
    whole default SFT path -- so the conditioning went back through every
    layer's ``k_proj`` and ``v_proj`` on all 2N evaluations.
    """
    from as15.mlx import dit

    updated: list[int] = []
    real_update = dit.MLXCrossAttentionCache.update

    def spy(self, key, value, layer_idx):
        updated.append(layer_idx)
        return real_update(self, key, value, layer_idx)

    monkeypatch.setattr(dit.MLXCrossAttentionCache, "update", spy)

    layers = 2
    forwards = []
    inner = _small_decoder(layers)

    def counting_decoder(**kwargs):
        forwards.append(kwargs["timestep"])
        return inner(**kwargs)

    _run_sampler(counting_decoder, sampler_mode="heun", **_cfg_kwargs())

    assert len(forwards) == 2 * STEPS
    assert updated == list(range(layers))


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


def test_only_one_process_at_a_time_may_write_a_cache(monkeypatch, tmp_path):
    """Two cold starts used to each run the same ~8.3 GB DiT conversion.

    They also raced on one fixed temporary name, so either could rename the
    other's half-written file over the real cache.
    """
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    out = convert.dit_cache_path("m", "a" * 40, "bf16")
    lock = out.with_suffix(".lock")

    with convert._cache_lock(out):
        assert lock.exists()  # and the parent directory was created for it
        # flock is held per open file description, so a second handle behaves
        # exactly like a second process.
        with lock.open("w") as rival, pytest.raises(BlockingIOError):
            fcntl.flock(rival, fcntl.LOCK_EX | fcntl.LOCK_NB)

    # Released on the way out, so the next writer proceeds immediately.
    with lock.open("w") as rival:
        fcntl.flock(rival, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_concurrent_writers_never_share_a_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    out = convert.dit_cache_path("m", "a" * 40, "bf16")
    out.parent.mkdir(parents=True)

    seen: list[Path] = []

    def record(tmp: Path) -> None:
        seen.append(tmp)
        tmp.write_bytes(b"")

    convert._publish(out, record)
    convert._publish(out, record)

    assert seen[0] != seen[1]
    # Same directory, so replace() is a rename and not a copy across devices.
    assert all(tmp.parent == out.parent for tmp in seen)
    # mx.save_safetensors dispatches on the extension and fails with a bare
    # FileNotFoundError if the temporary does not carry it.
    assert all(tmp.name.endswith(".safetensors") for tmp in seen)
    assert out.read_bytes() == b""


def test_a_failed_write_leaves_neither_a_partial_cache_nor_debris(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    out = convert.vae_cache_path("a" * 40)
    out.parent.mkdir(parents=True)

    def explode(tmp: Path) -> None:
        tmp.write_bytes(b"half a tensor")
        raise RuntimeError("no space left on device")

    with pytest.raises(RuntimeError):
        convert._publish(out, explode)

    assert not out.exists()
    assert list(out.parent.iterdir()) == []


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
