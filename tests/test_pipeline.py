"""What a request is allowed to say, and what a run leaves behind.

The three things here all cost a whole generation when they go wrong: a
setting that is quietly reinterpreted, a stage that keeps its memory after it
fails, and an output path that turns out to be unwritable once the audio
exists.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np
import pytest

from as15 import conditioning, models, pipeline
from as15.models import MODELS, Snapshot
from helpers import flac_comments, request

# --- resolving a request --------------------------------------------------


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1.0, 5.0, 1e9])
def test_a_duration_the_pipeline_cannot_honour_is_rejected(duration):
    """click's ``min=``/``max=`` are `<`/`>` comparisons, which NaN passes.

    A NaN duration then reached ``int(duration)`` in the metas block and
    ``round(duration * LATENT_FPS)`` in the latent window; an unbounded one
    sized a latent tensor from whatever the caller passed.
    """
    with pytest.raises(ValueError, match="duration"):
        pipeline.resolve_request(MODELS["xl-sft"], request(duration=duration))


# --- how long the take is -------------------------------------------------
#
# The latent grid is 40 ms, so a duration in seconds is a request rather than a
# length. It used to be answered three times over: the metas block floored it,
# the latent window rounded it, and the banner and the tags printed what was
# typed. A 12.9 s request told the model 12 seconds, generated 12.92, and wrote
# a file claiming 12.9 -- and lyric pacing is conditioned on the first of those.


@pytest.mark.parametrize(
    ("duration", "frames"),
    [(10.0, 250), (12.9, 323), (120.0, 3000), (120.5, 3013), (599.99, 15000)],
)
def test_a_duration_becomes_one_frame_count_with_halves_rounding_up(duration, frames):
    """``round`` sends a half to the even neighbour, which is not a length.

    120.5 s went *down* to 3012 frames while 121.5 s went up to 3038: the same
    half-frame of asking, resolved in two directions.
    """
    assert models.latent_frames_for(duration) == frames
    assert (
        pipeline.resolve_request(
            MODELS["xl-sft"], request(duration=duration)
        ).latent_frames
        == frames
    )


def test_the_latent_window_is_never_empty():
    """``resolve_request`` rejects anything this short; the floor is a backstop."""
    assert models.latent_frames_for(0.0) == 1
    assert models.latent_frames_for(0.01) == 1


def test_every_stage_is_told_the_length_that_will_actually_be_generated():
    """One resolution, and the four places that used to disagree read it.

    The frame count is what the DiT generates and the VAE decodes, the seconds
    are what those frames are worth, and the metas duration is that rounded to
    the whole seconds the block was trained in -- 13, not the 12 that flooring
    the request produced.
    """
    resolved = pipeline.resolve_request(MODELS["xl-sft"], request(duration=12.9))

    assert resolved.latent_frames == 323
    assert resolved.duration == 323 / models.LATENT_FPS == 12.92
    assert resolved.metas_duration == 13

    metas = conditioning.format_metas(None, None, None, resolved.metas_duration)
    assert "- duration: 13 seconds\n" in metas
    assert pipeline.describe(MODELS["xl-sft"], resolved)["AS15_DURATION"] == "12.92"


def test_the_duration_a_take_records_resolves_back_to_that_take():
    """The tags are a recipe, so AS15_DURATION has to be re-typeable.

    Recording the length of the audio rather than the request is only an
    improvement if handing it back produces the same frame count -- ``%g`` and
    the 40 ms grid have to round-trip over the whole allowed range, not just at
    the two ends of it.
    """
    lo = models.latent_frames_for(pipeline.MIN_DURATION)
    hi = models.latent_frames_for(pipeline.MAX_DURATION)
    for frames in range(lo, hi + 1):
        recorded = f"{models.seconds_for(frames):g}"
        assert models.latent_frames_for(float(recorded)) == frames, recorded


@pytest.mark.parametrize("guidance", [0.5, 0.0, -10.0, float("nan"), float("inf")])
def test_guidance_that_does_not_mean_what_it_says_is_rejected(guidance):
    """The loop turns CFG on only above 1.0.

    So 0.5 and -10 ran the same conditional-only pass as 1.0 while the banner
    reported the number the caller asked for, and inf went into the guidance
    arithmetic and took the latents with it.
    """
    with pytest.raises(ValueError, match="guidance"):
        pipeline.resolve_request(MODELS["xl-sft"], request(guidance=guidance))


def test_guidance_of_exactly_one_is_how_cfg_is_turned_off():
    """The bound above must not reject the documented way to disable CFG."""
    resolved = pipeline.resolve_request(MODELS["xl-sft"], request(guidance=1.0))
    assert resolved.guidance == 1.0


def test_a_distilled_checkpoint_reports_the_guidance_it_runs():
    """xl-turbo has no null branch, so CFG is dropped rather than honoured."""
    resolved = pipeline.resolve_request(MODELS["xl-turbo"], request(guidance=7.0))
    assert resolved.guidance == 1.0


def test_a_sampler_the_infer_method_cannot_run_is_rejected():
    """heun+sde ran Euler, and every report of the run said heun.

    The downgrade lived at the bottom of the diffusion loop, past the point
    where anything could still be said about it: the banner had printed heun,
    and ``describe`` reads the resolved request, so the take carried
    ``AS15_SAMPLER=heun`` -- a recipe that regenerates a different take. It is
    a pairing nobody can have meant, so it is refused up here with the rest of
    them, before a byte is fetched.
    """
    with pytest.raises(ValueError, match="heun"):
        pipeline.resolve_request(
            MODELS["xl-sft"], request(sampler="heun", infer_method="sde")
        )

    # Neither half is a mistake by itself.
    for spoiled in (
        {"sampler": "heun", "infer_method": "ode"},
        {"sampler": "euler", "infer_method": "sde"},
    ):
        resolved = pipeline.resolve_request(MODELS["xl-sft"], request(**spoiled))
        assert (resolved.sampler, resolved.infer_method) == (
            spoiled["sampler"],
            spoiled["infer_method"],
        )


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_a_seed_outside_the_key_range_is_rejected(seed):
    """``mx.random.key`` takes a uint64 and raises TypeError outside it.

    That happened inside the diffusion loop, minutes in, without naming the
    seed.
    """
    with pytest.raises(ValueError, match="seed"):
        pipeline.resolve_request(MODELS["xl-sft"], request(seed=seed))

    with pytest.raises(TypeError):
        mx.random.key(seed)


@pytest.mark.parametrize("bpm", [0, -120, "0", "  "])
def test_a_bpm_that_is_not_a_tempo_is_rejected(bpm):
    """``bpm or 'N/A'`` renders 0 as *unset*; a negative one is written out."""
    with pytest.raises(ValueError, match="bpm"):
        pipeline.resolve_request(MODELS["xl-sft"], request(bpm=bpm))


@pytest.mark.parametrize("time_signature", [5, 0, "4/4", "common"])
def test_a_time_signature_the_metas_block_was_not_trained_on_is_rejected(
    time_signature,
):
    """--time-signature has always documented 2, 3, 4 or 6; now it means it."""
    with pytest.raises(ValueError, match="time_signature"):
        pipeline.resolve_request(
            MODELS["xl-sft"], request(time_signature=time_signature)
        )


@pytest.mark.parametrize("time_signature", [2, 3, 4, 6, "4"])
def test_the_documented_time_signatures_are_accepted(time_signature):
    pipeline.resolve_request(MODELS["xl-sft"], request(time_signature=time_signature))


def test_settings_come_from_the_checkpoint_when_the_request_omits_them():
    """The CLI banner prints these, so they have to be the ones that run."""
    for spec in MODELS.values():
        resolved = pipeline.resolve_request(spec, request())
        assert (resolved.steps, resolved.shift, resolved.dcw) == (
            spec.steps,
            spec.shift,
            spec.dcw,
        )
        assert resolved.compute_dtype == "bfloat16"


def test_what_the_request_already_answered_is_carried_through_untouched():
    """The stages downstream see only the resolved request.

    Anything the request settled itself has to arrive there intact -- a field
    dropped or mistyped in the resolution is a run conditioned on, or sampled
    with, a value nobody asked for.
    """
    req = request(
        style_prompt="dream pop",
        lyrics="[verse]\nCity lights",
        language="fr",
        bpm=110,
        key_scale="C minor",
        time_signature=3,
        seed=7,
        sampler="heun",
        infer_method="ode",
        precision="fp32",
    )
    resolved = pipeline.resolve_request(MODELS["xl-sft"], req)

    for field in (
        "style_prompt",
        "lyrics",
        "language",
        "bpm",
        "key_scale",
        "time_signature",
        "seed",
        "sampler",
        "infer_method",
        "precision",
    ):
        assert getattr(resolved, field) == getattr(req, field), field


# --- stage cleanup -------------------------------------------------------


class _StageFailure(RuntimeError):
    """Whatever goes wrong inside a stage: an OOM, a bad file, a bug."""


def _run_with_stub_stages(
    monkeypatch,
    log: list[str],
    fail_in: str | None = None,
    seen: dict | None = None,
    **kwargs,
) -> None:
    """Run generate() with every stage stubbed, failing in one of them.

    *log* collects what the pipeline loaded, ran and handed back, in order, so
    that the cleanup can be asserted on -- including when generate() raises --
    without a checkpoint or 10 GB of weights. *seen*, if given, collects what
    each stage was told; *kwargs* spoil the request the run is made with.
    """
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

        def build(self, request, instruction=None):
            stage("condition")
            if seen is not None:
                seen["conditioned"] = request
            frames = request.latent_frames
            return conditioning.Conditioning(
                encoder_hidden_states=np.zeros((1, 1, 8), np.float32),
                context_latents=np.zeros(
                    (1, frames, 2 * models.LATENT_CHANNELS), np.float32
                ),
                null_condition_emb=np.zeros((1, 1, 8), np.float32),
                latent_frames=frames,
                text_prompt="",
                lyrics_text="",
            )

    class FakeVAE:
        def decode(self, latents):
            stage("decode")
            return mx.zeros((1, latents.shape[1] * models.VAE_HOP, 2))

    def fake_diffusion(**called_with):
        stage("diffuse")
        if seen is not None:
            seen["sampled"] = {
                "src_latents_shape": called_with["src_latents_shape"],
                "shift": called_with["shift"],
                "steps": called_with["infer_steps"],
                "guidance": called_with["guidance_scale"],
                "seed": called_with["seed"],
                "sampler": called_with["sampler_mode"],
                "infer_method": called_with["infer_method"],
                "dcw": called_with["dcw_enabled"],
            }
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

    pipeline.generate(MODELS["xl-sft"], request(**kwargs), progress=False)


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


# --- writing the output ---------------------------------------------------
#
# Everything a generation is for arrives in the last two seconds of a run that
# took minutes, so the questions "can this be written?" and "is this worth
# writing?" are asked before and around the write rather than left to
# whatever soundfile does with a bad argument.


def _tone(seconds: float = 0.05, channels: int = 2) -> np.ndarray:
    """A short sine, so a round trip has something to compare."""
    t = np.arange(int(seconds * models.SAMPLE_RATE)) / models.SAMPLE_RATE
    wave = 0.5 * np.sin(2 * np.pi * 440.0 * t, dtype=np.float32)
    return np.repeat(wave[:, None], channels, axis=1)


@pytest.mark.parametrize("name", ["song.mp3", "song.wav", "song", "song.flac.bak"])
def test_a_container_that_is_not_written_is_rejected(name, tmp_path):
    """soundfile takes the format from the extension and writes ~27 of them.

    Only one is a good answer for a lossless 48 kHz master, so the others are
    a typo rather than a request -- and a typo used to be discovered by
    soundfile, after the generation.
    """
    with pytest.raises(ValueError, match="extension"):
        pipeline.check_output_path(tmp_path / name)


def test_the_extension_is_matched_case_insensitively(tmp_path):
    pipeline.check_output_path(tmp_path / "SONG.FLAC")


def test_a_directory_is_not_something_to_write_a_song_over(tmp_path):
    target = tmp_path / "song.flac"
    target.mkdir()
    with pytest.raises(ValueError, match="directory"):
        pipeline.check_output_path(target)


def test_an_output_directory_that_does_not_exist_yet_is_allowed(tmp_path):
    """write_audio creates it -- and the preflight leaves it to write_audio.

    A preflight that created directories itself would litter one per failed
    generation, which is every run that dies in conditioning or diffusion.
    """
    pipeline.check_output_path(tmp_path / "takes" / "tuesday" / "song.flac")
    assert list(tmp_path.iterdir()) == []


def test_a_file_where_the_output_directory_should_be_is_rejected(tmp_path):
    (tmp_path / "takes").write_text("not a directory")
    with pytest.raises(ValueError, match="is a file"):
        pipeline.check_output_path(tmp_path / "takes" / "song.flac")


@pytest.mark.skipif(os.geteuid() == 0, reason="root may write anywhere")
def test_a_directory_that_cannot_be_written_to_is_rejected(tmp_path):
    """Permission is asked of the directory, which is what the rename needs."""
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        with pytest.raises(ValueError, match="not writable"):
            pipeline.check_output_path(locked / "song.flac")
    finally:
        locked.chmod(0o700)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_audio_that_is_not_finite_is_refused_rather_than_laundered(bad, tmp_path):
    """The peak limiter turns both into a plausible-looking file.

    A NaN fails ``peak > 0.999`` and is written through untouched; an
    infinity makes the scale factor 0.999/inf == 0, which multiplies the
    whole track down to zeros and NaNs. Both produce a file that plays.
    """
    audio = _tone()
    audio[7, 0] = bad
    with pytest.raises(ValueError, match="NaN or infinite"):
        pipeline.write_audio(tmp_path / "song.flac", audio, models.SAMPLE_RATE)
    assert list(tmp_path.iterdir()) == []


def test_the_written_file_reads_back_as_what_was_generated(tmp_path):
    """Positive control: the checks above must not have broken the write."""
    import soundfile as sf

    audio = _tone()
    out = tmp_path / "takes" / "song.flac"
    pipeline.write_audio(out, audio, models.SAMPLE_RATE)

    read, rate = sf.read(str(out), dtype="float32", always_2d=True)
    assert rate == models.SAMPLE_RATE
    assert read.shape == audio.shape
    # The container is 16-bit, so a sample can move by half a step and no more.
    assert np.abs(read - audio).max() <= 2**-15


def test_a_decode_that_overshot_full_scale_is_brought_back_under_it(tmp_path):
    """16-bit conversion wraps rather than clips, so a peak above 1.0 inverts.

    Scaling is conditional -- audio that already fits is written through
    untouched, because rescaling a quiet take would change what was
    generated.
    """
    import soundfile as sf

    loud = _tone() * 4.0
    out = tmp_path / "loud.flac"
    pipeline.write_audio(out, loud, models.SAMPLE_RATE)
    read, _ = sf.read(str(out), dtype="float32", always_2d=True)
    assert np.abs(read).max() <= 0.999
    # The shape of the waveform survives; only its level moved.
    assert np.corrcoef(read[:, 0], loud[:, 0])[0, 1] > 0.999

    quiet = _tone() * 0.25
    out = tmp_path / "quiet.flac"
    pipeline.write_audio(out, quiet, models.SAMPLE_RATE)
    read, _ = sf.read(str(out), dtype="float32", always_2d=True)
    assert np.abs(read - quiet).max() <= 2**-15


def test_a_failed_write_leaves_the_previous_take_intact(monkeypatch, tmp_path):
    """The write used to encode straight into the destination.

    A full disk, a crash or a ^C part way through therefore destroyed
    whatever was already there and left a truncated file in its place --
    with the generation that would have replaced it gone too.
    """
    import soundfile as sf

    out = tmp_path / "song.flac"
    pipeline.write_audio(out, _tone(), models.SAMPLE_RATE)
    before = out.read_bytes()

    def die(path, *args, **kwargs):
        Path(path).write_bytes(b"half a song")
        raise RuntimeError("no space left on device")

    monkeypatch.setattr(sf, "write", die)
    with pytest.raises(RuntimeError):
        pipeline.write_audio(out, _tone(), models.SAMPLE_RATE)

    assert out.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["song.flac"]


@pytest.mark.parametrize("channels", [1, 2, 6])
def test_the_channel_count_is_whatever_the_decode_produced(channels, tmp_path):
    """Nothing between the VAE config and the write requires two.

    The check that the checkpoint's geometry is the expected one lives in
    ``check_vae_geometry``; the writer's job is to write what it was given,
    not to assume a layout and silently mix down.
    """
    import soundfile as sf

    audio = _tone(channels=channels)
    out = tmp_path / "song.flac"
    pipeline.write_audio(out, audio, models.SAMPLE_RATE)

    read, _ = sf.read(str(out), dtype="float32", always_2d=True)
    assert read.shape == (audio.shape[0], channels)


# --- what the file says it is ---------------------------------------------
#
# The prompt, the lyrics and the seed exist only in the shell history that
# produced a take, which is to say until the next reboot. These put them in the
# file. ``as15.flac`` covers the block itself; what is checked here is that the
# recipe recorded is the one that ran.


def _tags(spec, **kwargs) -> dict[str, str]:
    return pipeline.describe(spec, pipeline.resolve_request(spec, request(**kwargs)))


def test_the_tags_record_the_settings_that_ran_not_the_ones_asked_for():
    """Same argument as the banner: a request's ``None`` describes no run.

    A take generated with the checkpoint's defaults would otherwise record
    nothing about steps or shift, and a turbo take asked for CFG would record
    a guidance the sampler never used.
    """
    tags = _tags(MODELS["xl-turbo"], guidance=7.0, seed=99)

    assert tags["AS15_STEPS"] == "8"  # xl-turbo's default, not the request's None
    assert tags["AS15_SHIFT"] == "3"
    assert tags["AS15_DCW"] == "on"
    assert tags["AS15_GUIDANCE"] == "1"  # distilled: CFG was dropped
    assert tags["AS15_SEED"] == "99"


def test_the_tags_name_the_commit_the_weights_came_from():
    """Upstream force-pushes under the same repo ID, so the ID is not an answer."""
    spec = MODELS["xl-sft"]
    assert _tags(spec)["AS15_CHECKPOINT"] == f"{spec.repo_id}@{spec.revision}"


def test_an_instrumental_carries_no_lyrics_field():
    """Absent and empty are different claims; players render the second blank."""
    spec = MODELS["xl-sft"]
    assert "LYRICS" not in _tags(spec)

    assert _tags(spec, lyrics="[verse]\nOh")["LYRICS"] == "[verse]\nOh"


def test_an_unseeded_draw_does_not_claim_a_seed():
    """The CLI always picks one; a caller building the request may not.

    Writing a seed there would offer a reproduction that does not reproduce.
    """
    assert "AS15_SEED" not in _tags(MODELS["xl-sft"])


def test_conditioning_metas_are_recorded_only_when_they_were_given():
    """The model is told "N/A" for an unset one, which is not a value to record."""
    assert "AS15_BPM" not in _tags(MODELS["xl-sft"])

    tags = _tags(MODELS["xl-sft"], bpm=128, key_scale="C minor", time_signature=3)
    assert tags["AS15_BPM"] == "128"
    assert tags["AS15_KEY"] == "C minor"
    assert tags["AS15_TIME_SIGNATURE"] == "3"


def test_every_tag_name_is_one_the_container_can_carry():
    """describe() names the fields, and flac.py is the only thing checking them.

    A name with an ``=`` or a non-ASCII character in it would fail the write
    at the end of a generation rather than here.
    """
    from as15 import flac

    tags = _tags(MODELS["xl-sft"], bpm=128, key_scale="C minor", time_signature=3)
    for name in tags:
        flac.check_field_name(name)


def test_the_tags_do_not_move_between_two_runs_of_the_same_request():
    """No clock, no host, no run counter.

    Regenerating at a fixed seed and diffing the file against the previous
    build is how a change is shown not to have moved the audio; a timestamp in
    the metadata would make every such diff non-empty.
    """
    assert _tags(MODELS["xl-sft"], seed=7) == _tags(MODELS["xl-sft"], seed=7)


def test_the_written_file_carries_the_generation_it_came_from(tmp_path):
    """End to end, because describe() and the writer are wired in two places."""
    out = tmp_path / "song.flac"
    tags = _tags(
        MODELS["xl-sft"],
        style_prompt="dream pop, warm analog tape",
        lyrics="[verse]\nCity lights",
    )
    pipeline.write_audio(out, _tone(), models.SAMPLE_RATE, tags)

    written = flac_comments(out.read_bytes())
    assert written["DESCRIPTION"] == "dream pop, warm analog tape"
    assert written["LYRICS"] == "[verse]\nCity lights"
    assert written["AS15_MODEL"] == "xl-sft"


def test_a_write_without_tags_leaves_the_encoder_to_it(tmp_path):
    """The default path is unchanged: no tags means no rewrite of the stream."""
    out = tmp_path / "song.flac"
    pipeline.write_audio(out, _tone(), models.SAMPLE_RATE)

    assert flac_comments(out.read_bytes()) == {}


def test_the_conditioning_and_the_diffusion_are_told_the_same_song(monkeypatch):
    """Every stage is handed the resolved request, and there is only one.

    The conditioner used to be passed the request's own fields and size its
    window from them, so the length it conditioned on and the length the
    sampler generated were two separate readings of the same number -- a track
    a frame short of the context it was generated against, at whichever
    duration the two roundings disagreed on.
    """
    log: list[str] = []
    seen: dict = {}
    _run_with_stub_stages(monkeypatch, log, seen=seen, duration=12.9)

    resolved = seen["conditioned"]
    assert resolved.latent_frames == 323
    assert resolved.metas_duration == 13
    assert seen["sampled"]["src_latents_shape"] == (1, 323, models.LATENT_CHANNELS)
    for field in (
        "shift",
        "steps",
        "guidance",
        "seed",
        "sampler",
        "infer_method",
        "dcw",
    ):
        assert seen["sampled"][field] == getattr(resolved, field), field

    # And the file's own account of the run is that same resolved request, so
    # the sampler the loop was handed is the sampler the take is tagged with.
    tags = pipeline.describe(MODELS["xl-sft"], resolved)
    assert tags["AS15_SAMPLER"] == seen["sampled"]["sampler"]
    assert tags["AS15_INFER_METHOD"] == seen["sampled"]["infer_method"]
