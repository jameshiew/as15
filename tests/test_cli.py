"""The command line, which is the only interface this package has.

Everything here is about failing in the first second rather than the fourth
minute, and about the banner describing the run that is actually about to
happen. ``generate`` is stubbed throughout: a regression that let a bad option
through should fail the test, not spend a CI run downloading 19 GB.
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from as15 import cli, conditioning, pipeline

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
    ["--out", "song.mp3"],
    ["--out", "song"],
]


@pytest.mark.parametrize("argv", CLI_REJECTS, ids=lambda a: " ".join(a))
def test_the_cli_rejects_unusable_options(argv, monkeypatch):
    """Every one of these has to fail before generate() fetches ~10 GB.

    generate() is stubbed to say so rather than to let a regression here spend
    a CI run downloading a checkpoint.
    """

    def unreachable(*args, **kwargs):
        raise AssertionError(f"generate() ran for {argv}")

    monkeypatch.setattr(cli, "generate", unreachable)

    result = CliRunner().invoke(cli.app, ["sing", "--prompt", "test", *argv])
    assert result.exit_code != 0


def _stub_generate(monkeypatch, seen: dict | None = None):
    """Let ``sing`` run to completion without a checkpoint."""

    def fake_generate(spec, request, device="auto", progress=True):
        if seen is not None:
            seen["request"] = request
            seen["progress"] = progress
        return pipeline.GenerationResult(
            audio=np.zeros((4, 2), dtype=np.float32), sample_rate=48_000, seed=0
        )

    monkeypatch.setattr(cli, "generate", fake_generate)
    monkeypatch.setattr(cli, "write_audio", lambda *args: None)


def test_the_cli_accepts_the_smallest_valid_step_count(monkeypatch, tmp_path):
    """The bound on --steps must reject 0 without rejecting 1.

    Also a positive control for the cases above: a --steps that no longer
    parses at all would fail those for the wrong reason.
    """
    seen: dict = {}
    _stub_generate(monkeypatch, seen)

    result = CliRunner().invoke(
        cli.app,
        [
            "sing",
            "-p",
            "x",
            "--steps",
            "1",
            "--shift",
            "2",
            "-o",
            str(tmp_path / "a.flac"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["request"].steps == 1
    assert seen["request"].shift == 2.0


def test_the_banner_reports_the_settings_the_run_will_use(monkeypatch, tmp_path):
    """It prints resolved values, and the checkpoint supplies most of them.

    Printing the request's own ``None``s -- or the CLI's own defaults, which
    are not the model's -- would describe a run nobody is about to make.
    """
    seen: dict = {}
    _stub_generate(monkeypatch, seen)

    result = CliRunner().invoke(
        cli.app, ["sing", "-p", "x", "-m", "xl-turbo", "-o", str(tmp_path / "a.flac")]
    )

    assert result.exit_code == 0, result.output
    assert "steps 8" in result.stderr  # xl-turbo's default, not the CLI's
    assert "shift 3" in result.stderr
    assert "dcw on" in result.stderr
    assert "guidance 1" in result.stderr  # distilled: CFG dropped, and said so
    assert seen["request"].steps is None, "the request still says 'the model's'"


def test_an_omitted_seed_is_chosen_and_reported(monkeypatch, tmp_path):
    """ "Reuse the seed to reproduce a take" needs the seed to have been shown.

    Left as None it would reach the sampler as an unseeded draw, and the run
    would be unreproducible with nothing saying so.
    """
    seen: dict = {}
    _stub_generate(monkeypatch, seen)

    result = CliRunner().invoke(
        cli.app, ["sing", "-p", "x", "-o", str(tmp_path / "a.flac")]
    )

    assert result.exit_code == 0, result.output
    seed = seen["request"].seed
    assert seed is not None
    assert f"seed {seed}" in result.stderr


def test_quiet_turns_off_the_progress_bar(monkeypatch, tmp_path):
    seen: dict = {}
    _stub_generate(monkeypatch, seen)

    CliRunner().invoke(cli.app, ["sing", "-p", "x", "-o", str(tmp_path / "a.flac")])
    assert seen["progress"] is True

    CliRunner().invoke(
        cli.app, ["sing", "-p", "x", "-q", "-o", str(tmp_path / "a.flac")]
    )
    assert seen["progress"] is False


def test_lyrics_come_from_a_file_or_stdin_or_neither(monkeypatch, tmp_path):
    """Three ways in, and the missing-file case has to be a usage error.

    An unreadable ``-L`` used to be whatever ``read_text`` raised, printed as
    a traceback under the banner.
    """
    seen: dict = {}
    _stub_generate(monkeypatch, seen)
    out = str(tmp_path / "a.flac")

    sheet = tmp_path / "lyrics.txt"
    sheet.write_text("[verse]\nCity lights")
    CliRunner().invoke(cli.app, ["sing", "-p", "x", "-L", str(sheet), "-o", out])
    assert seen["request"].lyrics == "[verse]\nCity lights"

    CliRunner().invoke(
        cli.app, ["sing", "-p", "x", "-L", "-", "-o", out], input="from stdin"
    )
    assert seen["request"].lyrics == "from stdin"

    result = CliRunner().invoke(cli.app, ["sing", "-p", "x", "-o", out])
    assert seen["request"].lyrics == ""
    assert "instrumental" in result.stderr

    result = CliRunner().invoke(
        cli.app, ["sing", "-p", "x", "-L", str(tmp_path / "nope.txt"), "-o", out]
    )
    assert result.exit_code != 0
    assert "No such file" in result.output + result.stderr


def test_the_cli_reports_input_it_cannot_condition_on(monkeypatch):
    """Only the tokenizer can catch this, and it loads with the conditioner.

    So it is the one bad-input case that gets past the option checks, and it
    has to land as an error rather than as a traceback.
    """

    def too_long(*args, **kwargs):
        raise conditioning.InputTooLong("the lyrics ... is 2200 tokens")

    monkeypatch.setattr(cli, "generate", too_long)
    monkeypatch.setattr(cli, "write_audio", lambda *args: None)

    result = CliRunner().invoke(cli.app, ["sing", "-p", "x", "-l", "y"])

    assert result.exit_code == 2
    assert "2200 tokens" in result.stderr
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_a_write_that_fails_after_the_run_is_an_error_not_a_traceback(
    monkeypatch, tmp_path
):
    """Four minutes of generation is worth more than a stack trace.

    The path was preflighted, so reaching here means something changed under
    the run -- the disk filled, the directory went away -- and the message has
    to name the file rather than the internals.
    """
    _stub_generate(monkeypatch)
    monkeypatch.setattr(
        cli,
        "write_audio",
        lambda *args: (_ for _ in ()).throw(OSError("no space left on device")),
    )

    out = tmp_path / "song.flac"
    result = CliRunner().invoke(cli.app, ["sing", "-p", "x", "-o", str(out)])

    assert result.exit_code == 1
    assert f"Could not write {out}" in result.stderr
    assert "no space left on device" in result.stderr


def test_the_cli_says_it_is_about_to_overwrite_a_take(monkeypatch, tmp_path):
    """Overwriting is the policy; being told after the fact is not.

    Said before the run rather than after, because after is four minutes too
    late to move the file that was there.
    """
    out = tmp_path / "song.flac"
    out.write_bytes(b"an earlier take")
    _stub_generate(monkeypatch)

    result = CliRunner().invoke(cli.app, ["sing", "-p", "x", "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert "will be overwritten" in result.stderr


@pytest.mark.parametrize(
    "shape,label",
    [((4,), "mono"), ((4, 1), "mono"), ((4, 2), "stereo"), ((4, 6), "6 channels")],
)
def test_the_cli_reports_the_channels_it_actually_wrote(shape, label):
    """This line said "stereo" whatever came back from the VAE."""
    assert cli._channels(np.zeros(shape, dtype=np.float32)) == label


def test_listing_the_models_names_the_commit_each_is_pinned_to():
    """The registry is the only place the pins are visible to a user."""
    from as15.models import MODELS

    result = CliRunner().invoke(cli.app, ["models"])

    assert result.exit_code == 0, result.output
    for key, spec in MODELS.items():
        assert key in result.output
        assert spec.revision[:8] in result.output
