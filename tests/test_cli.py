"""The command line, which is the only interface this package has.

Everything here is about failing in the first second rather than the fourth
minute, and about the banner describing the run that is actually about to
happen. The generation session is stubbed throughout: a regression that let a
bad option through should fail the test, not spend a CI run downloading 19 GB.
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
    ["--bpm", "29"],
    ["--bpm", "301"],
    ["--time-signature", "5"],
    ["--language", " "],
    ["--language", "en\nfr"],
    ["--key", "C major\n- bpm: 200"],
    # Last one wins, so this is `sing` with a blank prompt.
    ["--prompt", "   "],
    ["--out", "song.mp3"],
    ["--out", "song"],
]


def _unreachable_session(monkeypatch, why: str = "a session was opened"):
    """Fail loudly if ``sing`` gets as far as generating anything."""

    def unreachable(*args, **kwargs):
        raise AssertionError(why)

    monkeypatch.setattr(cli, "GenerationSession", unreachable)


@pytest.mark.parametrize("argv", CLI_REJECTS, ids=lambda a: " ".join(a))
def test_the_cli_rejects_unusable_options(argv, monkeypatch):
    """Every one of these has to fail before a session fetches ~10 GB.

    The session is stubbed to say so rather than to let a regression here spend
    a CI run downloading a checkpoint.
    """
    _unreachable_session(monkeypatch, f"a session was opened for {argv}")

    result = CliRunner().invoke(cli.app, ["sing", "--prompt", "test", *argv])
    assert result.exit_code != 0


def _stub_session(monkeypatch, seen: dict | None = None, write=None):
    """Let ``sing`` run to completion without a checkpoint.

    The takes it hands back are silent and instant, but they arrive the way
    real ones do -- one per seed, yielded in turn -- so the CLI's own loop is
    the thing under test.
    """

    class FakeSession:
        def __init__(self, spec, request, device="auto", progress=True):
            if seen is not None:
                seen["request"] = request
                seen["progress"] = progress

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return None

        def takes(self, seeds):
            if seen is not None:
                seen["seeds"] = list(seeds)
            for seed in seeds:
                yield pipeline.GenerationResult(
                    audio=np.zeros((4, 2), dtype=np.float32),
                    sample_rate=48_000,
                    seed=seed,
                )

    monkeypatch.setattr(cli, "GenerationSession", FakeSession)
    monkeypatch.setattr(cli, "write_audio", write or (lambda *args: None))


def test_the_cli_accepts_the_smallest_valid_step_count(monkeypatch, tmp_path):
    """The bound on --steps must reject 0 without rejecting 1.

    Also a positive control for the cases above: a --steps that no longer
    parses at all would fail those for the wrong reason.
    """
    seen: dict = {}
    _stub_session(monkeypatch, seen)

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
    _stub_session(monkeypatch, seen)

    result = CliRunner().invoke(
        cli.app, ["sing", "-p", "x", "-m", "xl-turbo", "-o", str(tmp_path / "a.flac")]
    )

    assert result.exit_code == 0, result.output
    assert "steps 8" in result.stderr  # xl-turbo's default, not the CLI's
    assert "shift 3" in result.stderr
    assert "dcw on" in result.stderr
    assert "guidance 1" in result.stderr  # distilled: CFG dropped, and said so
    assert seen["request"].steps is None, "the request still says 'the model's'"


def test_the_banner_reports_the_duration_the_take_will_have(monkeypatch, tmp_path):
    """The latent grid is 40 ms, so most durations are not generatable as typed.

    The banner used to echo the flag, which is the one number in the run that
    nothing downstream used: the model was told 12 seconds, 12.92 was generated
    and 12.9 was printed and written into the file.
    """
    _stub_session(monkeypatch)

    result = CliRunner().invoke(
        cli.app, ["sing", "-p", "x", "-d", "12.9", "-o", str(tmp_path / "a.flac")]
    )

    assert result.exit_code == 0, result.output
    assert "duration  12.92s" in result.stderr


def test_an_omitted_seed_is_chosen_and_reported(monkeypatch, tmp_path):
    """ "Reuse the seed to reproduce a take" needs the seed to have been shown.

    Left as None it would reach the sampler as an unseeded draw, and the run
    would be unreproducible with nothing saying so.
    """
    seen: dict = {}
    _stub_session(monkeypatch, seen)

    result = CliRunner().invoke(
        cli.app, ["sing", "-p", "x", "-o", str(tmp_path / "a.flac")]
    )

    assert result.exit_code == 0, result.output
    seed = seen["request"].seed
    assert seed is not None
    assert f"seed {seed}" in result.stderr


def test_quiet_turns_off_the_progress_bar(monkeypatch, tmp_path):
    seen: dict = {}
    _stub_session(monkeypatch, seen)

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
    _stub_session(monkeypatch, seen)
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

    class TooLong:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return None

        # A generator, like the real one, so this surfaces where it really does
        # -- on the first take being asked for rather than on the call.
        def takes(self, seeds):
            raise conditioning.InputTooLong("the lyrics ... is 2200 tokens")
            yield

    monkeypatch.setattr(cli, "GenerationSession", TooLong)
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
    _stub_session(monkeypatch)
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
    _stub_session(monkeypatch)

    result = CliRunner().invoke(cli.app, ["sing", "-p", "x", "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert "will be overwritten" in result.stderr


# --- more than one take ---------------------------------------------------
#
# Which take of a song is the good one is a listening decision, so the useful
# unit of work is several of them. What the CLI owes that is a name per take, a
# seed per take, and no run so brittle that one bad write loses the rest.


def test_a_batch_counts_its_seeds_up_from_the_one_given(monkeypatch, tmp_path):
    """Named by its first seed, so the whole batch is reproducible from it."""
    seen: dict = {}
    _stub_session(monkeypatch, seen)

    result = CliRunner().invoke(
        cli.app,
        [
            "sing",
            "-p",
            "x",
            "--seed",
            "100",
            "--takes",
            "3",
            "-o",
            str(tmp_path / "a.flac"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["seeds"] == [100, 101, 102]
    assert "seeds 100-102" in result.stderr


def test_each_take_of_a_batch_gets_its_own_file(monkeypatch, tmp_path):
    """Four takes into one path would leave the fourth and lose three."""
    written: list[str] = []
    _stub_session(monkeypatch, write=lambda path, *args: written.append(path.name))

    result = CliRunner().invoke(
        cli.app,
        [
            "sing",
            "-p",
            "x",
            "--seed",
            "7",
            "--takes",
            "3",
            "-o",
            str(tmp_path / "a.flac"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert written == ["a-01-seed-7.flac", "a-02-seed-8.flac", "a-03-seed-9.flac"]
    for name in written:
        assert f"wrote {tmp_path / name}" in result.stderr


def test_a_single_take_is_still_written_where_it_was_asked_for(monkeypatch, tmp_path):
    """The default run is unchanged: no index, no seed, no invented name."""
    written: list[str] = []
    _stub_session(monkeypatch, write=lambda path, *args: written.append(path.name))

    result = CliRunner().invoke(
        cli.app, ["sing", "-p", "x", "--seed", "7", "-o", str(tmp_path / "a.flac")]
    )

    assert result.exit_code == 0, result.output
    assert written == ["a.flac"]
    assert "takes" not in result.stderr


def test_every_take_of_a_batch_is_preflighted_before_the_first_one_runs(monkeypatch):
    """A batch that cannot be written is a batch that should not be generated.

    The check is on the derived names rather than on ``--out``, which is not
    one of them.
    """
    _unreachable_session(monkeypatch)

    result = CliRunner().invoke(
        cli.app, ["sing", "-p", "x", "--takes", "2", "-o", "song.mp3"]
    )
    assert result.exit_code != 0


def test_a_batch_that_would_run_past_the_last_seed_is_rejected(monkeypatch):
    """Counting up has an end, and reaching it minutes in would be a poor time."""
    _unreachable_session(monkeypatch)

    result = CliRunner().invoke(
        cli.app,
        ["sing", "-p", "x", "--seed", str(pipeline.MAX_SEED), "--takes", "2"],
    )
    assert result.exit_code != 0
    assert "largest seed" in result.output


def test_one_take_that_cannot_be_written_does_not_take_the_batch_with_it(
    monkeypatch, tmp_path
):
    """The takes behind a bad write are worth no less than the ones in front.

    A batch is minutes of diffusion each; losing the lot to one full disk --
    or to one decode that came back non-finite -- would be the expensive
    failure mode.
    """
    written: list[str] = []

    def write(path, *args):
        if "02" in path.name:
            raise OSError("no space left on device")
        written.append(path.name)

    _stub_session(monkeypatch, write=write)

    result = CliRunner().invoke(
        cli.app,
        [
            "sing",
            "-p",
            "x",
            "--seed",
            "7",
            "--takes",
            "3",
            "-o",
            str(tmp_path / "a.flac"),
        ],
    )

    assert written == ["a-01-seed-7.flac", "a-03-seed-9.flac"]
    assert "Could not write" in result.stderr
    # Reported as a failure even though two takes landed: a script that asked
    # for three and got two should not read the run as having worked.
    assert result.exit_code == 1


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


# --- planning -------------------------------------------------------------


def _plan_file(tmp_path, count: int):
    from as15.codes import format_codes

    path = tmp_path / "plan.codes"
    path.write_text(format_codes(range(count)), encoding="utf-8")
    return path


def test_a_supplied_plan_reaches_the_request(monkeypatch, tmp_path):
    seen: dict = {}
    _stub_session(monkeypatch, seen)

    result = CliRunner().invoke(
        cli.app,
        [
            "sing",
            "-p",
            "x",
            "-d",
            "10",
            "--audio-codes",
            str(_plan_file(tmp_path, 50)),
            "-o",
            str(tmp_path / "a.flac"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["request"].audio_codes == tuple(range(50))
    assert seen["request"].planner is None
    assert "plan      50 codes given" in result.output


def test_a_plan_file_that_holds_no_plan_is_a_usage_error(monkeypatch, tmp_path):
    """Before the checkpoints download, not after."""

    _unreachable_session(monkeypatch)
    path = tmp_path / "notes.txt"
    path.write_text("some notes about the song", encoding="utf-8")

    result = CliRunner().invoke(
        cli.app, ["sing", "-p", "x", "--audio-codes", str(path)]
    )
    assert result.exit_code != 0
    assert "no audio codes found" in result.output


def test_a_plan_too_short_for_the_song_is_a_usage_error(monkeypatch, tmp_path):
    _unreachable_session(monkeypatch)

    result = CliRunner().invoke(
        cli.app,
        [
            "sing",
            "-p",
            "x",
            "-d",
            "120",
            "--audio-codes",
            str(_plan_file(tmp_path, 50)),
        ],
    )
    assert result.exit_code != 0
    assert "needs 600" in result.output


def test_writing_a_plan_and_supplying_one_together_is_rejected(monkeypatch, tmp_path):
    _unreachable_session(monkeypatch)

    result = CliRunner().invoke(
        cli.app,
        [
            "sing",
            "-p",
            "x",
            "-d",
            "10",
            "--plan",
            "--audio-codes",
            str(_plan_file(tmp_path, 50)),
        ],
    )
    assert result.exit_code != 0


def test_an_unknown_planner_is_rejected_before_anything_downloads(monkeypatch):
    _unreachable_session(monkeypatch)

    result = CliRunner().invoke(
        cli.app, ["sing", "-p", "x", "--plan", "--planner", "7b"]
    )
    assert result.exit_code != 0
    assert "Unknown planner" in result.output


def test_the_planner_seed_follows_the_run_seed_unless_pinned(monkeypatch, tmp_path):
    """One --seed reproduces the whole run, plan included.

    Pinning --planner-seed on its own is what keeps a plan fixed while --seed
    moves the render, which is how you hear what the diffusion contributes.
    """
    seen: dict = {}
    _stub_session(monkeypatch, seen)
    args = ["sing", "-p", "x", "-d", "10", "--plan", "-o", str(tmp_path / "a.flac")]

    assert CliRunner().invoke(cli.app, [*args, "--seed", "5"]).exit_code == 0
    assert seen["request"].planner_seed == 5

    assert (
        CliRunner()
        .invoke(cli.app, [*args, "--seed", "5", "--planner-seed", "9"])
        .exit_code
        == 0
    )
    assert seen["request"].planner_seed == 9
    assert seen["request"].seed == 5


def test_the_banner_names_the_planner_and_what_it_costs(monkeypatch, tmp_path):
    """The 4B is an 8.4 GB download, which is worth saying before it starts."""
    _stub_session(monkeypatch)

    result = CliRunner().invoke(
        cli.app,
        ["sing", "-p", "x", "-d", "10", "--plan", "-o", str(tmp_path / "a.flac")],
    )
    assert result.exit_code == 0, result.output
    assert "planning  4b (8.4 GB)" in result.output
