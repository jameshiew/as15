"""Audio-code plans: parsing, the frame arithmetic, and what is rejected.

Nothing here loads anything. A plan is validated by the CLI before a checkpoint
is on disk, which is the whole reason :mod:`as15.codes` imports neither torch
nor MLX -- so these tests are the ones that have to stay fast and total.
"""

from __future__ import annotations

import pytest

from as15.codes import (
    CODEBOOK_SIZE,
    POOL_WINDOW,
    check_codes,
    codes_for_frames,
    format_codes,
    frames_for_codes,
    parse_codes,
    read_codes,
)

# --- the serialised form --------------------------------------------------


def test_a_plan_round_trips_through_its_serialised_form():
    codes = (0, 1, 63999, 12345)
    assert parse_codes(format_codes(codes)) == codes


def test_codes_are_scanned_out_of_whatever_carried_them():
    """The form is upstream's, and upstream's parser is a scan.

    That is what lets a plan arrive as the ``audio_codes`` field of a web UI's
    JSON sidecar, inside a TOML config, or pasted between the ``[audio_codes]``
    markers a streaming API wraps it in, without a format flag for each.
    """
    carriers = [
        '{"audio_codes": "<|audio_code_7|><|audio_code_8|>", "seed": 1}',
        "[audio_codes]<|audio_code_7|><|audio_code_8|>[/audio_codes]",
        "\n  <|audio_code_7|>\n  <|audio_code_8|>\n",
        'audio_codes = "<|audio_code_7|><|audio_code_8|>"',
    ]
    for carrier in carriers:
        assert parse_codes(carrier) == (7, 8), carrier


def test_a_file_that_carries_no_plan_is_an_error_not_an_empty_plan():
    """Scanning something that is not a plan otherwise looks like a plan of nothing."""
    with pytest.raises(ValueError, match="no audio codes found"):
        parse_codes("bpm: 120\nkeyscale: C major\n")


def test_reading_names_the_file_that_could_not_be_read(tmp_path):
    missing = tmp_path / "nope.codes"
    with pytest.raises(ValueError, match=r"nope\.codes"):
        read_codes(missing)


def test_reading_a_plan_that_is_not_utf8_is_a_plain_error(tmp_path):
    """A traceback out of the decoder is not what a mistyped path deserves."""
    path = tmp_path / "plan.codes"
    path.write_bytes(b"\xff\xfe<|audio_code_1|>")
    with pytest.raises(ValueError, match="could not read"):
        read_codes(path)


def test_a_plan_is_read_as_utf8_whatever_the_platform_prefers(tmp_path):
    path = tmp_path / "plan.codes"
    path.write_text("# plan for 'café'\n<|audio_code_5|>", encoding="utf-8")
    assert read_codes(path) == (5,)


# --- frame arithmetic -----------------------------------------------------


def test_a_plan_covers_whole_windows_and_rounds_up():
    """500 frames is exactly 100 codes; 501 needs a 101st.

    Rounding down instead would leave the last frames of the song conditioned
    on nothing, which is inaudible as an error and audible as a worse ending.
    """
    assert codes_for_frames(500) == 100
    assert codes_for_frames(501) == 101
    assert codes_for_frames(1) == 1
    assert frames_for_codes(100) == 500
    assert POOL_WINDOW == 5


# --- validation -----------------------------------------------------------


def test_a_negative_code_is_rejected_rather_than_wrapped():
    """The lookup would not raise: it wraps and returns a finite hint.

    Verified against the installed ResidualFSQ -- index -1 comes back as the
    last codebook entry, so a plan with a sign error conditions the song on
    plausible nonsense. This check is the only thing between the two.
    """
    with pytest.raises(ValueError, match="outside the codebook"):
        check_codes([1, -1, 2], frames=5)


def test_a_code_above_the_codebook_is_rejected():
    """The planner's vocabulary is wider than the codebook it indexes.

    65535 ``<|audio_code_N|>`` tokens over 64000 codes, so the top 1535 name
    nothing. Upstream clamps them into range; here a plan carrying one is
    wrong rather than quietly moved.
    """
    with pytest.raises(ValueError, match="outside the codebook"):
        check_codes([CODEBOOK_SIZE], frames=5)
    check_codes([CODEBOOK_SIZE - 1], frames=5)


def test_the_rejection_names_the_offending_code_and_where_it_is():
    with pytest.raises(ValueError, match=r"audio code 70000 at position 2"):
        check_codes([1, 2, 70000], frames=5)


def test_a_plan_too_short_for_the_song_is_rejected_in_seconds():
    """Upstream pads the shortfall with silence and says nothing.

    That returns a track whose last stretch quietly stops following the plan,
    so the message is in the unit the mismatch gets fixed in.
    """
    with pytest.raises(
        ValueError, match=r"20 codes, covering 4s; a 20s song needs 100"
    ):
        check_codes([0] * 20, frames=500)


def test_a_plan_longer_than_the_song_is_fine():
    """Codes cover whole 200 ms windows, so any song not a multiple overshoots."""
    check_codes([0] * 101, frames=501)
    check_codes([0] * 500, frames=5)


def test_an_empty_plan_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        check_codes([], frames=5)
