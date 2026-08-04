"""The comment block written into the encoded stream.

This is byte surgery on a file libsndfile has just finished writing, so the way
it goes wrong is a file that still plays: players skip metadata they cannot
parse, and soundfile decodes the frames without looking at the blocks in front
of them. Everything below therefore checks the bytes, with the readers in
``helpers`` -- written separately from the code under test, because a round
trip through a writer and its own inverse agrees with itself about any
consistent misreading of the format.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from as15 import flac
from as15.flac import MAGIC, STREAMINFO, VORBIS_COMMENT
from helpers import flac_blocks, flac_comments, flac_frames, flac_stream, flac_vendor


def _encode(path, seconds: float = 0.05, channels: int = 2) -> np.ndarray:
    """A small FLAC, written the way the pipeline writes one."""
    t = np.arange(int(seconds * 48_000)) / 48_000
    wave = 0.5 * np.sin(2 * np.pi * 440.0 * t, dtype=np.float32)
    audio = np.repeat(wave[:, None], channels, axis=1)
    sf.write(str(path), audio, 48_000, subtype="PCM_16")
    return audio


# --- writing tags ---------------------------------------------------------


def test_the_tags_written_are_the_tags_read_back(tmp_path):
    out = tmp_path / "song.flac"
    _encode(out)
    tags = {"DESCRIPTION": "dream pop", "AS15_SEED": "42"}

    flac.set_comments(out, tags)

    assert flac_comments(out.read_bytes()) == tags


def test_a_whole_lyric_sheet_survives_as_one_field(tmp_path):
    """Entries are length-prefixed, so a newline in a value is just content.

    This is the field libsndfile has no way to set, and the reason its ten are
    not enough: it is multi-line, it is not ASCII, and it is the only record of
    the words a take sings.
    """
    out = tmp_path / "song.flac"
    _encode(out)
    sheet = "[verse]\nCity lights are fading slow\n\n[chorus]\nHold me — été"

    flac.set_comments(out, {"LYRICS": sheet})

    assert flac_comments(out.read_bytes())["LYRICS"] == sheet


def test_the_audio_is_untouched_by_the_rewrite(tmp_path):
    """The frames move, and nothing else about them may.

    Inserting a block shifts every frame behind it, and a rewrite that lost or
    duplicated a byte there would still decode -- FLAC resynchronises on the
    next frame header -- just to different audio.
    """
    out = tmp_path / "song.flac"
    audio = _encode(out)
    before = flac_blocks(out.read_bytes())[0][2]

    flac.set_comments(out, {"AS15_SEED": "42"})

    read, rate = sf.read(str(out), dtype="float32", always_2d=True)
    assert rate == 48_000
    assert np.abs(read - audio).max() <= 2**-15
    # STREAMINFO carries the MD5 of the unencoded audio, which is a check on
    # the frames that does not go back through the decoder this test just used.
    assert flac_blocks(out.read_bytes())[0][2] == before


def test_the_encoders_vendor_string_is_kept(tmp_path):
    """It names what encoded the frames, and that is still libFLAC afterwards."""
    out = tmp_path / "song.flac"
    _encode(out)
    before = flac_vendor(out.read_bytes())
    assert before is not None and b"libFLAC" in before

    flac.set_comments(out, {"AS15_SEED": "42"})

    assert flac_vendor(out.read_bytes()) == before


def test_retagging_replaces_the_block_rather_than_adding_one(tmp_path):
    """A second block is legal to write and ambiguous to read.

    Readers take the first, or the last, or merge them. libsndfile has already
    written one before this ever runs, so appending would have made every file
    the ambiguous case.
    """
    out = tmp_path / "song.flac"
    _encode(out)

    flac.set_comments(out, {"AS15_SEED": "1", "AS15_MODEL": "xl-sft"})
    flac.set_comments(out, {"AS15_SEED": "2"})

    kinds = [kind for kind, _, _ in flac_blocks(out.read_bytes())]
    assert kinds.count(VORBIS_COMMENT) == 1
    assert flac_comments(out.read_bytes()) == {"AS15_SEED": "2"}


def test_the_block_layout_stays_legal(tmp_path):
    """STREAMINFO first, and the last-block flag on the last block only.

    Both are how a reader finds the frames: a flag left set on an interior
    block makes it read the next block header as audio.
    """
    out = tmp_path / "song.flac"
    _encode(out)

    flac.set_comments(out, {"AS15_SEED": "42"})

    blocks = flac_blocks(out.read_bytes())
    assert blocks[0][0] == STREAMINFO
    assert blocks[1][0] == VORBIS_COMMENT
    assert [last for _, last, _ in blocks] == [False] * (len(blocks) - 1) + [True]


def test_an_empty_tag_set_is_a_block_with_no_entries(tmp_path):
    """Rather than no block, which is a different file to a reader."""
    out = tmp_path / "song.flac"
    _encode(out)

    flac.set_comments(out, {})

    assert flac_comments(out.read_bytes()) == {}
    assert flac_vendor(out.read_bytes()) is not None


def test_a_file_with_no_comment_block_of_its_own_gains_one(tmp_path):
    """Nothing guarantees the encoder wrote one; the surgery must not assume it.

    libsndfile does today. A file from anywhere else may not, and the
    difference is between inserting a block and replacing one.
    """
    out = tmp_path / "song.flac"
    _encode(out)
    data = out.read_bytes()
    stripped = [(k, b) for k, _, b in flac_blocks(data) if k != VORBIS_COMMENT]
    out.write_bytes(flac_stream(stripped, flac_frames(data)))

    flac.set_comments(out, {"AS15_SEED": "42"})

    assert flac_comments(out.read_bytes()) == {"AS15_SEED": "42"}
    # Nothing to carry a vendor over from, and inventing one would name an
    # encoder that did not encode this.
    assert flac_vendor(out.read_bytes()) == b""


# --- input the format cannot carry ----------------------------------------


@pytest.mark.parametrize("name", ["", "A=B", "café", "tab\there", "new\nline"])
def test_a_field_name_the_format_cannot_carry_is_rejected(name, tmp_path):
    """Each of these writes a file that parses and reads back as something else.

    ``A=B`` splits at the first ``=`` into a field ``A`` whose value starts
    ``B=``; the rest encode to bytes outside the permitted range, which a
    strict reader drops and a loose one keeps under a name nobody asked for.
    """
    out = tmp_path / "song.flac"
    _encode(out)
    before = out.read_bytes()

    with pytest.raises(ValueError, match="field"):
        flac.set_comments(out, {name: "x"})

    # Names are checked before the file is opened, so a bad one cannot leave a
    # half-rewritten stream behind.
    assert out.read_bytes() == before


def test_both_ends_of_the_permitted_range_are_field_names():
    """0x20 to 0x7D, so a space and a closing brace have to pass.

    A tighter check written from the tags this package happens to use would
    reject names the format allows, which is a different rule than the one the
    docstring claims.
    """
    flac.check_field_name(" ")
    flac.check_field_name("}")
    flac.check_field_name("AS15_TIME_SIGNATURE")


@pytest.mark.parametrize(
    "data,match",
    [
        (b"RIFFsomething", "no fLaC marker"),
        (MAGIC + b"\x00\x00", "part way through"),
        (MAGIC + b"\x80\x00\x10\x00" + b"short", "claims 4096 bytes"),
        (MAGIC + b"\x84\x00\x00\x00", "not STREAMINFO"),
    ],
)
def test_a_stream_that_is_not_one_is_named_rather_than_rewritten(data, match, tmp_path):
    """The input is a file soundfile has just written, so reaching any of these
    means something replaced it mid-write -- worth an error that names the
    format rather than an IndexError from the middle of a parser."""
    out = tmp_path / "song.flac"
    out.write_bytes(data)

    with pytest.raises(ValueError, match=match):
        flac.set_comments(out, {"AS15_SEED": "42"})


def test_a_comment_block_too_large_to_describe_is_refused(tmp_path):
    """The header holds the length in 24 bits, so ~16 MB of lyrics wraps.

    Conditioning caps the sheet at 2048 tokens long before this, but the wrap
    would be silent: the length truncates and the frames are then read from
    the middle of the block.
    """
    out = tmp_path / "song.flac"
    _encode(out)

    with pytest.raises(ValueError, match="does not fit"):
        flac.set_comments(out, {"LYRICS": "la " * (flac.MAX_BLOCK_BYTES // 2)})
