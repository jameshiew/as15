"""Vorbis comments in the FLAC this package writes.

libsndfile -- which is what soundfile encodes through -- sets ten Vorbis
fields and no others: title, copyright, software, artist, comment, date,
album, license, tracknumber and genre. Lyrics are not among them, and neither
is anywhere to record the prompt, the checkpoint or the seed a file came from.
So the comment block is assembled here and installed into the encoded stream
afterwards.

A tagging library would have done this, and there is no non-GPL one. mutagen
is GPL-2.0-or-later and pytaglib GPL-3.0-or-later; mediafile and music-tag are
MIT but import mutagen at runtime, which is the same licence with a wrapper on
it. tinytag reads and does not write, audio-metadata has been unmaintained
since 2020 (it still pins ``attrs<19.4``), and pyFLAC binds libFLAC's encoder
and decoder but not its metadata API. What is left to do by hand is one block
holding a length-prefixed list of ``NAME=value`` strings, which is the forty
lines below.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

MAGIC = b"fLaC"

# The block types this cares about. STREAMINFO is mandatory and must come
# first; VORBIS_COMMENT is the one being replaced. The rest -- padding,
# seektable, pictures -- are copied through untouched.
STREAMINFO = 0
VORBIS_COMMENT = 4

# A block header carries its payload length in 24 bits.
MAX_BLOCK_BYTES = 2**24 - 1


def check_field_name(name: str) -> None:
    """Reject a comment field name the format cannot carry.

    A name is ASCII 0x20 to 0x7D with ``=`` excluded, because ``=`` is the
    separator: a field called ``A=B`` would be read back as a field ``A`` whose
    value starts ``B=``, and a non-ASCII one is simply not a field name. Both
    produce a file that parses, so nothing downstream would report either.

    Raises:
        ValueError: naming the offending field.
    """
    if not name:
        raise ValueError("a comment field name cannot be empty.")
    bad = [c for c in name if not (0x20 <= ord(c) <= 0x7D) or c == "="]
    if bad:
        raise ValueError(
            f"comment field {name!r} contains {''.join(bad)!r}; field names are "
            "ASCII 0x20-0x7D excluding '='."
        )


def _split(data: bytes) -> tuple[list[tuple[int, bytes]], bytes]:
    """*data*'s metadata blocks as ``(type, payload)``, and the frames after them.

    Raises:
        ValueError: if *data* is not a FLAC stream, or its metadata is cut short.
    """
    if not data.startswith(MAGIC):
        raise ValueError("not a FLAC stream: no fLaC marker.")

    blocks: list[tuple[int, bytes]] = []
    pos = len(MAGIC)
    while True:
        header = data[pos : pos + 4]
        if len(header) < 4:
            raise ValueError("FLAC metadata ends part way through a block header.")
        last = bool(header[0] & 0x80)
        kind = header[0] & 0x7F
        length = int.from_bytes(header[1:4], "big")
        pos += 4

        payload = data[pos : pos + length]
        if len(payload) < length:
            raise ValueError(
                f"FLAC metadata block of type {kind} claims {length} bytes and "
                f"the file holds {len(payload)}."
            )
        blocks.append((kind, payload))
        pos += length

        if last:
            break

    if blocks[0][0] != STREAMINFO:
        raise ValueError(
            f"FLAC stream begins with a type {blocks[0][0]} block, not STREAMINFO."
        )
    return blocks, data[pos:]


def _join(blocks: list[tuple[int, bytes]], frames: bytes) -> bytes:
    """The stream *blocks* and *frames* make, with the last-block flag set once.

    Raises:
        ValueError: if a block is longer than its header can describe.
    """
    out = bytearray(MAGIC)
    for index, (kind, payload) in enumerate(blocks):
        if len(payload) > MAX_BLOCK_BYTES:
            # In practice only the comment block can reach this; every other
            # block came off disk, where it had already been through the same
            # 24-bit field.
            raise ValueError(
                f"a metadata block of {len(payload)} bytes does not fit the "
                f"{MAX_BLOCK_BYTES} a FLAC block header can describe."
            )
        out.append((0x80 if index == len(blocks) - 1 else 0x00) | kind)
        out += len(payload).to_bytes(3, "big")
        out += payload
    return bytes(out) + frames


def _entry(value: bytes) -> bytes:
    return len(value).to_bytes(4, "little") + value


def _vendor(payload: bytes) -> bytes:
    """The vendor string an existing comment block opens with."""
    length = int.from_bytes(payload[:4], "little")
    return payload[4 : 4 + length]


def _comment_block(vendor: bytes, tags: Mapping[str, str]) -> bytes:
    """A VORBIS_COMMENT payload: little-endian, alone among FLAC's blocks.

    Values are UTF-8 and length-prefixed rather than terminated, so a newline
    in one is ordinary content -- which is what makes a whole lyric sheet a
    single field rather than something that needs escaping.
    """
    entries = [_entry(f"{name}={value}".encode()) for name, value in tags.items()]
    return _entry(vendor) + len(entries).to_bytes(4, "little") + b"".join(entries)


def set_comments(path: Path, tags: Mapping[str, str]) -> None:
    """Replace *path*'s Vorbis comments with *tags*, keeping its vendor string.

    The vendor string identifies the encoder that produced the frames, which is
    still libFLAC afterwards, so it is carried over rather than reasserted.

    The whole file is read and rewritten: a metadata block cannot grow in place
    without moving every audio frame behind it. That is safe here because the
    caller writes through :func:`as15.atomic.publish` and hands this the
    temporary -- a rewrite interrupted half way leaves a corrupt file that
    publish then deletes, never the destination.

    Raises:
        ValueError: if a field name is unusable, or *path* is not a FLAC file.
    """
    for name in tags:
        check_field_name(name)

    blocks, frames = _split(path.read_bytes())
    vendor = next(
        (_vendor(payload) for kind, payload in blocks if kind == VORBIS_COMMENT), b""
    )
    kept = [(kind, payload) for kind, payload in blocks if kind != VORBIS_COMMENT]
    # STREAMINFO has to stay first and the rest are unordered, so the comments
    # go straight after it -- where every other encoder puts them, and where a
    # reader that stops early still finds them.
    kept.insert(1, (VORBIS_COMMENT, _comment_block(vendor, tags)))

    path.write_bytes(_join(kept, frames))
