"""Audio codes: the 5 Hz plan a song can be conditioned on.

An audio code is one entry in the DiT checkpoint's residual-FSQ codebook,
standing for 200 ms of audio -- five 40 ms latent frames. A whole song's worth
of them is a *plan*: a coarse sketch of how the track moves, which the DiT then
renders. They come either from the 5 Hz planner LM (:mod:`as15.planner`) or
from a file someone else's run produced.

Nothing here imports torch or MLX. The CLI validates a plan before anything is
downloaded or loaded, and paying for torch to find out that a file is empty
would defeat the point.

The serialised form is upstream's, exactly: a run of ``<|audio_code_N|>``
tokens with no separators, no header and no version. That is not much of a
format, but it is the only one every ACE-Step surface speaks -- its CLI's TOML,
its web UI's JSON sidecar, its HTTP API and its clipboard all carry that same
string -- so a plan written here can be rendered there and back.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

# One code per 200 ms: the detokenizer expands each into ``pool_window_size``
# latent frames, and the latent grid is 25 Hz.
POOL_WINDOW = 5

# How many codes the codebook holds -- 8*8*8*5*5*5 from the checkpoint's
# ``fsq_input_levels``. This is the runtime's contract rather than a cached
# copy of the config, for the same reason the VAE geometry is: a plan is
# validated before any checkpoint is on disk. ``AudioCodeDecoder`` checks the
# checkpoint it loads against it, so a checkpoint that disagrees is an error
# rather than a wrong-codebook plan.
#
# The planner's vocabulary is wider than this: it carries 65535
# ``<|audio_code_N|>`` tokens, of which only the first 64000 index anything.
CODEBOOK_SIZE = 64_000

_CODE = re.compile(r"<\|audio_code_(\d+)\|>")


def codes_for_frames(frames: int, pool_window: int = POOL_WINDOW) -> int:
    """How many codes it takes to cover *frames* latent frames.

    Rounded up. A plan that stops mid-window leaves the tail of the song
    conditioned on nothing, and the DiT reads the whole context block.
    """
    return -(-frames // pool_window)


def frames_for_codes(count: int, pool_window: int = POOL_WINDOW) -> int:
    """How many latent frames *count* codes cover."""
    return count * pool_window


def parse_codes(text: str) -> tuple[int, ...]:
    """Every ``<|audio_code_N|>`` in *text*, in order.

    Scanned rather than parsed, which is what upstream does and what makes
    this accept a plan however it was carried: on its own, inside the
    ``audio_codes`` field of a JSON sidecar, in a TOML config, or pasted out of
    a web UI between two ``[audio_codes]`` markers. Anything that is not a code
    token is skipped.

    Raises:
        ValueError: if there is no code token in *text* at all, which is the
            one case where scanning silently returning nothing would look like
            an empty plan rather than the wrong file.
    """
    codes = tuple(int(match) for match in _CODE.findall(text))
    if not codes:
        raise ValueError(
            "no audio codes found; a plan is a run of <|audio_code_N|> tokens, "
            "as written by `as15 plan` or by ACE-Step's own tools."
        )
    return codes


def format_codes(codes: Sequence[int]) -> str:
    """*codes* in the serialised form, which is the one upstream reads."""
    return "".join(f"<|audio_code_{code}|>" for code in codes)


def read_codes(path: Path) -> tuple[int, ...]:
    """Parse the plan in the file at *path*.

    Decoded as UTF-8 explicitly rather than in the platform's encoding, so a
    plan is the same plan on every machine.

    Raises:
        ValueError: if the file cannot be read or holds no plan. Both are
            usage errors at the CLI, and neither deserves a traceback.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read the audio-code plan {path}: {exc}") from None
    try:
        return parse_codes(text)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def check_codes(
    codes: Sequence[int],
    frames: int,
    codebook_size: int = CODEBOOK_SIZE,
    pool_window: int = POOL_WINDOW,
) -> None:
    """Reject a plan the DiT would be misconditioned on rather than refuse.

    Both bounds exist because neither failure announces itself. A negative
    index wraps around the codebook and returns a perfectly finite hint from
    the wrong end of it -- verified against the installed ``ResidualFSQ``,
    which is why this does not simply let the lookup raise. A plan shorter than
    the song leaves its tail conditioned on nothing; upstream pads that with
    silence and says nothing, which returns a track whose last minute quietly
    stops following the plan.

    A plan *longer* than the song is fine and is cropped: the codes cover whole
    200 ms windows, so any song whose length is not a multiple of that has to
    overshoot by design.

    Raises:
        ValueError: naming the code and what is wrong with it.
    """
    if not codes:
        raise ValueError("the audio-code plan is empty.")

    for position, code in enumerate(codes):
        if not 0 <= code < codebook_size:
            raise ValueError(
                f"audio code {code} at position {position} is outside the "
                f"codebook, which holds {codebook_size} codes (0 to "
                f"{codebook_size - 1})."
            )

    needed = codes_for_frames(frames, pool_window)
    if len(codes) < needed:
        # Reported in seconds as well as codes, because that is the unit the
        # mismatch gets fixed in -- but without naming a duration to pass,
        # which for a very short plan would be one the CLI rejects in turn.
        have = frames_for_codes(len(codes), pool_window) / 25
        want = frames / 25
        raise ValueError(
            f"the audio-code plan is {len(codes)} codes, covering {have:g}s; a "
            f"{want:g}s song needs {needed}. Generate a longer plan, or ask "
            f"for a shorter song."
        )
