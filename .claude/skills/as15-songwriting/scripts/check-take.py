#!/usr/bin/env -S uv run --quiet --with numpy --with soundfile python
"""What a take can be checked for without listening to it.

Listening is still the decision. This catches the takes not worth auditioning:
one that came back the wrong length, one pinned to the ceiling, one whose level
never moves -- the flat-render failure the model has most often -- and a
``[fade out]`` that never faded.

    scripts/check-take.py out/take-*.flac

Reads each file and prints a row per take plus an RMS envelope drawn in ASCII,
so a batch can be compared at a glance before any of it is played. Every number
is measured off the decoded audio; none of it says whether the song is any good.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

WINDOW_SECONDS = 4.0
RAMP = " .:-=+*#%@"


@dataclass(frozen=True)
class Take:
    """One take's measurements. Named fields rather than a dict so the
    comparisons below are checked rather than taken on trust."""

    name: str
    seconds: float
    rate: int
    channels: int
    peak: float
    dynamic_range_db: float
    width: float
    tail_db: float
    shape: str


def measure(path: Path) -> Take:
    audio, rate = sf.read(path, always_2d=True)
    mono = audio.mean(axis=1)
    window = int(WINDOW_SECONDS * rate)

    # Whole windows only: a short tail window reads as a level drop that the
    # audio does not have, which would fake a fade on every take.
    starts = range(0, max(len(mono) - window + 1, 1), window)
    rms = np.array([np.sqrt(np.square(mono[i : i + window]).mean()) for i in starts])
    db = 20 * np.log10(np.maximum(rms, 1e-6))

    # Side/mid tells mono-ish apart from wide. A turbo draft sits near 0.2-0.3
    # and a finished sft take nearer 0.8, so this separates them before a
    # listen does.
    if audio.shape[1] >= 2:
        mid = audio.mean(axis=1)
        side = (audio[:, 0] - audio[:, 1]) / 2
        mid_rms = np.sqrt(np.square(mid).mean())
        width = float(np.sqrt(np.square(side).mean()) / mid_rms) if mid_rms else 0.0
    else:
        width = 0.0

    # Percentile spread rather than max-minus-min. A take ending in [fade out]
    # has a near-silent last window, and max-minus-min reads that single window
    # as enormous dynamics: a flat draft measured 42.7 dB that way while its
    # envelope never moved. p90-p10 asks the question actually being asked --
    # does the level move *during* the song -- and the fade is reported on its
    # own line below.
    spread = float(np.percentile(db, 90) - np.percentile(db, 10))
    lo, hi = db.min(), db.max()
    shape = "".join(
        RAMP[min(len(RAMP) - 1, int((d - lo) / (hi - lo) * (len(RAMP) - 0.01)))]
        if hi > lo
        else RAMP[0]
        for d in db
    )
    return Take(
        name=path.name,
        seconds=len(mono) / rate,
        rate=int(rate),
        channels=int(audio.shape[1]),
        peak=float(np.abs(audio).max()),
        dynamic_range_db=float(spread),
        width=width,
        tail_db=float(db[-1] - db.max()),
        shape=shape,
    )


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv]
    if not paths:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: check-take.py TAKE.flac [TAKE.flac ...]", file=sys.stderr)
        return 2

    missing = [p for p in paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"no such file: {p}", file=sys.stderr)
        return 2

    # A glob like out/<name>/* picks up the lyric sheet and the plan alongside
    # the takes. Say which file is not audio rather than unwinding a
    # libsndfile traceback over it, and measure the rest of the batch anyway.
    rows = []
    for path in paths:
        try:
            rows.append(measure(path))
        except sf.LibsndfileError as exc:
            print(f"not audio, skipping: {path} ({exc})", file=sys.stderr)
    if not rows:
        print("nothing to measure.", file=sys.stderr)
        return 2
    width = max(len(r.name) for r in rows)
    print(f"{'take':<{width}}  {'length':>8}  {'peak':>8}  {'range':>7}  {'width':>6}")
    for r in rows:
        # Pinned rather than clipped: takes come back at 0.998993 across every
        # seed, which is a ceiling in the pipeline rather than four takes that
        # each happened to peak. Either way there is no headroom left for
        # whatever you do next, which is the thing worth saying.
        flag = " PINNED" if r.peak >= 0.99 else ""
        print(
            f"{r.name:<{width}}  {r.seconds:7.1f}s  {r.peak:8.4f}  "
            f"{r.dynamic_range_db:6.1f}dB  {r.width:6.2f}{flag}"
        )
    print()
    for r in rows:
        print(f"{r.name}\n  {r.shape}")
        # A tail well below the loudest window is what a real fade looks like;
        # a sheet ending in [fade out] that stops here at -2 dB did not fade.
        print(f"  last window {r.tail_db:+.1f} dB against the loudest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
