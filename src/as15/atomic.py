"""Install a file at its destination only once it is whole.

Both things this package writes -- the converted weight caches and the
generated audio -- are slow enough that an interruption part way through is a
real possibility, and both are read back later by something that cannot tell a
truncated file from a finished one. Writing to a temporary and renaming means
the destination only ever holds a complete file, and that a failed write
leaves whatever was already there untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import uuid4


def publish(target: Path, write: Callable[[Path], object]) -> None:
    """Install whatever *write* produces at *target*, atomically.

    The temporary lives in the destination directory so ``replace`` is a
    same-filesystem rename, and carries a random component so a second writer
    cannot rename the first one's half-finished file into place. It keeps
    *target*'s extension because both callers dispatch on it:
    ``mx.save_safetensors`` fails with a bare ``FileNotFoundError`` under any
    other name, and soundfile takes the container from it.
    """
    tmp = target.with_name(f"{target.stem}.{uuid4().hex[:8]}.tmp{target.suffix}")
    try:
        write(tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
