"""Claims the README makes that the repository can check for itself.

Only the ones that are derivable. Measurements -- timings, peak memory, the
0.999378 correlation against the reference DiT -- cannot be re-derived from a
checkout and are not pinned here; what is pinned is prose that restates
something already written down somewhere else in the tree, because that is the
prose that goes stale without anyone touching the sentence.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_the_advertised_dependency_count_is_the_real_one():
    """The pitch is a dependency count, so it has to be the dependency count.

    It said nine while the list had grown to eleven -- a claim that is checked
    by the reader against ``pyproject.toml`` and by nobody against anything
    else. Adding a dependency now fails here, and the fix is to update the
    sentence deliberately rather than to discover the drift later.
    """
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())
    count = len(declared["project"]["dependencies"])

    claim = re.search(
        r"\*\*(\d+) runtime dependencies\*\*", (ROOT / "README.md").read_text()
    )
    assert claim is not None, "README no longer states a runtime dependency count"
    assert int(claim.group(1)) == count, (
        f"README says {claim.group(1)} runtime dependencies; pyproject declares "
        f"{count}. Update the sentence in README.md, or drop the dependency."
    )
