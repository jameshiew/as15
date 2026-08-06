#!/usr/bin/env bash
# The planner's reasoning block, out of the log it is buried in.
#
# `as15 plan` reasons on stderr -- a bpm, a key, a time signature and a prose
# caption of the arrangement -- and the caption can contradict the prompt. It
# arrives interleaved with a carriage-returned progress bar, so it needs the
# returns turned into newlines and the bar filtered out before it can be read.
#
#     uv run as15 plan -p "..." -L lyrics.txt -d 200 -o out/song.codes 2>out/plan.log
#     scripts/read-caption.sh out/plan.log
#
# Check the caption against the prompt -- the vocal, the instrumentation, and
# whether the structure it describes is the one the lyric tags asked for --
# before spending a render on it. Re-planning costs one LM pass; finding the
# disagreement in the finished take costs the render.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: read-caption.sh PLAN.log" >&2
    echo "  where PLAN.log is the stderr of a \`uv run as15 plan\` run" >&2
    exit 2
fi

if [ ! -f "$1" ]; then
    echo "no such file: $1" >&2
    exit 2
fi

# A plan log always contains a `caption:` line; a render log never does. Check
# that first rather than filtering and hoping: handed render.log -- which sits
# in the same directory and is the likeliest wrong argument -- the filter
# approach exits 0 having printed the whole DiT progress bar. tqdm renders as
# `8.38s/it]` once a step takes over a second, so a guard filtering only
# `it/s]` misses it entirely and fails open with garbage.
block=$(tr '\r' '\n' <"$1" | grep -viE 'planning:|fetching|[0-9]+(\.[0-9]+)?(it/s|s/it)\]|^[[:space:]]*$')

if ! printf '%s\n' "$block" | grep -q 'caption:'; then
    echo "no planner reasoning block in $1 -- it has no 'caption:' line." >&2
    echo "Is it the stderr of an \`as15 plan\` run? A render log will not have one." >&2
    exit 1
fi

printf '%s\n' "$block"
