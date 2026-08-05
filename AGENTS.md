# AGENTS.md

Guidance for agents working in this repository. `CLAUDE.md` is a symlink to
this file.

## Commits

### Before committing

Run the full gate suite and let it pass:

```bash
just check
```

That is `ruff check`, `tombi lint`, `ruff format --check`, `tombi format
--check`, `ty check`, `pytest`, in the order that gives the most useful first
failure. `just tidy` formats and autofixes first, then runs the same gates.
Every one of those runs under `uv run` from the dev dependency group, so
`uv sync` is the only setup -- do not reach for a globally installed tool.

The tests do not cover generation. They do run models -- a reduced-width DiT,
VAE and planner LM, built in the test and compared against a reference -- but
never a checkpoint, and nothing in them is an end-to-end generation. A change that
could alter generated audio therefore still has to be verified by generating,
and the commit message says how. Past commits have used: conditioning output compared
bit-for-bit, reconverted weights compared by SHA-256, and a full regeneration
at a fixed seed compared byte-for-byte against the previous build. If a change
is provably audio-neutral for a structural reason, say that reason instead.

### Subject

One line, imperative mood, capitalised, no trailing full stop, under ~55
characters. It names the change, not the files touched.

```
Upgrade to Python 3.14
Depend on typer directly, not typer-slim
Add ruff, ty and a Justfile; fix the lints they found
```

### Body

Wrap at ~75 columns. Blank line after the subject.

Explain **why**, and what was ruled out. The diff already shows what changed;
the body is for the reasoning that is not recoverable from it -- the upstream
behaviour that forced the change, the constraint behind a version cap, the bug
a defensive branch exists for, why an obvious simpler approach does not work.
State findings plainly, including ones that contradict an earlier assumption in
the codebase.

Then say how the change was verified, concretely enough to re-run.

Use ASCII throughout: `--` for an em dash, not `—`. Backticks around identifiers
are optional and used sparingly. Bullet lists are fine for enumerating several
independent findings.

### Trailer

Every commit an agent contributed to ends with a blank line and a
`Co-Authored-By:` naming the model that actually did the work -- not a fixed
name. An agent running as Opus 5 writes:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

and one running as anything else substitutes its own model name. Do not add
any other generated-by marker, and do not put the trailer on commits the agent
did not touch.

Commits by agents should not be signed.

### Scope

One logical change per commit. Mechanical churn (a formatter pass, a lockfile
regeneration) goes in the same commit as the change that caused it, and the
body says which lines are churn -- see the Python 3.14 commit, where the lock
diff is explained as dropped cp312/cp313 wheels rather than moved versions.

Do not commit generated audio (`out/`, `*.wav`) or the `.venv`; `.gitignore`
already covers them.

### Branching

It's fine to commit directly to `main`, and it's fine to commit proactively
without being asked. Push only when asked to.
