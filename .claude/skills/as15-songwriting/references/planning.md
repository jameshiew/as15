# Planning with the 5Hz LM (`--plan`)

Read this when a song is still vague and you want the LM to settle its
arrangement, when a take came back with a voice or an instrument nobody asked
for, or when you want one arrangement rendered several ways.

The README's **Planning** section covers the mechanics -- planner sizes, the
code format, plan-length rules, interoperability with upstream. This file is the
songwriting judgement on top of it.

## What it changes

By default the DiT starts from silence and writes the whole track from the
prompt and lyrics alone. `--plan` runs the planner LM first: it sketches the
song as one audio code per 200 ms and the DiT renders that sketch instead.

```bash
uv run as15 sing -p "..." -L lyrics.txt -d 180 --plan -o out/song.flac
```

| | Direct (default) | `--plan` |
| --- | --- | --- |
| Extra download | none | 1.2--8.4 GB, once |
| Extra time | none | ~2.5 min for a 2-minute plan on the 4B |
| Control | prompt and lyrics only | the plan also fixes the arrangement |

`--planner` picks the size: `0.6b`, `1.7b`, or `4b` (default, and upstream's own
pick for quality). The planner is loaded and released before conditioning, so it
does not raise the peak.

Size buys long-tail knowledge rather than general polish. The 4B earns its
8.4 GB on unusual genres, uncommon instruments and dense arrangements -- the
material a smaller model has thin coverage of. Upstream's own framing is that a
bigger LM does *not* automatically improve ordinary pop or rock, and a
four-on-the-floor house track is something all three have seen ten thousand
times. So `1.7b` is worth an A/B on straightforward material rather than assumed
to be worse; the comparison costs one plan each.

The planner also **fills in metas you left unset** -- it settles a bpm, a key
and a time signature and writes them into its reasoning block. Anything you set
with `--bpm` / `--key` / `--time-signature` overrides it, and the duration is
always yours. So `--plan` is a way to get a considered tempo and key rather than
`N/A`, without having to pick them yourself.

## Read the reasoning block

It is not only metas. The planner writes itself a full prose caption of the song
it is about to sketch, and **that caption can contradict your prompt.** A prompt
saying `warm female vocals` came back with:

```
caption: ... The lead vocal is delivered by a smooth male singer using expressive
falsetto, complemented by layered backing vocals including female harmonies ...
```

The DiT is conditioned on the plan *and* on your text, so the two then fight for
the whole render. The fix is the ordinary one for prompt ambiguity -- say it
twice, in different words (`warm female lead vocal, soulful female topline`) --
and plan again. A re-plan is one LM pass; finding it after a six-minute sft
render is not.

The block goes to stderr, buried in the planner's progress bar, so redirect and
read it back:

```bash
uv run as15 plan -p "..." -L lyrics.txt -d 200 -o out/song.codes 2>out/plan.log
${CLAUDE_SKILL_DIR}/scripts/read-caption.sh out/plan.log
```

Check it against the prompt before spending a render on it: the vocal, the
instrumentation, and whether the structure it describes is the one your lyric
tags asked for.

**The check is a filter, not a guarantee.** A caption that agrees with your
prompt means the planner heard you; the DiT can still deliver something else,
and nothing in this repo can verify the voice on a finished take -- not the
tags, which record what was asked for, and not `check-take.py`, which measures
level and width and says nothing about who is singing. Catching a disagreement
here is cheap and worth doing every time. Confirming the take is right is a
listening job, and it stays one.

## Freeze the plan, then render it several ways

Plans are worth keeping, and not only to save the time. The planner is most
useful while the song is still vague -- it will settle a tempo, a key and an
arrangement out of a thin brief, which is exactly the job you do not want to do
yourself at that stage. It is least useful once your specification is precise,
where a fresh `--plan` is one more opportunity to reinterpret a song you had
already decided: a re-plan re-rolls the caption, and the caption is what moved
the voice, the genre or the instrumentation last time. **Writing the plan to a
file is how you stop asking.** Use the LM early, then freeze what it gave you.

Planning is also one LM pass against fifty DiT passes, so write it once and
render it several ways:

```bash
uv run as15 plan -p "..." -L lyrics.txt -d 180 -o out/song.codes
uv run as15 sing -p "..." -L lyrics.txt -d 180 --audio-codes out/song.codes -g 5 -o out/a.flac
uv run as15 sing -p "..." -L lyrics.txt -d 180 --audio-codes out/song.codes -g 8 -o out/b.flac
```

The prompt, lyrics and duration passed to `sing` must be the ones the plan was
written for -- the plan is a sketch *of that song*, and the DiT is still
conditioned on the text as well. A plan shorter than the duration is rejected,
naming how many codes are needed. `--plan` and `--audio-codes` are mutually
exclusive: one writes a plan and the other supplies one.

## Seeds

`--seed` covers the whole run, plan included. Pin `--planner-seed` on its own to
keep one plan while `--seed` moves the render, which is how you hear what the
diffusion is contributing on top of the sketch. `--plan --takes 4` does that in
one command -- the plan is written once for the batch, because a plan is a
property of the song rather than of a take, and the four renders differ only in
their seed.

Every planned take stores its plan in `AS15_AUDIO_CODES`, along with
`AS15_PLANNER` and `AS15_PLANNER_SEED` when this run wrote it. A plan that
arrived in a file names no planner -- the take cannot vouch for what wrote it.
Recover one from a take you liked with:

```bash
ffprobe -v error -show_entries format_tags=AS15_AUDIO_CODES -of csv=p=0 out/a.flac > out/a.codes
```

## A plan crosses checkpoints, and a seed does not

This is the useful part. A turbo seed does not reproduce a turbo take on sft, so
the old loop was "draft on turbo, then audition sft takes until one lands". A
plan is just the arrangement, and both checkpoints read it the same way -- so
you can settle the arrangement on turbo, cheaply, and then render *that same
arrangement* on sft:

```bash
uv run as15 plan -p "..." -L lyrics.txt -d 200 -o out/song.codes
uv run as15 sing -m xl-turbo --audio-codes out/song.codes -p "..." -L lyrics.txt -d 200 -o out/draft.flac
uv run as15 sing --audio-codes out/song.codes -p "..." -L lyrics.txt -d 200 -o out/final.flac
```

The draft is no longer only a check on the words -- it is a preview of the shape
the final take will have.
