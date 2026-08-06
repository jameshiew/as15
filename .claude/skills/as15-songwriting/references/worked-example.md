# Worked example: "Hold the Morning"

The canonical run -- 4B plan, `xl-sft`, everything else at its default. Every
number below was measured on it. Read this when you want to see the whole
sequence end to end, or when a section of `SKILL.md` needs a concrete case.

**The spec** (SKILL.md §0), settled before anything else: a night nobody wants
to end; defiance turning into euphoria by the last drop; contemporary soulful
deep house; one warm female lead; Rhodes, sub bass, filtered pads, shuffled hats
over a four-on-the-floor kick; sparse verse rising through a build into a wide
drop, three times; the section order set out under **Duration** below.

**Genre first.** The brief was only "something the model generates well", so the
genre-fit table picked it: soulful deep house. Four-on-the-floor gives a 200 s
take a grid to hold, the hook repeats so the lyric budget goes into one line,
and the vocal sits under reverb -- which is where coarse vocal synthesis stops
mattering. It also avoids the named weak spot, guitar-forward rock.

**Prompt**

```
soulful deep house, warm female lead vocal, soulful female topline,
four-on-the-floor kick, deep round sub bass, dusty Rhodes chords,
filtered analog pads, crisp shuffled hi-hats, spacious plate reverb,
late-night club warmth
```

Thirty words over genre, voice, five instruments, texture and production --
measured at 51 tokens, so about a quarter of the budget. The voice is named
twice on purpose; see the re-plan below.

**Metas** -- `--bpm 122` (house is tempo-defined), `--key "D minor"`,
`--time-signature 4`, `--language en`.

**Duration** -- intro 14 + verse 16 + build 16 + drop 24 + breakdown 12 +
verse 16 + build 16 + drop 24 + bridge 14 + build 10 + drop 24 + outro 14 =
~200, so `-d 200`.

**Lyrics** (`out/hold-the-morning.txt`)

```
[intro]

[verse]
Nobody counted the hours
Nobody watched the door
The bassline knows my name
I don't need mine anymore

[build]
Four in the morning, still standing
Ceiling is starting to glow
Hold it, hold it, hold it
Don't let it go

[chorus - drop]
Hold the morning off a while
Every hand up in the light
We were never going home
Not tonight (not tonight)
Hold the morning off

[breakdown]

[verse]
Somebody's coat on the floor
Somebody's song on repeat
The whole room breathes as one
And nothing is waiting outside

[build]
Four in the morning, still standing
Ceiling is starting to glow
Hold it, hold it, hold it
Don't let it go

[chorus - drop]
Hold the morning off a while
Every hand up in the light
We were never going home
Not tonight (not tonight)
Hold the morning off

[bridge - low energy]
Let the light stand in the street
Let it wait outside for me

[build]
One more hour
One more hour
ONE MORE HOUR

[chorus - drop]
HOLD THE MORNING OFF A WHILE
EVERY HAND UP IN THE LIGHT
We were never going home
Not tonight (not tonight)
Hold the morning off

[outro]
Not tonight (not tonight)
[fade out]
```

Lines run 6--8 syllables and stay parallel across the two verses. One metaphor
-- refusing the dawn -- from three angles. `[build]` into `[chorus - drop]`
three times is the shape the genre already wants; the empty `[intro]` and
`[breakdown]` are deliberate instrumental sections; `(not tonight)` is a backing
vocal and the capitalised last chorus lifts it. The first two choruses are
word-for-word identical, which is what makes one listen enough to say whether
the model can land the same hook twice; only the third departs from them.

**Plan, and check it**

```bash
uv run as15 plan -p "$PROMPT" -L out/hold-the-morning.txt -d 200 \
  --bpm 122 --key "D minor" --time-signature 4 --seed 1122 \
  -o out/hold-the-morning.codes 2>out/plan.log
${CLAUDE_SKILL_DIR}/scripts/read-caption.sh out/plan.log
```

The first plan's caption described "a smooth male singer using expressive
falsetto" against a prompt that said female vocals. The prompt gained a second,
differently-worded statement of the voice (`warm female lead vocal, soulful
female topline`) and the plan was rewritten at a new seed; the second came back
with "a powerful female lead vocal ... layered backing vocals ... dynamic builds
leading into instrumental drops". 2 min 36 s each time on an otherwise idle
machine. **Do not skip this check** -- it is the cheapest step in the run and it
guards the most expensive one.

**Draft on turbo from that plan, then render on sft from the same plan**

```bash
uv run as15 sing -m xl-turbo --audio-codes out/hold-the-morning.codes \
  -p "$PROMPT" -L out/hold-the-morning.txt -d 200 \
  --bpm 122 --key "D minor" --time-signature 4 --seed 1123 \
  -o out/hold-the-morning-draft.flac
```

```bash
uv run as15 sing --audio-codes out/hold-the-morning.codes \
  -p "$PROMPT" -L out/hold-the-morning.txt -d 200 \
  --bpm 122 --key "D minor" --time-signature 4 --seed 1123 \
  -o out/hold-the-morning.flac
```

24 s of diffusion against 6 min 3 s, from one arrangement -- so the draft is a
real preview of the final's shape, not a different song at the same tempo.

## What you can check without listening

Listening is still the decision. But a few cheap measurements catch a dead run
before it wastes an audition, and they showed the sft take was the better one on
every axis:

| | draft (turbo) | final (sft) |
| --- | --- | --- |
| Dynamic range (p90--p10 of 4 s RMS windows) | 5.9 dB | **7.5 dB** |
| Stereo width (side/mid RMS) | 0.27 | **0.83** |
| Sample peak | 0.999, pinned to the ceiling | **0.877**, with headroom |
| Fade depth (last window against the loudest) | -8.9 dB | -15.5 dB |

An earlier version of this table reported 8.9 dB and 15.5 dB as *dynamic range*.
Those are the fade-depth numbers in the last row: they came from a max-minus-min
spread, and on a sheet ending in `[fade out]` the near-silent final window
dominates that figure entirely. So the old number was measuring how hard the
take faded, not how much the chorus opened up. Measured properly, the dynamic
gap between the two checkpoints is real but modest -- **the large, reliable
differences are stereo width and peak headroom.**

`scripts/check-take.py` measures all of it, and takes a whole batch at once:

```bash
${CLAUDE_SKILL_DIR}/scripts/check-take.py out/hold-the-morning*.flac
```

Worth checking on any take: that it is the length you asked for, that the peak
is not pinned, that the RMS envelope moves rather than sitting flat, and -- if
the sheet ends in `[fade out]` -- that the tail actually decays. This one went
from -14.8 dB to -61.7 dB over its last four seconds.

None of that says whether it is any good. It says whether it is worth your ears.
