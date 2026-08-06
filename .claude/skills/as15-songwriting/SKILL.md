---
name: as15-songwriting
description: Write and generate songs with as15 (ACE-Step 1.5 XL on MLX). Covers settling the song before prompting, which genres the model renders well, style prompts, lyric sheets with structure tags, choosing duration/BPM/key/time signature, picking a checkpoint, planning with the 5Hz LM, the draft-then-render loop, choosing between takes, and fixing takes that come back wrong -- skipped lyrics, unrequested instruments, a chorus no bigger than the verse, harsh audio. Use whenever the user wants to write, plan, generate, improve or troubleshoot a song, a lyric sheet, a style prompt or a take in this repository, including when they only ask for "a command to make a track" or hand you a take that came back wrong.
allowed-tools: Read, Write, Bash(uv run as15:*), Bash(${CLAUDE_SKILL_DIR}/scripts/check-take.py:*), Bash(${CLAUDE_SKILL_DIR}/scripts/read-caption.sh:*), Bash(ffprobe:*)
---

# Songwriting for as15

`as15` generates exactly what you hand it. Nothing writes a style prompt for you
and nothing invents lyrics -- **you are the songwriter.** A meta you leave unset
is not inferred: it reaches the conditioning as the literal string `N/A` and the
model improvises around it. The one exception is `--plan`, which runs a planner
LM that settles a tempo and a key and sketches the arrangement before the DiT
starts. Even then the words and the style are yours.

So the job is to produce four things, then run one command:

| Output | Flag | Notes |
| --- | --- | --- |
| Style prompt | `-p` | Genre, instruments, voice, texture, production |
| Lyric sheet | `-L` (file) or `-l` (string) | Structure tags + words; omit for an instrumental |
| Musical metas | `--bpm`, `--key`, `--time-signature`, `--language` | Unset means `N/A`, not "guess" |
| Run settings | `-d`, `-m`, `--seed`, `--takes`, `-o` | Duration is a hard container for the lyrics |

```bash
uv run as15 sing -p "STYLE PROMPT" -L lyrics.txt -d 180 --bpm 96 --key "A minor" -o out/song.flac
```

Write lyric sheets to a file and pass `-L`. Multi-line lyrics through `-l` fight
the shell over quoting and newlines for no benefit. `out/` is gitignored, which
is where takes belong.

**Where the rest lives.** This file is the judgement -- the decisions the CLI
cannot make for you. Load a reference when you reach it. Paths below sit in this
skill's own directory, `${CLAUDE_SKILL_DIR}`; run the scripts by that full path,
since the working directory is the repo root rather than here. When you hand one
of those commands to someone to run themselves, **write the resolved path**, not
the variable -- their shell expands it to nothing and the command silently
becomes `/scripts/...`.

| File | Read it when |
| --- | --- |
| `references/troubleshooting.md` | A take came back wrong -- symptom by symptom, in the order worth trying |
| `references/planning.md` | Using `--plan`, or the planner changed the voice, genre or instrumentation |
| `references/sampling.md` | Tempted to override `-g`, `-s`, `--sampler`, `--dcw`; or choosing between finished takes |
| `references/worked-example.md` | You want one complete run end to end, with measured numbers |
| `references/after-the-render.md` | The take is worth keeping, you need to recover the settings from an existing take, or someone asks for a remix/extend as15 does not do |
| `scripts/read-caption.sh` | Reading a planner's reasoning block out of its log |
| `scripts/check-take.py` | Measuring a take, or a batch, before spending ears on it |

The repo's own `README.md` is the reference for mechanics -- planner sizes, the
Vorbis tag list, timings, `--takes` batching. It is accurate and one `Read` away.

---

## 0. Settle the song before you touch a flag

Answer these seven before writing anything as15 reads. Each one turns into a
flag or a line of the sheet, and answering them separately is what stops you
asking the model to invent the subject, the hook, the voice, the structure and
the production all at once.

| Question | Where the answer goes |
| --- | --- |
| What is the song about, in one sentence? | the lyric sheet |
| What should the listener feel by the last chorus? | the energy arc in `-p` |
| Genre and era? | `-p`, against §1's genre-fit table |
| Whose voice, and how do they sing? | `-p` |
| Which three to six instruments define it? | `-p` |
| How do verse, pre-chorus, chorus and bridge differ? | `-p` and the section tags |
| What is the section order? | the lyric sheet, and the arithmetic behind `-d` |

"A sad love song" answers none of them. This does:

> A nocturnal synth-pop song about realising, on the last train home, that a
> finished relationship has stopped having a vote. Close, sparse verses that
> open out into an optimistic chorus.

That is a direction for the model *and* a criterion for you: when four takes
come back you can say which one is the song, rather than which one is the
cleanest audio.

---

## 1. Style prompt (`-p`)

The single biggest lever on what comes back. Comma-separated tags, a natural
sentence, or a mix -- all work. Cover several dimensions rather than piling
detail onto one.

| Dimension | Examples |
| --- | --- |
| Genre | dream pop, boom bap, deep house, reggaeton, pop punk, lo-fi soul |
| Emotion | melancholic, euphoric, menacing, wistful, defiant, intimate |
| Instruments | brushed drums, upright bass, 808s, Rhodes, string section, detuned synth |
| Voice | female vocals, male vocals, breathy, raspy, belted, choir, double-tracked |
| Texture | warm, crisp, airy, saturated, muddy, glassy, punchy |
| Era / reference | 80s synth-pop, 90s boom bap, 2000s indie, vintage soul |
| Production | analog tape, bedroom recording, studio-polished, heavily compressed, wide reverb |
| Energy arc | sparse close verses, rising pre-chorus, wide layered chorus, half-time bridge |

**Rules that are specific to as15:**

- **Do not put BPM, key, time signature or duration in the prompt.** They have
  their own flags and land in a separate `# Metas` block of the trained prompt
  format (`src/as15/conditioning.py`). Saying "120 BPM" in the prompt spends
  tokens telling the model something it is also being told properly, and the two
  can disagree.
- **There is nowhere to put a negative.** as15 has no negative prompt, and CFG's
  unconditional branch is a stored null embedding rather than text you supply,
  so "no vocals" or "no guitars" just puts the words *vocals* and *guitars* into
  your caption. State a short positive palette instead, and say the thing that
  matters twice in different words.
- **The prompt has a hard 256-token budget**, and 53--55 of those are spent on
  the instruction and metas lines wrapped around it -- so about 200 tokens for
  your text. Measured on the tokenizer, comma-separated tag prompts run about
  1.6--2.1 tokens a word, so **200 tokens is nearer 100--125 words than 150.**
  Prose runs leaner, around 1.3. Over budget is **rejected**, not truncated: the
  run stops before generating and tells you how many tokens to cut.
- A tight prompt of 15--40 words is usually the sweet spot -- well inside the
  budget. More detail is more control and less room for the model to be
  interesting; less is the reverse.

**Craft:**

1. Specific beats vague -- "sparse piano ballad, breathy female vocal, close-mic'd,
   tape hiss" over "a sad song".
2. Avoid contradictions ("string quartet" + "hardcore metal"). The model will not
   arbitrate; it averages, and the average is mud. If you want a shift, describe
   it as one: "opens on solo cello, breaks into distorted guitars at the chorus".
3. Repetition reinforces. Naming a texture twice in different words biases harder
   than naming it once.
4. Reference tags ("in the style of 80s synthwave") carry a lot of aesthetic per
   token -- cheaper than enumerating the same thing instrument by instrument.
   Name eras, scenes and traditions rather than artists: "brooding 80s
   synthwave, gated snare, arpeggiated bass" is the part you actually wanted,
   and it does not bet the take on a catalogue nobody here has a licence to.
5. Keep the prompt and the lyric tags consistent. If the prompt says piano ballad,
   a `[guitar solo]` tag is a fight the output loses.
6. **Describe audible decisions, not judgements.** "Epic, emotional, beautiful,
   powerful, catchy, atmospheric" is how you would like the review to read;
   nobody in a studio can act on it, and neither can the model. "Sparse
   close-mic'd verses opening into wide layered choruses" is a set of
   instructions. Every adjective should imply a decision someone had to make.
7. **Say how the song moves, not only how it sounds.** A prompt that describes
   one texture gets a take with that texture end to end -- the flattest failure
   the model has, and the one users complain about most. Spend a clause on the
   arc: "sparse verses build through rising pre-choruses into wide choruses;
   brief half-time bridge". The section tags mark where; the prompt says by how
   much.

### Genre fit

Genre is the first choice and the model is not equally good at all of them. From
upstream's own [1.5 page](https://ace-step.github.io/ace-step-v1.5.github.io/)
and the reported results in
[discussion #235](https://github.com/ace-step/ACE-Step-1.5/discussions/235):

| Renders reliably | Struggles |
| --- | --- |
| House, techno, drum and bass, lo-fi | Guitar-forward punk and metal -- comes back "poppy-rock sounding" |
| Pop, synth-pop, indie | Rap outside English; `zh_rap` is named on the model card |
| Folk and acoustic singer-songwriter | Lyrics in any language the take is not conditioned for |
| Orchestral and cinematic | Several niche genres stacked -- the signals cancel and the output muds |
| Boom bap, trap, lo-fi hip hop | |

Two of upstream's stated weaknesses are worth designing around rather than
fighting. **Coarse vocal synthesis lacking nuance**: a genre where the vocal is
processed -- house, synth-pop, shoegaze -- hides it, and a style whose whole
appeal is an unadorned voice a foot from the microphone is the hardest thing you
can ask for. **Limited multilingual lyrics compliance**: English is the safe
default and `--language` has to match the words either way.

A steady grid is also the cheapest thing you can give a long take. Four-on-the-floor
at a fixed `--bpm` gives the model something to hold across three or four
minutes; the same length of rubato has nothing keeping it honest.

---

## 2. Lyric sheet (`-L` / `-l`)

Lyrics carry the words *and* the timeline: structure, energy, where the vocal
stops. Budget is **2048 tokens** including the language header -- around 1500
words, far more than a song needs -- and over budget is rejected the same way.

### Structure tags

The README writes them lowercase (`[verse]`, `[chorus]`). Stay consistent within
a sheet.

| Group | Tags |
| --- | --- |
| Core | `[intro]` `[verse]` `[pre-chorus]` `[chorus]` `[bridge]` `[outro]` |
| Dynamics | `[build]` `[drop]` `[breakdown]` |
| No vocal | `[instrumental]` `[guitar solo]` `[piano interlude]` `[drum break]` |
| Ending | `[fade out]` |

Numbered repeats (`[verse 1]`, `[verse 2]`) are fine. A modifier after a dash is
fine and should stay short: `[chorus - anthemic]`, not
`[chorus - anthemic - stacked harmonies - huge - epic]`. Long descriptions belong
in `-p`.

Separate sections with a blank line.

### Vocal and energy tags

`[whispered]` `[raspy vocal]` `[falsetto]` `[belted]` `[spoken word]`
`[harmonies]` `[call and response]` `[ad-lib]`
`[low energy]` `[building energy]` `[high energy]` `[explosive]`

Use them sparingly and only where the prompt already supports them. These are
conventions rather than a validated vocabulary -- nothing here tests generation
-- so confirm any non-core tag by listening.

### Writing that sings

- **6--10 syllables a line.** Lines in the same structural position should match
  within a syllable or two, so the model can settle into a meter.
- **Repeat the chorus verbatim while you are drafting.** Identical words each
  time is the only way to hear whether the model can land the same hook twice --
  if the second chorus drifts, that is the model, not your rewrite. Vary the
  final chorus, add ad-libs, change a line for the last repeat once the shape is
  proven, not before.
- **CAPITALS read as intensity.** `HOLD THE LINE` is shouted; `hold the line` is
  not. Use it on a hook, not a verse.
- **Parentheses read as backing vocals**: `We rise again (again)`.
- **Held vowels are unreliable.** `hoooold ooon` is a coin toss; write `hold on`
  and let the melody hold it. Same for long runs of repeated letters anywhere in
  a line.
- **One breath per line.** If you cannot say it out loud in one go, it will not
  be sung in one either.
- **One controlling metaphor per song**, explored from several angles. Water then
  fire then flight gives the listener nothing to hold.
- **Watch for AI-flavoured filler**: stacked adjectives ("neon skies, electric
  hearts, endless dreams"), rhymes forced at the cost of sense, and content that
  spills across a section boundary so the chorus arrives mid-thought.

### Instrumentals

Omit lyrics entirely -- there is no flag for it. The CLI warns on stderr that it
is generating an instrumental, and the FLAC gets no `LYRICS` tag. Put the
arrangement in `-p`, and keep every vocal word out of it: "female vocals",
"topline", "choir", "sung". Writing "no vocals" does the opposite of what it
looks like -- see the negative-prompt rule in §1.

Upstream's own guide offers a second form: a sheet of section tags and no words,
as a map of an instrumental piece.

```text
[intro - ambient pads]
[main theme - piano and restrained strings]
[development - rising percussion]
[climax - full orchestra]
[outro - solo piano, fade out]
```

as15 passes that through as an ordinary lyric sheet -- it is not empty, so there
is no instrumental warning and the FLAC does carry a `LYRICS` tag, and the
words in those tags are as available to be sung as any other. **Treat it as an
experiment, not a technique**: empty lyrics is the form that reliably produces
an instrumental. If you try it and a voice appears, that is the expected
failure, not bad luck -- fall back to empty lyrics and put the arrangement
in `-p`.

---

## 3. Metas (`--bpm`, `--key`, `--time-signature`, `--language`)

Nothing infers these. Unset renders as `- bpm: N/A` in the conditioning, which is
a legitimate choice -- the model picks -- but if the song depends on a tempo or a
feel, say so.

| Flag | Accepts | Notes |
| --- | --- | --- |
| `--bpm` | whole number, **30--300** | Beats per minute, never bars -- in 3/4 a bar is three of them. Slow 60--80, mid 90--120, fast 130--180 |
| `--key` | free text, one line | `"C major"`, `"A minor"`, `"F# minor"`. Common keys behave best |
| `--time-signature` | **`2`, `3`, `4` or `6`** | A bare integer -- `3` for a waltz, **not** `"3/4"`. Anything else is rejected |
| `--language` | code, default `en` | Lowercased, written into the `# Languages` header; must match the lyrics |

`--time-signature` is the one that catches people: it takes the numerator on its
own rather than a `4/4`-style string, and the valid set is fixed at
`(2, 3, 4, 6)` -- the signatures the metas block was trained with. Anything else
is rejected before the run starts.

All four are settled before anything loads: a tempo becomes a whole number, a
blank `--key` is the same as leaving it off rather than a key with no name, and
`--language` is stripped and lowercased. What the conditioning is told is what
the file's `AS15_*` tags record, so a take's metas can be typed straight back
in.

Set `--bpm` when you need to match other material, when the genre is tempo-defined
(house ~124, boom bap ~90, drum and bass ~174), or when the lyric density needs a
specific pace to fit. Set `--key` when you want a specific colour or a fixed
vocal range. Otherwise leaving them off is fine -- and if you would rather have a
considered choice than `N/A`, `--plan` makes one and prints it
(`references/planning.md`).

---

## 4. Duration (`-d`)

10 to 600 seconds, default 120. **This is a container, not a hint** -- the model
has to fit the whole lyric sheet into it. Too short and the delivery is crammed
and verses go missing; too long and you get instrumental drift at the end. The
120 default is not a decision; a run that never set `-d` has not chosen a length.

Rough section costs at 90--120 BPM in 4/4:

| Section | Bars | Seconds |
| --- | --- | --- |
| Intro / outro | 4--8 | 5--15 each |
| Verse (8 lines) | 16 | 25--35 |
| Pre-chorus (4 lines) | 4--8 | 12--18 |
| Chorus (4--8 lines) | 8--12 | 20--30 |
| Bridge | 8 | 15--25 |
| Instrumental / solo | 4--8 | 10--20 |

| Shape | Duration |
| --- | --- |
| Verse--chorus--verse--chorus | 110--150 |
| Plus pre-choruses and a bridge | 170--220 |
| Plus intro, outro and a solo | 210--280 |

Sanity check before generating: **sung lines x ~4 s, plus the instrumental
sections.** If that exceeds `-d`, cut lines or raise the duration. Slower tempos
need more seconds for the same words; faster ones fit more but still need air.
When in doubt, go longer -- a song with room to breathe beats one that rushes.

That check counts sung lines, so **an instrumental has nothing to count** -- go
straight to the bar arithmetic below, budgeting bars per section from the
arrangement you described in `-p`. It is the only estimate available when there
is no sheet, and it is the better one either way.

Once `--bpm` is set, count the arrangement in bars instead and the estimate stops
being a guess:

```text
seconds = bars x beats-per-bar x 60 / bpm
```

96 bars of 4/4 at 112 BPM is `96 x 4 x 60 / 112` = 206 s, so ask for 205--215.
The 4 s a line above is the same arithmetic with the bars left implicit -- about
two bars a line at 120 BPM -- and it drifts once the tempo does. Round up: the
sung sections are the ones that suffer when the container is tight, and the
instrumental ones absorb the slack.

Upstream calls two to four minutes the band where structure holds. Below it the
form has nowhere to happen; well above it, expect repetition and drift late in
the take, whatever the sheet says. The 600 s ceiling is what the CLI accepts, not
a length worth aiming at.

---

## 5. Checkpoint and sampling

```bash
uv run as15 models
```

| | `xl-sft` (default) | `xl-turbo` |
| --- | --- | --- |
| Steps | 50 | 8 |
| Guidance | 7.0 | none (distilled; `-g` forced to 1.0) |
| Shift | 1.0 | 3.0 |
| DCW | off | on |
| Relative speed | 1x | ~15x |
| Use for | the take you keep | drafting and iteration |

**Iterate on turbo, render on sft.** That ratio is the whole workflow. It is 15x
and not the 6x of the step counts because sft runs CFG, which is two forward
passes per step. The README's **Timings** section has the measured numbers; the
one worth carrying in your head is that a full-length sft take is minutes, not
the ~18 that scaling the README's 30 s row would predict -- a large part of every
step is fixed work over the text conditioning, so per-second cost falls as the
take gets longer.

**Everything else leaves its default.** `-s`, `--dcw`, `--precision` and
`--sampler` already follow the checkpoint, and the two that look most like
quality knobs are the likeliest to cost you a render: `-s 30` on sft is *below*
its 50, not above it, and forcing DCW on a non-distilled checkpoint makes output
mushy. `-g` between 5 and 7 is the one worth varying, on sft only.
`references/sampling.md` has the reasoning; read it before overriding any.

Planning is the other axis and has its own file: `references/planning.md`. In
short -- `--plan` sketches the arrangement with a 5Hz LM before the DiT starts,
it settles any metas you left unset, and **its reasoning caption can contradict
your prompt**, so read the caption before paying for a render.

---

## 6. The loop

1. **Plan.** Style prompt, structure, duration, metas. Write the lyric sheet to a
   file, e.g. `lyrics.txt` or `out/<name>.txt`.
2. **Draft on turbo** at a fixed seed so changes are attributable to the change
   rather than to the draw:

   ```bash
   uv run as15 sing -m xl-turbo --seed 42 -p "..." -L lyrics.txt -d 180 -o out/draft.flac
   ```

3. **Listen, change one thing, regenerate.** Prompt wording, a lyric line, the
   duration. Same seed throughout.

   Freezing the draw is what makes a comparison mean anything, and `--seed` is
   not the only randomness in a planned run: either pin `--planner-seed` too, or
   -- better -- render from a plan file so the LM does not run again at all.

   | Comparisons worth making | Comparisons that teach you nothing |
   | --- | --- |
   | Same seed, one instrument changed in `-p` | A new prompt and a new seed |
   | Same seed, one verse shortened | A prompt change and a duration change |
   | Same seed and plan, `-g 5` against `-g 7` | A checkpoint change and a prompt change |
   | Same plan, turbo against sft | A re-plan and anything else |
   | Same everything, four consecutive seeds | Four takes of four different songs |

   The right column is how most sessions actually go, and it is why they end
   with a take nobody can reproduce and no idea which change earned it.
4. **Explore takes** once the prompt is right. The seed is the whole difference
   between two takes of a settled song, so generate several at once:

   ```bash
   uv run as15 sing -m xl-turbo --seed 100 --takes 4 -p "..." -L lyrics.txt -d 180 -o out/take.flac
   ```

   That writes `out/take-01-seed-100.flac` through `out/take-04-seed-103.flac`,
   sharing one conditioning pass and one load of each model rather than paying
   for four. Listen; keep the seed of the one that lands. A take is a function
   of its seed, so re-rendering that seed on its own gives the same audio back.

   **Choose the song, not the cleanest audio.** A take that scored zero on the
   hook is not a candidate however spotless the rest of it is -- polishing a
   strong composition is ordinary work, rescuing a clean forgettable one is
   starting again with extra steps. `references/sampling.md` has the seven-axis
   score sheet that turns that into a decision rather than a vibe.

   `${CLAUDE_SKILL_DIR}/scripts/check-take.py out/take-*.flac` measures length, peak, dynamic range
   and stereo width across a batch, and draws each take's RMS envelope. It says
   which takes are worth your ears, not which one is good.
5. **Render on sft** with the seed and settings that worked:

   ```bash
   uv run as15 sing --seed 42 -p "..." -L lyrics.txt -d 180 -o out/final.flac
   ```

   A turbo seed does not reproduce the same take on sft -- different checkpoint,
   different schedule. Expect to audition a few, or write a plan first and pass
   the same `--audio-codes` to both, which does carry across
   (`references/planning.md`).

For the take you actually keep, the highest-quality configuration is `xl-sft`
with a 4B plan, everything else left at its default:

```bash
uv run as15 plan -p "..." -L lyrics.txt -d 200 --bpm 72 --key "A minor" -o out/song.codes 2>out/plan.log
${CLAUDE_SKILL_DIR}/scripts/read-caption.sh out/plan.log      # does the caption match the prompt?
uv run as15 sing --audio-codes out/song.codes -p "..." -L lyrics.txt -d 200 --bpm 72 --key "A minor" -o out/final.flac
```

The middle line is not optional. Reading the plan's caption costs nothing and
catches a prompt the planner misread before the render pays for it.

Then vary one thing at a time from that same plan -- the seed, or `-g` between 5
and 7 -- and keep the take that sounds best. Nothing in this repo can tell you
which that is; it is a listening decision.

Output must be `.flac`; the path is checked before the run, not after 10 minutes
of diffusion. An existing file is overwritten and you are warned first. Every
take carries its own recipe in Vorbis comments, so a take can always be
re-rendered from itself -- see `references/after-the-render.md`.

Pre-fetch weights before a first run so the download is not mistaken for a hang:

```bash
uv run as15 download -m xl-sft
```

---

## 7. When it comes back wrong

**Read `references/troubleshooting.md`** -- it maps each symptom to the changes
worth making, in the order worth making them. Almost none of them are sampler
problems, and reaching for `-s` first is the most reliable way to spend fifteen
minutes and learn nothing.

It covers: lines skipped or a verse raced through; an instrument nobody asked
for; a chorus no bigger than the verse; harsh, flat or hissy audio; the planner
changing the voice, genre or language; vocals in something meant to be
instrumental; the wrong language or accent; a run that refuses to start; a whole
batch wrong the same way; and a take that is fine but forgettable.

Two are worth knowing without opening anything, being the commonest and the most
misdiagnosed. **A crammed `-d` is the usual cause of skipped and rushed
lyrics** -- redo the §4 arithmetic first, and note that a run still on the 120 s
default never chose a length at all. And **a chorus no bigger than the verse is
a prompt problem**, not a sampler one: put the arc in `-p`, not only in the tags.

---

## 8. After the render

Once a take is worth keeping -- comping, fixing dynamics, recovering a recipe
from a file, what as15 structurally cannot do (no remix, extend, repaint or
reference audio), and upstream's guidance on originality -- see
`references/after-the-render.md`.
