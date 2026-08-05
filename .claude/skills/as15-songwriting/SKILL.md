---
name: as15-songwriting
description: Write and generate songs with as15 (ACE-Step 1.5 XL on MLX). Covers settling the song before prompting, which genres the model renders well, style prompts, lyric sheets with structure tags, choosing duration/BPM/key/time signature, picking a checkpoint, planning with the 5Hz LM, the draft-then-render loop, choosing between takes, and fixing takes that come back wrong -- skipped lyrics, unrequested instruments, a chorus no bigger than the verse, harsh audio. Use when the user wants to create, write, plan, generate or troubleshoot a song in this repository.
allowed-tools: Read, Write, Bash
---

# Songwriting for as15

`as15` generates exactly what you hand it. Nothing writes a style prompt for you
and nothing invents lyrics -- **you are the songwriter.** A meta you leave unset
is not inferred: it reaches the conditioning as the literal string `N/A` and the
model improvises around it.

The one exception is `--plan` (§6), which runs a planner LM that will settle a
tempo and a key for you and sketch the arrangement before the DiT starts. Even
then the words and the style are yours.

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
- **The prompt has a hard 256-token budget**, and roughly 54 of those are spent
  on the instruction and metas lines wrapped around it -- so about 200 tokens,
  call it 150 words, for your text. Over budget is **rejected**, not truncated:
  the run stops before generating and tells you how many tokens and roughly how
  many characters to cut.
- A tight prompt of 15--40 words is usually the sweet spot. More detail is more
  control and less room for the model to be interesting; less is the reverse.

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
   Name eras, scenes and traditions rather than artists: an artist's name is a
   bet on what the model memorised of a catalogue you have no licence to, and
   "brooding 80s synthwave, gated snare, arpeggiated bass" is the part you
   actually wanted anyway.
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
conventions rather than a validated vocabulary -- nothing in this repo tests
generation (see AGENTS.md), so treat any tag beyond the core structure ones as
something to confirm by listening.

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

Omit lyrics entirely. The CLI warns on stderr that it is generating an
instrumental, and the FLAC gets no `LYRICS` tag. Put the arrangement in `-p`.

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
is no instrumental warning and the FLAC does carry a `LYRICS` tag. Whether it
buys you structure or gets the tags sung is a listening question; if a voice
appears, fall back to empty lyrics and describe the arrangement in `-p`.

---

## 3. Metas (`--bpm`, `--key`, `--time-signature`, `--language`)

Nothing infers these. Unset renders as `- bpm: N/A` in the conditioning, which is
a legitimate choice -- the model picks -- but if the song depends on a tempo or a
feel, say so.

| Flag | Accepts | Notes |
| --- | --- | --- |
| `--bpm` | whole number, **30--300** | slow 60--80, mid 90--120, fast 130--180 |
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
considered choice than `N/A`, `--plan` (§6) makes one and prints it.

---

## 4. Duration (`-d`)

10 to 600 seconds, default 120. **This is a container, not a hint** -- the model
has to fit the whole lyric sheet into it. Too short and the delivery is crammed
and verses go missing; too long and you get instrumental drift at the end.

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
| Guidance | 7.0 | none (distilled; `-g` ignored) |
| Shift | 1.0 | 3.0 |
| DCW | off | on |
| Relative speed | 1x | ~15x |
| Use for | the take you keep | drafting and iteration |

**Do not scale the README's 30 s timings linearly** -- they overstate a
full-length take by around 3x. A large part of every step is fixed work over the
text conditioning, so the per-second cost falls as the take gets longer. Measured
on an M5, 32 GB, for a **200 s** song:

| Stage | Time |
| --- | --- |
| 4B plan (1000 codes) | 2 min 36 s |
| `xl-turbo` diffusion | 24 s |
| `xl-sft` diffusion | 6 min 3 s |
| VAE decode (either) | ~36 s |

Peak was 9.1--9.2 GB either way, since decode is chunked. A full-length sft render
is minutes rather than the ~18 that linear scaling predicts -- but turbo is still
**15x** cheaper, and that ratio is the whole workflow: iterate on turbo, render on
sft. (It is 15x and not the 6x of the step counts because sft runs CFG, which is
two forward passes per step.)

- `-g / --guidance` (sft only): 7.0 is the default. Lower (3--5) follows the
  prompt more loosely and often sounds more natural; higher is more literal and
  can get brittle. `1.0` disables CFG and halves the cost. Upstream's own
  documentation contradicts itself over which checkpoints take CFG at all -- the
  model zoo says sft and base, the Gradio guide describes the field as base-only
  -- so here the checkpoint decides and the table above is the answer. A recipe
  from elsewhere quoting a guidance number for turbo is describing a surface that
  is not this one.
- `-s / --steps`: leave alone unless you are deliberately trading quality for
  time. Below the checkpoint default, quality drops off quickly. Above it is
  mostly folklore: turbo at 20 or 50 steps circulates as a fix for skipped words,
  but it is unreplicated single-user reporting, it costs the 15x that made turbo
  worth using, and 8 is the count the checkpoint was distilled for. Skipped words
  are cheaper to fix in the sheet -- see §8.
- `--sampler heun` evaluates the model twice per step for a second-order step --
  roughly double the diffusion time. Most worthwhile on 8-step turbo, where the
  step count is small enough that accuracy per step matters.
- **`--dcw` / `--no-dcw`: leave alone.** as15 already follows the checkpoint --
  off for `xl-sft`, on for `xl-turbo`. Wavelet-domain correction was tuned for
  the distilled models, and forcing it on a non-distilled one makes output mushy
  and distorted. The override is there for experiments, not for tuning a take.
- `--precision`: `bf16`. `fp32` doubles memory for no measurable gain.

---

## 6. Planning (`--plan`)

By default the DiT starts from silence and writes the whole track from the
prompt and lyrics alone. `--plan` runs the **5Hz planner LM** first: it sketches
the song as one audio code per 200 ms and the DiT renders that sketch instead.

```bash
uv run as15 sing -p "..." -L lyrics.txt -d 180 --plan -o out/song.flac
```

| | Direct (default) | `--plan` |
| --- | --- | --- |
| Extra download | none | 1.2--8.4 GB, once |
| Extra time | none | ~2 min for a 2-minute plan on the 4B |
| Control | prompt and lyrics only | the plan also fixes the arrangement |

`--planner` picks the size: `0.6b`, `1.7b`, or `4b` (default, and upstream's own
pick for quality). The planner is loaded and released before conditioning, so it
does not raise the peak.

Size buys long-tail knowledge rather than general polish. The 4B earns its 8.4 GB
on unusual genres, uncommon instruments and dense arrangements -- the material a
smaller model has thin coverage of. Upstream's own framing is that a bigger LM
does *not* automatically improve ordinary pop or rock, and a four-on-the-floor
house track is something all three have seen ten thousand times. So `1.7b` is
worth an A/B on straightforward material rather than assumed to be worse; the
comparison costs one plan each.

The planner also **fills in metas you left unset** -- it settles a bpm, a key and
a time signature and writes them into its reasoning block, which is printed on
stderr. Anything you set with `--bpm` / `--key` / `--time-signature` overrides
it, and the duration is always yours. So `--plan` is a way to get a considered
tempo and key rather than `N/A`, without having to pick them yourself.

### Read the reasoning block

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

The block goes to stderr, buried in the planner's progress bar, so it needs
unpicking to read:

```bash
uv run as15 plan -p "..." -L lyrics.txt -d 200 -o out/song.codes 2>plan.log
tr '\r' '\n' < plan.log | grep -viE 'planning:|Fetching'
```

Check it against the prompt before spending a render on it: the vocal, the
instrumentation, and whether the structure it describes is the one your lyric
tags asked for.

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
naming how many codes are needed.

Seeds: `--seed` covers the whole run, plan included. Pin `--planner-seed` on its
own to keep one plan while `--seed` moves the render, which is how you hear what
the diffusion is contributing on top of the sketch. `--plan --takes 4` does that
in one command -- the plan is written once for the batch, because a plan is a
property of the song rather than of a take, and the four renders differ only in
their seed.

Every planned take stores its plan in `AS15_AUDIO_CODES`, along with
`AS15_PLANNER` and `AS15_PLANNER_SEED` when this run wrote it. A plan that
arrived in a file names no planner -- the take cannot vouch for what wrote it.
Recover one from a take you liked with:

```bash
ffprobe -v error -show_entries format_tags=AS15_AUDIO_CODES -of csv=p=0 out/a.flac > out/a.codes
```

**A plan crosses checkpoints, and a seed does not.** This is the useful part. A
turbo seed does not reproduce a turbo take on sft, so the old loop was "draft on
turbo, then audition sft takes until one lands". A plan is just the arrangement,
and both checkpoints read it the same way -- so you can settle the arrangement on
turbo, cheaply, and then render *that same arrangement* on sft:

```bash
uv run as15 plan -p "..." -L lyrics.txt -d 200 -o out/song.codes
uv run as15 sing -m xl-turbo --audio-codes out/song.codes -p "..." -L lyrics.txt -d 200 -o out/draft.flac
uv run as15 sing --audio-codes out/song.codes -p "..." -L lyrics.txt -d 200 -o out/final.flac
```

The draft is no longer only a check on the words -- it is a preview of the shape
the final take will have.

---

## 7. The loop

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

   **Choose the song, not the cleanest audio.** Score each take 0--2 on seven
   things and the choice stops being a vibe:

   | Dimension | Question |
   | --- | --- |
   | Hook | Can you remember the chorus after one listen? |
   | Form | Are the sections distinct, and in the order the sheet asked for? |
   | Prosody | Do the words sit on the beat, or fight it? |
   | Vocal | Convincing, and intelligible? |
   | Groove | Do the drums and bass move on purpose? |
   | Dynamics | Is the chorus genuinely bigger than the verse? |
   | Audio | Clicks, clipping, hiss, harshness? |

   Do not let the total decide on its own: a take that scored zero on Hook is
   not a candidate however spotless the rest of it is. Polishing a strong
   composition is ordinary work; rescuing a clean forgettable one is starting
   again with extra steps. Audio is also the axis most likely to improve on its
   own at step 5 -- see the measurements in the worked example.
5. **Render on sft** with the seed and settings that worked:

   ```bash
   uv run as15 sing --seed 42 -p "..." -L lyrics.txt -d 180 -o out/final.flac
   ```

   A turbo seed does not reproduce the same take on sft -- different checkpoint,
   different schedule. Expect to audition a few, or write a plan first (§6) and
   pass the same `--audio-codes` to both, which does carry across.

For the take you actually keep, the highest-quality configuration is `xl-sft`
with a 4B plan, everything else left at its default:

```bash
uv run as15 plan -p "..." -L lyrics.txt -d 200 --bpm 72 --key "A minor" -o out/song.codes 2>plan.log
tr '\r' '\n' < plan.log | grep -viE 'planning:|Fetching'   # does the caption match the prompt?
uv run as15 sing --audio-codes out/song.codes -p "..." -L lyrics.txt -d 200 --bpm 72 --key "A minor" -o out/final.flac
```

The middle line is not optional -- see §6. Reading the plan's caption costs
nothing and catches a prompt the planner misread before the render pays for it.

Then vary one thing at a time from that same plan -- the seed, or `-g` between 5
and 7 -- and keep the take that sounds best. Nothing in this repo can tell you
which that is; it is a listening decision.

Output must be `.flac`; the path is checked before the run, not after 10 minutes
of diffusion. An existing file is overwritten and you are warned first.

Every take carries its own recipe in Vorbis comments -- the prompt, the lyrics,
`AS15_SEED`, `AS15_MODEL`, `AS15_CHECKPOINT`, steps, guidance, shift, sampler,
DCW, any metas that were set, and the plan it was rendered from
(`describe()` in `src/as15/pipeline.py`). None
of it is a clock or a machine ID, so the same command twice gives a
byte-identical file. Read a take back with:

```bash
ffprobe -v error -show_entries format_tags -of default out/final.flac
```

The style prompt comes back as `comment`: ffmpeg maps the Vorbis `DESCRIPTION`
field onto its own key, so asking for `format_tags=DESCRIPTION` returns nothing.
Every `AS15_*` field keeps its name, so a single value pulls out cleanly -- which
is how you recover the seed of a take worth re-rendering:

```bash
ffprobe -v error -show_entries format_tags=AS15_SEED -of csv=p=0 out/final.flac
```

Pre-fetch weights before a first run so the download is not mistaken for a hang:

```bash
uv run as15 download -m xl-sft
```

---

## 8. When it comes back wrong

Find the symptom, then work the right-hand column in the order it is written and
stop when it is fixed. Almost none of these are sampler problems, and reaching
for `-s` first is the most reliable way to spend fifteen minutes and learn
nothing.

| Symptom | What to change, in order |
| --- | --- |
| **Lines skipped, a verse missing, the bridge raced through** | Redo the §4 arithmetic -- a crammed container is far and away the commonest cause. Then shorten lines, even out syllables within a section, simplify the form, strip stacked tag modifiers, and draw four fresh seeds. Not more steps: skipping is reported at every step count there is, so no number of them guarantees delivery. |
| **An instrument nobody asked for** | Read the plan's reasoning caption (§6) -- it usually named it first. Then cut the contradiction in `-p` that invited it and state a short positive palette. There is nowhere to put a negative: as15 has no negative prompt, CFG's unconditional branch is a stored null embedding rather than text you supply, so "no guitars" is just the word "guitars" in your caption. Re-plan, or drop `--plan` and see if it was the LM's idea. |
| **The chorus is no bigger than the verse** | Put the arc in `-p` and not only in the tags (§1, rule 7): the model will happily render one texture for four minutes. Shorten the hook so it has room to open up, make the verse genuinely sparse in the writing, and expect to finish the contrast with a fader (§9). |
| **Harsh, brittle, hissy, everything at one level** | Expected on a turbo draft. The measured draft below pinned its peak at 0.999 with 8.9 dB of range; sft from the same plan came back at 0.877 and 15.5 dB. Render on sft before treating it as a prompt problem. If it survives sft, audition more seeds, then fix it in the mix. |
| **The planner changed the voice, the genre or the language** | §6, and it is the reason that section exists. Read the block, say the thing twice in different words, re-plan at a new seed. If it keeps drifting, drop `--plan`: the direct path cannot contradict you, having no opinion to contradict you with. |
| **Vocals in something meant to be instrumental** | Empty lyrics is the reliable form; a tag-only sheet is not. Strip every vocal word from `-p` as well -- "female vocals", "topline", "choir", "sung" -- then draw more seeds. |
| **Wrong language, or an accent you did not ask for** | `--language` has to match the words, and matching it is necessary rather than sufficient: upstream names multilingual lyric compliance as a standing limitation and English as the safe case. Non-English takes need more seeds and more listening; `zh_rap` is called out on the model card as its own weak spot. |
| **The run refuses to start** | Read the message and believe it. Prompt over 256 tokens, lyrics over 2048, `--time-signature` outside `(2, 3, 4, 6)`, `--bpm` outside 30--300, output not `.flac`, or a plan with fewer codes than the duration needs. All of it is checked before a single weight loads, which is why a typo costs a second. |

---

## 9. After the render

### What as15 will not do

The model has more tasks than this CLI exposes. There is **no remix, cover,
repaint, extend or reference-audio path here.** as15 implements the text-to-audio
half only, and the audio-to-codes tokenizer the editing tasks need is never even
built (`src/as15/conditioning.py`). Advice found elsewhere about Remix strength,
Repaint windows or a reference track is describing upstream's Gradio and ComfyUI
surfaces, not this one, and there is no flag here to map it onto.

What there is instead: re-render the same plan at another seed, another
guidance, or the other checkpoint. That gives you variations of one song rather
than a transformation of a recording -- and for a section that will not come out
right, it is regenerate-and-comp rather than repair in place.

### Finish it outside the generator

The complaints that recur about ACE-Step output are vocal expressiveness,
dynamic contrast, drum transitions, groove and how memorable the melody is.
Those are the hardest things to generate and none of them is a step count. as15
writes 48 kHz stereo FLAC, which is a fine thing to import:

- comp the best sections together -- takes rendered from one plan share an
  arrangement, so they are the ones most likely to line up, but check both
  boundaries by ear;
- automate the verse-to-chorus level change the model under-plays;
- reinforce or replace the drums, particularly transitions and fills;
- de-ess, tame the harshness, clear the low end;
- master last, once the arrangement has stopped moving.

### Originality

MIT covers this code and upstream's weights. It says nothing about anyone else's
songs. Upstream's own guidance is to check output for originality, disclose that
a track is AI-generated, and get permission for protected material used as a
source or a reference -- which in practice means keeping artist names out of `-p`
(§1) and existing lyrics out of the sheet.

---

## Worked example

"Hold the Morning" -- the canonical run: 4B plan, `xl-sft`, everything else at
its default. Every number below was measured on it.

**The spec** (§0), settled before anything else: a night nobody wants to end;
defiance turning into euphoria by the last drop; contemporary soulful deep house;
one warm female lead; Rhodes, sub bass, filtered pads, shuffled hats over a
four-on-the-floor kick; sparse verse rising through a build into a wide drop,
three times; the section order set out under **Duration** below.

**Genre first.** The brief was only "something the model generates well", so §1's
genre-fit table picked it: soulful deep house. Four-on-the-floor gives a 200 s
take a grid to hold, the hook repeats so the lyric budget goes into one line, and
the vocal sits under reverb -- which is where coarse vocal synthesis stops
mattering. It also avoids the named weak spot, guitar-forward rock.

**Prompt**

```
soulful deep house, warm female lead vocal, soulful female topline,
four-on-the-floor kick, deep round sub bass, dusty Rhodes chords,
filtered analog pads, crisp shuffled hi-hats, spacious plate reverb,
late-night club warmth
```

Thirty words over genre, voice, five instruments, texture and production. The
voice is named twice on purpose -- see the re-plan below.

**Metas** -- `--bpm 122` (house is tempo-defined), `--key "D minor"`,
`--time-signature 4`, `--language en`.

**Duration** -- intro 14 + verse 16 + build 16 + drop 24 + breakdown 12 +
verse 16 + build 16 + drop 24 + bridge 14 + build 10 + drop 24 + outro 14 = ~200,
so `-d 200`.

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

Lines run 6--8 syllables and stay parallel across the two verses. One metaphor --
refusing the dawn -- from three angles. `[build]` into `[chorus - drop]` three
times is the shape the genre already wants; the empty `[intro]` and `[breakdown]`
are deliberate instrumental sections; `(not tonight)` is a backing vocal and the
capitalised last chorus lifts it. The first two choruses are word-for-word
identical, which is what makes one listen enough to say whether the model can
land the same hook twice; only the third departs from them.

**Plan, and check it**

```bash
uv run as15 plan -p "$PROMPT" -L out/hold-the-morning.txt -d 200 \
  --bpm 122 --key "D minor" --time-signature 4 --seed 1122 \
  -o out/hold-the-morning.codes 2>plan.log
tr '\r' '\n' < plan.log | grep -viE 'planning:|Fetching'
```

The first plan's caption described "a smooth male singer using expressive
falsetto" against a prompt that said female vocals. The prompt gained a second,
differently-worded statement of the voice (`warm female lead vocal, soulful female
topline`) and the plan was rewritten at a new seed; the second came back with "a
powerful female lead vocal ... layered backing vocals ... dynamic builds leading
into instrumental drops". 2 min 36 s each time. **Do not skip this check** -- it
is the cheapest step in the run and it guards the most expensive one.

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

### What you can check without listening

Listening is still the decision. But a few cheap measurements catch a dead run
before it wastes an audition, and they showed the sft take was the better one on
every axis:

| | draft (turbo) | final (sft) |
| --- | --- | --- |
| Dynamic range, 4 s RMS windows | 8.9 dB | **15.5 dB** |
| Stereo width (side/mid RMS) | 0.27 | **0.83** |
| Sample peak | 0.999, pinned to the ceiling | 0.877, with headroom |

Worth checking on any take: that it is the length you asked for, that the peak is
not pinned at 1.0, that the RMS envelope moves rather than sitting flat, and --
if the sheet ends in `[fade out]` -- that the tail actually decays. This one went
from -14.8 dB to -61.7 dB over its last four seconds.

```bash
uv run --with numpy --with soundfile python - <<'EOF'
import numpy as np, soundfile as sf
x, sr = sf.read("out/hold-the-morning.flac")
m, w = x.mean(axis=1), 4 * 48000
rms = np.array([np.sqrt((m[i:i+w]**2).mean()) for i in range(0, len(m)-w+1, w)])
db = 20 * np.log10(np.maximum(rms, 1e-6))
print(f"{len(m)/sr:.1f}s  peak={np.abs(x).max():.3f}  dyn={db.max()-db.min():.1f}dB")
print("".join(" .:-=+*#%@"[min(9, int((d-db.min())/(db.max()-db.min())*9.99))] for d in db))
EOF
```

None of that says whether it is any good. It says whether it is worth your ears.
