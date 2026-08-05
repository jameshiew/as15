---
name: as15-songwriting
description: Write and generate songs with as15 (ACE-Step 1.5 XL on MLX). Covers style prompts, lyric sheets with structure tags, choosing duration/BPM/key/time signature, picking a checkpoint, and the draft-then-render loop. Use when the user wants to create, write, plan or generate a song in this repository.
allowed-tools: Read, Write, Bash
---

# Songwriting for as15

`as15` generates exactly what you hand it. Nothing writes a style prompt for you,
nothing invents lyrics, and nothing guesses a tempo or a key -- **you are the
planner.** A meta you leave unset is not inferred: it reaches the conditioning as
the literal string `N/A` and the model improvises around it.

So the job is to produce four things, then run one command:

| Output | Flag | Notes |
| --- | --- | --- |
| Style prompt | `-p` | Genre, instruments, voice, texture, production |
| Lyric sheet | `-L` (file) or `-l` (string) | Structure tags + words; omit for an instrumental |
| Musical metas | `--bpm`, `--key`, `--time-signature`, `--language` | Unset means `N/A`, not "guess" |
| Run settings | `-d`, `-m`, `--seed`, `-o` | Duration is a hard container for the lyrics |

```bash
uv run as15 sing -p "STYLE PROMPT" -L lyrics.txt -d 180 --bpm 96 --key "A minor" -o out/song.flac
```

Write lyric sheets to a file and pass `-L`. Multi-line lyrics through `-l` fight
the shell over quoting and newlines for no benefit. `out/` is gitignored, which
is where takes belong.

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
5. Keep the prompt and the lyric tags consistent. If the prompt says piano ballad,
   a `[guitar solo]` tag is a fight the output loses.

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
- **CAPITALS read as intensity.** `HOLD THE LINE` is shouted; `hold the line` is
  not. Use it on a hook, not a verse.
- **Parentheses read as backing vocals**: `We rise again (again)`.
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

---

## 3. Metas (`--bpm`, `--key`, `--time-signature`, `--language`)

Nothing infers these. Unset renders as `- bpm: N/A` in the conditioning, which is
a legitimate choice -- the model picks -- but if the song depends on a tempo or a
feel, say so.

| Flag | Accepts | Notes |
| --- | --- | --- |
| `--bpm` | any number > 0 | slow 60--80, mid 90--120, fast 130--180 |
| `--key` | free text | `"C major"`, `"A minor"`, `"F# minor"`. Common keys behave best |
| `--time-signature` | **`2`, `3`, `4` or `6`** | A bare integer -- `3` for a waltz, **not** `"3/4"`. Anything else is rejected |
| `--language` | code, default `en` | Written into the `# Languages` header; must match the lyrics |

`--time-signature` is the one that catches people: it takes the numerator on its
own rather than a `4/4`-style string, and the valid set is fixed at
`(2, 3, 4, 6)` -- the signatures the metas block was trained with. Anything else
is rejected before the run starts.

Set `--bpm` when you need to match other material, when the genre is tempo-defined
(house ~124, boom bap ~90, drum and bass ~174), or when the lyric density needs a
specific pace to fit. Set `--key` when you want a specific colour or a fixed
vocal range. Otherwise leaving them off is fine.

---

## 4. Duration (`-d`)

10 to 600 seconds, default 120. **This is a container, not a hint** -- the model
has to fit the whole lyric sheet into it. Too short and the delivery is crammed
and verses go missing; too long and you get instrumental drift at the end.

Rough section costs at 90--120 BPM in 4/4:

| Section | Seconds |
| --- | --- |
| Intro / outro | 5--15 each |
| Verse (8 lines) | 25--35 |
| Pre-chorus (4 lines) | 12--18 |
| Chorus (4--8 lines) | 20--30 |
| Bridge | 15--25 |
| Instrumental / solo | 10--20 |

| Shape | Duration |
| --- | --- |
| Verse--chorus--verse--chorus | 110--150 |
| Plus pre-choruses and a bridge | 170--220 |
| Plus intro, outro and a solo | 210--280 |

Sanity check before generating: **sung lines x ~4 s, plus the instrumental
sections.** If that exceeds `-d`, cut lines or raise the duration. Slower tempos
need more seconds for the same words; faster ones fit more but still need air.
When in doubt, go longer -- a song with room to breathe beats one that rushes.

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
| Relative speed | 1x | ~6x |
| Use for | the take you keep | drafting and iteration |

Scaling the README's M5 timings, a 120 s take is roughly **10 minutes** of
diffusion on `xl-sft` and roughly **40 seconds** on `xl-turbo`. That ratio is the
whole workflow: iterate on turbo, render on sft.

- `-g / --guidance` (sft only): 7.0 is the default. Lower (3--5) follows the
  prompt more loosely and often sounds more natural; higher is more literal and
  can get brittle. `1.0` disables CFG and halves the cost.
- `-s / --steps`: leave alone unless you are deliberately trading quality for
  time. Below the checkpoint default, quality drops off quickly.
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

The planner also **fills in metas you left unset** -- it settles a bpm, a key and
a time signature and writes them into its reasoning block, which is printed on
stderr. Anything you set with `--bpm` / `--key` / `--time-signature` overrides
it, and the duration is always yours. So `--plan` is a way to get a considered
tempo and key rather than `N/A`, without having to pick them yourself.

Plans are worth keeping. Planning is one LM pass; rendering is fifty DiT passes,
so write the plan once and render it several ways:

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
the diffusion is contributing on top of the sketch.

Every planned take stores its plan in `AS15_AUDIO_CODES`, so a take can be
re-rendered from the file it produced.

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
4. **Explore takes** by dropping `--seed` once the prompt is right. The chosen
   seed is printed on stderr and stored in the file, so a good take is never lost.
5. **Render on sft** with the seed and settings that worked:

   ```bash
   uv run as15 sing --seed 42 -p "..." -L lyrics.txt -d 180 -o out/final.flac
   ```

   A turbo seed does not reproduce the same take on sft -- different checkpoint,
   different schedule. Expect to audition a few.

Output must be `.flac`; the path is checked before the run, not after 10 minutes
of diffusion. An existing file is overwritten and you are warned first.

Every take carries its own recipe in Vorbis comments -- the prompt, the lyrics,
`AS15_SEED`, `AS15_MODEL`, `AS15_CHECKPOINT`, steps, guidance, shift, sampler,
DCW, and any metas that were set (`describe()` in `src/as15/pipeline.py`). None
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

## Worked example

Idea: "a late-night drive song about leaving a city, dream pop, female vocals".

**Prompt**

```
dream pop, female vocals, shimmering reverb guitars, warm analog tape,
patient mid-tempo drums, wide stereo synth pads, melancholic but not heavy
```

**Metas** -- `--bpm 96`, `--key "A minor"`, no `--time-signature` (4 is implied
by the genre and leaving it `N/A` costs nothing), `--language en`.

**Duration** -- intro 8 + verse 30 + chorus 25 + verse 30 + chorus 25 + bridge 20
+ chorus 25 + outro 12 = ~175, so `-d 180`.

**Lyrics** (`lyrics.txt`)

```
[intro]

[verse]
Headlights count the empty lanes
Radio is mostly rain
Every mile I used to know
Turns to somewhere I don't go

[chorus]
Let the city keep the light
I am driving out tonight
Every bridge behind me burns
Soft and slow (soft and slow)

[verse]
Took the long way past your street
Didn't slow, didn't keep
Half a life in one back seat
Half a song I can't complete

[chorus]
Let the city keep the light
I am driving out tonight
Every bridge behind me burns
Soft and slow (soft and slow)

[bridge - building energy]
And the dark is not a threat
It is only what comes next

[chorus - anthemic]
LET THE CITY KEEP THE LIGHT
I AM DRIVING OUT TONIGHT
Every bridge behind me burns
Soft and slow (soft and slow)

[outro]
[fade out]
```

Lines run 7--8 syllables and stay parallel between the two verses. One metaphor --
the drive out -- worked from three angles. The `(soft and slow)` echo is a backing
vocal, the capitalised final chorus lifts it, and every tag is something the
prompt already supports.

**Draft, then render**

```bash
uv run as15 sing -m xl-turbo --seed 7 -p "dream pop, female vocals, shimmering reverb guitars, warm analog tape, patient mid-tempo drums, wide stereo synth pads, melancholic but not heavy" -L lyrics.txt -d 180 --bpm 96 --key "A minor" -o out/draft.flac
```

```bash
uv run as15 sing --seed 7 -p "dream pop, female vocals, shimmering reverb guitars, warm analog tape, patient mid-tempo drums, wide stereo synth pads, melancholic but not heavy" -L lyrics.txt -d 180 --bpm 96 --key "A minor" -o out/final.flac
```
