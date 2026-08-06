# Sampling flags, and why almost all of them stay at their default

as15 resolves every one of these from the checkpoint before the run starts, and
the defaults are the checkpoint's own. Read this before overriding one; the
short version is that `-g` is the only knob with a good reason to move, and only
on `xl-sft`.

## `-g / --guidance` (sft only)

7.0 is the default. Lower (3--5) follows the prompt more loosely and often sounds
more natural; higher is more literal and can get brittle. `1.0` disables CFG
entirely and halves the cost, because the unconditional branch stops being
evaluated.

Upstream's own documentation contradicts itself over which checkpoints take CFG
at all -- the model zoo says sft and base, the Gradio guide describes the field
as base-only. Here the checkpoint decides: `xl-turbo` is distilled, declares no
CFG support, and any `-g` you pass is replaced by 1.0 before the run. A recipe
from elsewhere quoting a guidance number for turbo is describing a surface that
is not this one.

Varying `-g` between 5 and 7 against a frozen plan is a genuine A/B: same
arrangement, same seed, two readings of it.

## `-s / --steps`

Leave alone unless you are deliberately trading quality for time.

Below the checkpoint default, quality drops off quickly -- and this is the one
people get backwards. `xl-sft` defaults to **50**, so `-s 30` is rendering at
60% of the count the checkpoint expects, not raising it. `xl-turbo` defaults to
**8**, the count it was distilled for.

Past the default is mostly folklore. Turbo at 20 or 50 steps circulates as a fix
for skipped words, but it is unreplicated single-user reporting, it costs the
15x that made turbo worth using in the first place, and skipping is reported at
every step count there is. Skipped words are a duration problem far more often
than a sampler one -- see `troubleshooting.md`.

## `--sampler`

`euler` (default) or `heun`. Heun evaluates the model twice per step and
averages the predictions for a second-order step -- roughly double the diffusion
time for higher per-step accuracy. That trade is most worthwhile on 8-step
turbo, where the step count is small enough that accuracy per step matters, and
least worthwhile on 50-step sft, which already subdivides finely.

Heun is ODE-only: the corrector has no SDE step to correct, so that pairing is
rejected rather than run.

## `--dcw` / `--no-dcw`

**Leave alone.** as15 already follows the checkpoint -- off for `xl-sft`, on for
`xl-turbo`.

Wavelet-domain correction was tuned for the distilled models. Left on for a
non-distilled checkpoint it makes output mushy and distorted; this is upstream's
"garbled audio on Apple Silicon" report, and fixing it is one of the reasons
this repo exists. The override is there for experiments, not for tuning a take.

## `--precision`

`bf16`. `fp32` doubles memory for no measurable gain -- the MLX port was
verified at 0.999378 output correlation against the reference PyTorch DiT, the
residual being bf16 rounding, and the two are spectrally indistinguishable.

## Choosing between takes

Once the prompt is settled, the seed is the whole difference between two takes,
and which one is good is a listening decision. Score each take 0--2 on seven
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

Do not let the total decide on its own: a take that scored zero on Hook is not a
candidate however spotless the rest of it is. Polishing a strong composition is
ordinary work; rescuing a clean forgettable one is starting again with extra
steps. Audio is also the axis most likely to improve on its own when you move
from a turbo draft to the sft render.

`${CLAUDE_SKILL_DIR}/scripts/check-take.py out/take-*.flac` measures length, peak, dynamic range and
stereo width across a whole batch and draws each take's RMS envelope, which
narrows the field before you spend ears on it. A flat envelope means the chorus
never opened; a take that is not the length you asked for has already failed.

### Coming back to a batch you did not just generate

A stale directory of takes and no memory of what was asked for is the common
case, and it needs the pieces in a particular order:

1. **Recover the brief from the files, not from your notes.** Every take
   carries the prompt, the sheet and the whole recipe in its own tags -- see
   `after-the-render.md`. Notes written alongside a batch go stale; the tags
   cannot, because the run wrote them.
2. **Measure the batch** with `check-take.py`, which triages it without an
   opinion about which is good.
3. **Then score what survives** on the seven axes above, against the recovered
   prompt rather than against what you now wish you had asked for.

Takes rendered from one frozen plan differ only by seed, and their tags say so
-- identical `AS15_AUDIO_CODES` across the batch is the proof that you are
comparing performances of one arrangement rather than several songs.
