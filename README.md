# as15

Generate songs with [ACE-Step 1.5 XL](https://huggingface.co/collections/ACE-Step/ace-step-15)
on Apple Silicon. The 4B diffusion transformer and the audio VAE both run
natively on **MLX**; you give it a style prompt and lyrics and it writes a wav.

Built for a 32 GB M-series Mac. A 30 s clip peaks around 9.6 GB.

```bash
uv run as15 sing \
  -p "dream pop, female vocals, shimmering reverb guitars, warm analog tape" \
  -L lyrics.txt \
  -d 120 -o song.wav
```

## Why not just use the upstream repo

The official [ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) repo works,
but it is ~92k lines of Python and pulls in gradio, lightning, tensorboard, peft,
numba, torchcodec, torchao, nano-vllm, modelscope and more. This is a single-purpose
CLI: **9 runtime dependencies**, no web UI, no training code, no server.

It also fixes three things that bite on Apple Silicon:

- **DCW is off by default for `xl-sft`.** Wavelet-domain correction was tuned for
  the distilled turbo models. Left on for the non-distilled checkpoints it makes
  output mushy and distorted — this is
  [issue #1259](https://github.com/ace-step/ACE-Step-1.5/issues/1259), the "garbled
  audio on Apple Silicon" report, and upstream's CLI still cannot turn it off at all.
  Here it follows the checkpoint (off + `shift 1.0` for sft/base, on + `shift 3.0`
  for turbo) and is overridable with `--dcw` / `--no-dcw`.
- **VAE decode is chunked.** Upstream's MLX path decodes the whole track in one
  go; decode memory grows linearly with duration and past ~90 s a single tensor
  exceeds Metal's 20.1 GB maximum buffer size and the run dies. `as15` decodes in
  overlapping windows — bit-identical output, flat ~7.3 GB regardless of length,
  so full-length songs work.
- **No 5Hz LM.** The LM is a planner that invents lyrics and captions from a short
  idea. If you are supplying both, it is 3.7 GB of weights doing nothing.

## Install

Needs macOS on Apple Silicon and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Weights download from Hugging Face on first use (~19 GB per DiT checkpoint, plus
~1.5 GB shared VAE and text encoder) into the usual `~/.cache/huggingface`.

Fetch and pre-convert ahead of time:

```bash
uv run as15 download -m xl-sft
```

## Usage

```bash
uv run as15 sing --help
uv run as15 models
```

Lyrics come from `--lyrics` / `-l`, `--lyrics-file` / `-L`, or `-L -` for stdin.
Use section tags the model understands:

```
[verse]
City lights are fading slow
Neon rivers start to flow

[chorus]
Hold me in the afterglow
```

Useful options:

| Flag | Default | Notes |
| --- | --- | --- |
| `-m, --model` | `xl-sft` | `xl-sft` (best) or `xl-turbo` (~6x faster) |
| `-d, --duration` | `120` | Seconds, 10–600 |
| `-s, --steps` | model default | 50 for sft, 8 for turbo |
| `-g, --guidance` | `7.0` | CFG scale; ignored by turbo, which is distilled |
| `--seed` | random | Reuse to reproduce a take |
| `--bpm`, `--key`, `--time-signature` | unset | Written into the conditioning metadata |
| `--dcw / --no-dcw` | per model | See above; leave alone unless experimenting |
| `--precision` | `bf16` | `fp32` doubles memory for no measurable gain |

## How it runs

| Stage | Runtime | Params | When |
| --- | --- | --- | --- |
| Qwen3 text encoder + condition encoder | PyTorch (MPS) | 0.6B + 0.6B | once |
| DiT decoder | **MLX** | 4.17B | every step |
| Oobleck VAE decode | **MLX** | 0.06B | once, chunked |

The conditioning stage runs the checkpoint's own `trust_remote_code` modules, so it
is the reference implementation rather than a re-derivation, and it is released
before the DiT loads to keep peak memory down. Everything that runs per-step is MLX.

DiT weights are published as fp32 (~20 GB); they are converted once to bf16 MLX
safetensors (~8.3 GB) in `~/.cache/as15`. The port was verified against the
reference PyTorch DiT: **0.999378** output correlation on identical inputs, the
residual being bf16 rounding. bf16 and fp32 generations are spectrally
indistinguishable, so bf16 is the default.

## Timings

M5, 32 GB, 30 s of audio:

| Model | Steps | Diffusion | Decode | Peak |
| --- | --- | --- | --- | --- |
| `xl-turbo` | 8 | ~10 s | ~6 s | 9.7 GB |
| `xl-sft` | 50 | ~158 s | ~8 s | 9.6 GB |

Decode is chunked, so peak memory is flat in duration; diffusion time scales with it.

## Tests

```bash
uv run pytest -q
```

These pin the conditioning and weight-layout invariants that produce
correct-looking-but-wrong audio when broken.

## Licence

Code here is MIT. The ACE-Step 1.5 checkpoints are MIT and, per the model cards,
trained on licensed, royalty-free/public-domain and synthetic data with commercial
use permitted — check the model card yourself before shipping anything.

The MLX DiT, VAE, DCW and sampler modules under `src/as15/mlx/` are vendored from
ACE-Step 1.5 (MIT) and adapted: repo coupling removed, the `pytorch_wavelets`
bridge dropped for pure-MLX Haar, and a compute-dtype option added.
