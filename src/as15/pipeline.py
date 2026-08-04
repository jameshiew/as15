"""Text-to-music generation: conditioning -> MLX diffusion -> MLX VAE decode."""

from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np

from . import __version__
from .atomic import publish
from .convert import convert_dit, convert_vae, resolve_precision
from .flac import set_comments
from .mlx.sampler import check_guidance, check_sampling_options
from .models import (
    BASE_REPO,
    BASE_REVISION,
    LATENT_CHANNELS,
    SAMPLE_RATE,
    VAE_HOP,
    ModelSpec,
    Snapshot,
    check_vae_geometry,
    ensure_snapshot,
    load_dit_config,
)

# Only the files we actually use; skips the 3.7 GB 5Hz LM and the 2B turbo DiT
# that also live in the shared base repo.
BASE_PATTERNS = ["vae/*", "Qwen3-Embedding-0.6B/*", "config.json"]

# VAE decode allocates in proportion to output length: a 60 s decode peaks
# around 16 GB, and past ~90 s a single tensor exceeds Metal's maximum buffer
# size outright. Decode in windows instead. 512 frames is ~20 s of audio.
DECODE_CHUNK_FRAMES = 512
# Overlap discarded from each interior edge, so windows never see a truncated
# receptive field. Matches the upstream tiled-decode overlap.
DECODE_OVERLAP_FRAMES = 64


def tiled_decode(
    vae,
    latents: mx.array,
    chunk_frames: int = DECODE_CHUNK_FRAMES,
    overlap: int = DECODE_OVERLAP_FRAMES,
) -> mx.array:
    """Decode latents in overlapping windows, keeping only each window's core.

    The Oobleck decoder is fully convolutional and maps one latent frame to
    exactly ``VAE_HOP`` samples, so decoding a padded window and trimming the
    padding back off reproduces the un-tiled result wherever the overlap
    covers the receptive field.
    """
    total = latents.shape[1]
    if total <= chunk_frames:
        return vae.decode(latents)

    parts: list[mx.array] = []
    for start in range(0, total, chunk_frames):
        end = min(start + chunk_frames, total)
        lo = max(0, start - overlap)
        hi = min(total, end + overlap)

        audio = vae.decode(latents[:, lo:hi, :])
        mx.eval(audio)

        head = (start - lo) * VAE_HOP
        tail = (hi - end) * VAE_HOP
        parts.append(
            audio[:, head : audio.shape[1] - tail, :] if tail else audio[:, head:, :]
        )
        mx.eval(parts[-1])
        del audio
        mx.clear_cache()

    return mx.concatenate(parts, axis=1)


@dataclass
class GenerationRequest:
    style_prompt: str
    lyrics: str
    duration: float = 120.0
    language: str = "en"
    bpm: int | str | None = None
    key_scale: str | None = None
    time_signature: str | int | None = None
    steps: int | None = None
    guidance: float | None = None
    # None means "use the checkpoint's recommended default" (see ModelSpec).
    shift: float | None = None
    seed: int | None = None
    sampler: str = "euler"
    infer_method: str = "ode"
    dcw: bool | None = None
    precision: str = "bf16"


@dataclass
class GenerationResult:
    audio: np.ndarray  # [samples, channels], float32
    sample_rate: int
    seed: int | None
    timings: dict[str, float | str] = field(default_factory=dict)
    # Vorbis comments describing the run, for write_audio to embed.
    tags: dict[str, str] = field(default_factory=dict)


# Duration bounds, shared by the CLI flag and by callers that build a request
# directly. The lower bound is roughly where a generation stops being a song;
# the upper is what a 32 GB machine can decode in one pass.
MIN_DURATION = 10.0
MAX_DURATION = 600.0

# The time signatures the metas block was trained with.
VALID_TIME_SIGNATURES = (2, 3, 4, 6)

# ``mx.random.key`` takes a uint64: outside this range it raises TypeError out
# of the binding, minutes into a run, with no mention of the seed.
MAX_SEED = 2**64 - 1


@dataclass(frozen=True)
class Settings:
    """What a request actually runs with, once model defaults are filled in."""

    steps: int
    guidance: float
    shift: float
    dcw: bool
    compute_dtype: str


def _numeric(value: int | str | float) -> float | None:
    """The number *value* denotes, or None if it is free text."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return float(value)


def _check_bpm(bpm: int | str | None) -> None:
    if bpm is None:
        return
    if isinstance(bpm, str) and not bpm.strip():
        raise ValueError("bpm must not be blank; leave it unset instead.")
    value = _numeric(bpm)
    # `bpm or 'N/A'` in the metas block renders 0 as *unset*, and a negative or
    # non-finite tempo reaches the text encoder verbatim. Neither is a tempo.
    if value is not None and not value > 0:
        raise ValueError(f"bpm must be greater than zero, got {bpm!r}.")


def _check_time_signature(time_signature: str | int | None) -> None:
    if time_signature is None:
        return
    value = _numeric(time_signature)
    if value not in VALID_TIME_SIGNATURES:
        allowed = ", ".join(str(t) for t in VALID_TIME_SIGNATURES)
        raise ValueError(
            f"time_signature must be one of {allowed}, got {time_signature!r}."
        )


def resolve_settings(spec: ModelSpec, request: GenerationRequest) -> Settings:
    """Fill in the checkpoint's defaults, rejecting a request that cannot run.

    Both the pipeline and the CLI go through here -- the pipeline before it
    fetches ~10 GB of weights, the CLI before it prints what it is about to do
    -- so what the banner reports is what the loop runs, and a request that
    would otherwise be quietly reinterpreted fails in under a second.

    Every bound below exists because the value is used somewhere that cannot
    tell a mistake from a setting.

    Raises:
        ValueError: naming the field, for the CLI to turn into a usage error.
    """
    steps = request.steps if request.steps is not None else spec.steps
    shift = request.shift if request.shift is not None else spec.shift
    dcw = request.dcw if request.dcw is not None else spec.dcw
    guidance = request.guidance if request.guidance is not None else spec.guidance

    check_sampling_options(request.sampler, request.infer_method, shift, steps)
    compute_dtype = resolve_precision(request.precision)

    # Written as a chained comparison rather than `d < MIN or d > MAX`, which
    # NaN passes: click's `min=`/`max=` are that second form, so `--duration
    # nan` parses and then reaches `int(duration)` in the metas block.
    if not MIN_DURATION <= request.duration <= MAX_DURATION:
        raise ValueError(
            f"duration must be between {MIN_DURATION:g} and {MAX_DURATION:g} "
            f"seconds, got {request.duration}."
        )

    # Shared with the diffusion loop, which enforces the same bound on its own
    # argument -- a caller reaching it directly gets the same answer.
    check_guidance(guidance)
    if not spec.supports_cfg:
        # Distilled checkpoints are trained to run without a null branch;
        # forcing CFG on them doubles cost and degrades output.
        guidance = 1.0

    if request.seed is not None and not 0 <= request.seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}, got {request.seed}.")

    if not request.language.strip():
        raise ValueError("language must be a code such as 'en', not blank.")

    _check_bpm(request.bpm)
    _check_time_signature(request.time_signature)

    return Settings(
        steps=steps,
        guidance=guidance,
        shift=shift,
        dcw=dcw,
        compute_dtype=compute_dtype,
    )


def describe(
    spec: ModelSpec, request: GenerationRequest, settings: Settings
) -> dict[str, str]:
    """The Vorbis comments a generation leaves in its own output.

    Everything a take cannot be recovered from the audio: the words that were
    sung, the prompt they were sung to, and the recipe for generating it again.
    The alternative is remembering which shell history line produced which
    file, which lasts until the next reboot.

    Read off the *resolved* settings for the same reason the banner is -- what
    is recorded has to be what ran, not what was asked for. A turbo take says
    ``guidance 1`` because CFG was dropped, and a request that left steps unset
    records the checkpoint's number rather than nothing.

    Nothing here is a clock or a machine ID: the tags are a function of the
    request alone, so two runs of the same command still produce byte-identical
    files and a regeneration can be diffed against the take it replaces.
    """
    tags = {"DESCRIPTION": request.style_prompt}
    # An instrumental has no lyric sheet, and an empty LYRICS field is not the
    # same claim as an absent one -- players render the first as a blank pane.
    if request.lyrics.strip():
        tags["LYRICS"] = request.lyrics

    tags["AS15_VERSION"] = __version__
    tags["AS15_MODEL"] = spec.key
    # The commit, not the repo ID: upstream force-pushes weights under the same
    # ID, so the ID alone does not say which bytes sang this.
    tags["AS15_CHECKPOINT"] = f"{spec.repo_id}@{spec.revision}"
    # Absent for an unseeded draw. The CLI always picks a seed, but a caller
    # building the request itself may leave it None, and a number here would
    # promise a reproducibility the file does not have.
    if request.seed is not None:
        tags["AS15_SEED"] = str(request.seed)

    tags["AS15_DURATION"] = f"{request.duration:g}"
    tags["AS15_LANGUAGE"] = request.language
    tags["AS15_STEPS"] = str(settings.steps)
    tags["AS15_GUIDANCE"] = f"{settings.guidance:g}"
    tags["AS15_SHIFT"] = f"{settings.shift:g}"
    tags["AS15_SAMPLER"] = request.sampler
    tags["AS15_INFER_METHOD"] = request.infer_method
    tags["AS15_DCW"] = "on" if settings.dcw else "off"
    # The name the flag takes rather than the dtype it resolves to, so every
    # AS15_ tag is something that can be typed straight back at the CLI.
    tags["AS15_PRECISION"] = request.precision

    # The conditioning metas, which are only in the file if they were set: the
    # model is told "N/A" for an unset one, and recording that as a value would
    # make an unset tempo indistinguishable from one deliberately given.
    for name, value in (
        ("AS15_BPM", request.bpm),
        ("AS15_KEY", request.key_scale),
        ("AS15_TIME_SIGNATURE", request.time_signature),
    ):
        if value is not None:
            tags[name] = str(value)
    return tags


def _resolve_snapshots(spec: ModelSpec) -> tuple[Snapshot, Snapshot]:
    dit_snapshot = ensure_snapshot(spec.repo_id, spec.revision)
    base_snapshot = ensure_snapshot(
        BASE_REPO, BASE_REVISION, allow_patterns=BASE_PATTERNS
    )
    return dit_snapshot, base_snapshot


def _load_vae(base_snapshot: Snapshot):
    from .mlx.vae import MLXOobleckVAE

    path = convert_vae(base_snapshot)
    cfg = json.loads((base_snapshot.path / "vae" / "config.json").read_text())
    check_vae_geometry(cfg)
    vae = MLXOobleckVAE(
        downsampling_ratios=cfg["downsampling_ratios"],
        channel_multiples=cfg["channel_multiples"],
        decoder_channels=cfg["decoder_channels"],
        decoder_input_channels=cfg["decoder_input_channels"],
        audio_channels=cfg["audio_channels"],
    )
    vae.load_weights(str(path))
    mx.eval(vae.parameters())
    return vae


def _load_dit(dit_snapshot: Snapshot, precision: str):
    from .mlx.dit import MLXDiTDecoder

    path = convert_dit(dit_snapshot, precision)
    config = load_dit_config(dit_snapshot.path)
    # Every key in the cache is one the model has: load_weights is strict, and
    # the pop that used to make that true hid whatever else the converter had
    # put there. Conditioning reads the null embedding from the checkpoint.
    weights = mx.load(str(path))
    dit = MLXDiTDecoder.from_config(config)
    dit.load_weights(list(weights.items()))
    mx.eval(dit.parameters())
    return dit


def generate(
    spec: ModelSpec,
    request: GenerationRequest,
    device: str = "auto",
    progress: bool = True,
) -> GenerationResult:
    from .conditioning import Conditioner
    from .mlx.sampler import mlx_generate_diffusion

    # Reject a malformed request here rather than after the snapshots have been
    # fetched and the conditioner has run, which is minutes in.
    settings = resolve_settings(spec, request)
    timings: dict[str, float | str] = {}

    # The peak counter is process-global and never decays, so a second
    # generate() in the same process would otherwise report whichever earlier
    # call -- or the weight conversion -- happened to allocate the most.
    mx.reset_peak_memory()

    # --- Conditioning (PyTorch), released before the DiT is loaded ----------
    t0 = time.time()
    dit_snapshot, base_snapshot = _resolve_snapshots(spec)
    timings["resolve"] = time.time() - t0

    t0 = time.time()
    conditioner = Conditioner(dit_snapshot.path, base_snapshot.path, device=device)
    timings["load_conditioner"] = time.time() - t0

    # Every stage from here on hands its memory back on the way out, failure
    # included. Only the successful path used to, which is enough for the CLI
    # -- the process exits either way -- but leaves a caller that catches the
    # failure and retries holding the whole dead attempt, so the retry dies of
    # an out-of-memory naming some entirely different stage.
    t0 = time.time()
    with conditioner:
        cond = conditioner.build(
            style_prompt=request.style_prompt,
            lyrics=request.lyrics,
            duration=request.duration,
            language=request.language,
            bpm=request.bpm,
            key_scale=request.key_scale,
            time_signature=request.time_signature,
        )
        timings["condition"] = time.time() - t0
    del conditioner

    # --- Diffusion (MLX) ---------------------------------------------------
    dit = None
    try:
        t0 = time.time()
        dit = _load_dit(dit_snapshot, request.precision)
        timings["load_dit"] = time.time() - t0

        result = mlx_generate_diffusion(
            mlx_decoder=dit,
            encoder_hidden_states_np=cond.encoder_hidden_states,
            context_latents_np=cond.context_latents,
            src_latents_shape=(1, cond.latent_frames, LATENT_CHANNELS),
            seed=request.seed,
            infer_method=request.infer_method,
            shift=settings.shift,
            infer_steps=settings.steps,
            guidance_scale=settings.guidance,
            null_condition_emb_np=(
                cond.null_condition_emb if settings.guidance > 1.0 else None
            ),
            sampler_mode=request.sampler,
            dcw_enabled=settings.dcw,
            compute_dtype=settings.compute_dtype,
            disable_tqdm=not progress,
        )
        timings.update(result["time_costs"])
        latents = result["target_latents"]
    finally:
        # Same ordering argument as Conditioner.release(): collect first so
        # that every dead array has returned its buffer to MLX's cache, then
        # clear the cache so the buffers go back to the OS.
        del dit
        gc.collect()
        mx.clear_cache()

    # --- Decode (MLX) ------------------------------------------------------
    vae = audio = None
    try:
        t0 = time.time()
        vae = _load_vae(base_snapshot)
        audio = tiled_decode(vae, mx.array(latents).astype(mx.float32))
        mx.eval(audio)
        timings["decode"] = time.time() - t0

        audio_np = np.array(audio[0])  # [samples, channels]
    finally:
        del vae, audio
        gc.collect()
        mx.clear_cache()

    timings["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    return GenerationResult(
        audio=audio_np,
        sample_rate=SAMPLE_RATE,
        seed=request.seed,
        timings=timings,
        tags=describe(spec, request, settings),
    )


# The one container we write. soundfile takes the format from the extension
# and will write a couple of dozen of them, but the rest were never a choice
# so much as whatever libsndfile happened to accept: lossy ones throw away
# what the VAE just spent 8 GB decoding, and the obscure lossless ones (SD2,
# W64, PVF) answer no question anyone has. FLAC is lossless, half the size of
# WAV, and reads everywhere.
OUTPUT_SUFFIX = ".flac"
OUTPUT_SUBTYPE = "PCM_16"


def check_output_path(path: Path) -> None:
    """Fail now if the finished audio could not be written to *path*.

    Everything here is a property of the path alone, so it can be answered
    before a note is generated. It used to surface from :func:`write_audio`,
    which runs after the weights have downloaded, after conditioning and
    after fifty diffusion steps -- a mistyped extension or an output
    directory that does not exist cost the whole run.

    Nothing is created: the ``mkdir`` stays in :func:`write_audio`, so a
    preflight that passes has not left a directory behind for a generation
    that then fails. Permission is checked on the nearest existing ancestor,
    which is the one ``mkdir`` needs it on.

    Raises:
        ValueError: naming what about the path cannot work.
    """
    if path.suffix.lower() != OUTPUT_SUFFIX:
        raise ValueError(
            f"the output format comes from the extension, and {OUTPUT_SUFFIX} "
            f"is the only one written; {path.name!r} asks for "
            f"{path.suffix or 'none'}."
        )
    if path.is_dir():
        raise ValueError(f"{path} is a directory, not the file to write.")

    existing = next(p for p in (path.parent, *path.parent.parents) if p.exists())
    if not existing.is_dir():
        raise ValueError(f"{existing} is a file, so {path.parent} cannot be created.")
    # The write lands on a temporary in this directory and is renamed over the
    # destination, so what has to be writable is the directory. An existing
    # read-only file at *path* is replaced fine.
    if not os.access(existing, os.W_OK):
        raise ValueError(f"{existing} is not writable.")


def write_audio(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    tags: Mapping[str, str] | None = None,
) -> None:
    """Write audio, peak-limiting only if the decode overshot full scale.

    *tags* are Vorbis comments -- :func:`describe` builds the ones a generation
    carries -- installed after the encoder has closed, because libsndfile can
    only set ten fixed fields and lyrics are not one of them. See
    :mod:`as15.flac`.

    The file appears whole or not at all: soundfile encodes into a temporary
    alongside it, which is renamed over *path* once the encoder has closed.
    A write that ran out of disk, or was interrupted, used to leave a
    truncated file where the song should be -- having already destroyed the
    take that was there before. Tagging happens on that same temporary, so a
    rewrite that fails part way through is discarded with it.

    :func:`check_output_path` runs first, so what the CLI preflights and what
    the write accepts are the same rules rather than two copies of them.

    Raises:
        ValueError: if *path* cannot be written, or the audio is not finite.
    """
    import soundfile as sf

    check_output_path(path)

    # Checked before the peak-limiting below, which launders both: a NaN fails
    # `peak > 0.999` and is written through untouched, and an infinity makes
    # the scale factor 0.999/inf == 0, turning every sample into 0 or NaN.
    # Either way the result is a file of silence or noise, written as if the
    # generation had worked.
    if not np.all(np.isfinite(audio)):
        raise ValueError("the decode produced samples that are NaN or infinite.")

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.999:
        # Leave a shade of headroom so 16-bit conversion cannot wrap.
        audio = audio * (0.999 / peak)

    def encode(tmp: Path) -> None:
        sf.write(str(tmp), audio, sample_rate, subtype=OUTPUT_SUBTYPE)
        if tags:
            set_comments(tmp, tags)

    path.parent.mkdir(parents=True, exist_ok=True)
    publish(path, encode)
