"""Text-to-music generation: conditioning -> MLX diffusion -> MLX VAE decode."""

from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np

from . import __version__
from .atomic import publish
from .codes import check_codes, codes_for_frames, format_codes
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
    latent_frames_for,
    load_dit_config,
    resolve_planner,
    round_half_up,
    seconds_for,
)

# Only the files conditioning uses; skips the 2B turbo DiT and the 3.4 GB
# 1.7B planner that also live in the shared base repo. A run that plans with
# the 1.7B fetches it separately, in planner_path, with its own pattern.
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
    # A 5 Hz audio-code plan to condition on, or None for the direct path.
    # ``planner`` names a checkpoint to *write* one with (see PLANNERS); the
    # pipeline runs it, fills ``audio_codes`` in and resolves again, so what
    # generates is always a request whose plan is already settled.
    audio_codes: Sequence[int] | None = None
    planner: str | None = None
    planner_seed: int | None = None
    # Which planner wrote ``audio_codes``, when one did. Provenance, not a
    # setting: the pipeline fills it in after planning, and nothing reads it
    # back except the tags. A take is reproducible from the plan alone, but
    # the plan does not say what wrote it.
    planned_by: str | None = None


@dataclass
class GenerationResult:
    audio: np.ndarray  # [samples, channels], float32
    sample_rate: int
    seed: int | None
    timings: dict[str, float] = field(default_factory=dict)
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
class ResolvedGenerationRequest:
    """A request with every default filled in and every derived value settled.

    A :class:`GenerationRequest` is what someone asked for, and several of its
    fields do not answer a question any stage can act on: ``steps=None`` means
    "the checkpoint's", and a duration is a real number where the runtime has a
    40 ms frame grid. Each stage used to reach into the request and answer those
    for itself, so they could answer differently -- ``duration=12.9`` told the
    model 12 seconds, sized the latent window for 12.92, printed 12.9 in the
    banner and recorded 12.9 in the file, and the one that mattered for lyric
    pacing was the one that was wrong.

    Resolution happens once, here, and every stage downstream reads this: the
    banner, the conditioner, the diffusion loop and :func:`describe`. What is
    reported is what ran because there is nothing else left to report.
    """

    # What to generate.
    style_prompt: str
    lyrics: str
    language: str
    bpm: int | str | None
    key_scale: str | None
    time_signature: str | int | None

    # How long. Three numbers because the request's duration is not any of them:
    # *latent_frames* is what the DiT generates and the VAE decodes, *duration*
    # is what those frames are worth in seconds and therefore what the file will
    # be, and *metas_duration* is whole seconds because that is the format the
    # ``# Metas`` block was trained in.
    latent_frames: int
    duration: float
    metas_duration: int

    # How to sample.
    seed: int | None
    steps: int
    guidance: float
    shift: float
    sampler: str
    infer_method: str
    dcw: bool
    precision: str
    compute_dtype: str

    # What to condition the context block on. ``audio_codes`` is a settled plan
    # -- checked here against the frame count, so the conditioner is never
    # handed one too short for the song -- and ``planner`` is set only while a
    # plan is still to be written.
    audio_codes: tuple[int, ...] | None
    planner: str | None
    planner_seed: int | None
    planned_by: str | None


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


def resolve_request(
    spec: ModelSpec, request: GenerationRequest
) -> ResolvedGenerationRequest:
    """Settle a request into the one form every stage runs from.

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

    # The duration collapses to a frame count here and is never interpreted
    # again: the latent grid is 40 ms, so the request's real number names a
    # take that cannot be generated, and rounding it separately in the metas
    # block, the context window and the tags is how they came to disagree.
    frames = latent_frames_for(request.duration)
    duration = seconds_for(frames)

    if request.planner is not None:
        resolve_planner(request.planner)
    if request.planner is not None and request.audio_codes is not None:
        raise ValueError(
            "a run either conditions on the plan it is given or writes one, "
            "not both; drop either the audio codes or the planner."
        )
    codes = None
    if request.audio_codes is not None:
        codes = tuple(request.audio_codes)
        # Against the frame count rather than the requested duration: a plan
        # covers whole 200 ms windows, and it is the frames that have to be
        # covered. This runs for a plan the planner just wrote too, so an LM
        # that stopped early is caught here rather than conditioning the last
        # verse on silence.
        check_codes(codes, frames)
        # Cropped to what the song uses, for the same reason every other value
        # here is resolved: the conditioner only ever reads the frames it is
        # generating, so a longer plan would be recorded in the file as the
        # recipe while a prefix of it was what actually ran.
        codes = codes[: codes_for_frames(frames)]

    return ResolvedGenerationRequest(
        style_prompt=request.style_prompt,
        lyrics=request.lyrics,
        language=request.language,
        bpm=request.bpm,
        key_scale=request.key_scale,
        time_signature=request.time_signature,
        latent_frames=frames,
        duration=duration,
        # Of the length being generated, not of the length asked for: a 12.9 s
        # request is a 12.92 s take, and the model is told 13 rather than the
        # 12 that flooring the request produced.
        metas_duration=round_half_up(duration),
        seed=request.seed,
        steps=steps,
        guidance=guidance,
        shift=shift,
        sampler=request.sampler,
        infer_method=request.infer_method,
        dcw=dcw,
        precision=request.precision,
        compute_dtype=compute_dtype,
        audio_codes=codes,
        planner=request.planner,
        planner_seed=request.planner_seed,
        planned_by=request.planned_by,
    )


def describe(spec: ModelSpec, request: ResolvedGenerationRequest) -> dict[str, str]:
    """The Vorbis comments a generation leaves in its own output.

    Everything a take cannot be recovered from the audio: the words that were
    sung, the prompt they were sung to, and the recipe for generating it again.
    The alternative is remembering which shell history line produced which
    file, which lasts until the next reboot.

    Read off the *resolved* request for the same reason the banner is -- what
    is recorded has to be what ran, not what was asked for. A turbo take says
    ``guidance 1`` because CFG was dropped, a request that left steps unset
    records the checkpoint's number rather than nothing, and the duration is the
    length of the audio in the file rather than the number typed at the CLI.
    Every one of them still resolves back to itself, so a tag can be handed
    straight back to ``as15 sing`` and generate the same take.

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

    # The take's own length, which is a whole number of 40 ms frames. Recording
    # the request's duration instead described a file that was never written:
    # two of them resolve to the same audio, and neither is what the audio is.
    tags["AS15_DURATION"] = f"{request.duration:g}"
    tags["AS15_LANGUAGE"] = request.language
    tags["AS15_STEPS"] = str(request.steps)
    tags["AS15_GUIDANCE"] = f"{request.guidance:g}"
    tags["AS15_SHIFT"] = f"{request.shift:g}"
    tags["AS15_SAMPLER"] = request.sampler
    tags["AS15_INFER_METHOD"] = request.infer_method
    tags["AS15_DCW"] = "on" if request.dcw else "off"
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

    # The plan, in full, in the form `--audio-codes` reads. It is the single
    # largest thing separating this take from a text-only one and cannot be
    # recovered from the audio, so recording a digest would leave the file
    # describing a recipe it does not carry -- and a planned take could never
    # be re-rendered at other settings. A two-minute plan is 600 codes, about
    # 11 KB of comment against a ~20 MB master.
    if request.audio_codes is not None:
        tags["AS15_AUDIO_CODES"] = format_codes(request.audio_codes)
        tags["AS15_AUDIO_CODE_COUNT"] = str(len(request.audio_codes))
    # Only for a plan this run wrote. A plan that arrived in a file was written
    # by something this take cannot vouch for, and naming a planner it did not
    # run would be a worse claim than saying nothing.
    if request.planned_by is not None:
        tags["AS15_PLANNER"] = request.planned_by
        if request.planner_seed is not None:
            tags["AS15_PLANNER_SEED"] = str(request.planner_seed)
    return tags


def _resolve_snapshots(spec: ModelSpec) -> tuple[Snapshot, Snapshot]:
    dit_snapshot = ensure_snapshot(spec.repo_id, spec.revision)
    base_snapshot = ensure_snapshot(
        BASE_REPO, BASE_REVISION, allow_patterns=BASE_PATTERNS
    )
    return dit_snapshot, base_snapshot


def planner_path(spec) -> Path:
    """Download the planner *spec* names and return the directory it lives in.

    Only the planner's own files are fetched. The 1.7B is published as a
    directory of the shared base repo rather than a repo of its own, so asking
    for it without a pattern would pull the VAE, the text encoder and a second
    DiT along with it.
    """
    patterns = [f"{spec.subdir}/*"] if spec.subdir else None
    snapshot = ensure_snapshot(spec.repo_id, spec.revision, allow_patterns=patterns)
    return snapshot.path / spec.subdir if spec.subdir else snapshot.path


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
    # fetched and the conditioner has run, which is minutes in. Nothing below
    # reads *request* again: every stage is handed the resolved one, so none of
    # them can settle a default or a duration differently from the others.
    resolved = resolve_request(spec, request)
    timings: dict[str, float] = {}

    # The peak counter is process-global and never decays, so a second
    # generate() in the same process would otherwise report whichever earlier
    # call -- or the weight conversion -- happened to allocate the most.
    mx.reset_peak_memory()

    # --- Planning (MLX), released before anything else loads ---------------
    #
    # First because it is the largest single stage after the DiT -- the 4B
    # planner is 8.4 GB -- and every later stage needs its plan. Resolving
    # again with the plan filled in rather than patching it into the resolved
    # request keeps resolution a pure function of the request, and puts the
    # LM's own output through the same length check a supplied plan gets.
    if resolved.planner is not None:
        from .planner import write_plan

        t0 = time.time()
        plan = write_plan(
            planner_path(resolve_planner(resolved.planner)),
            resolved,
            resolved.latent_frames,
            resolved.planner_seed,
            progress=progress,
        )
        timings["plan"] = time.time() - t0
        request = replace(
            request, audio_codes=plan.codes, planner=None, planned_by=plan.planner
        )
        resolved = resolve_request(spec, request)

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
        cond = conditioner.build(resolved)
        timings["condition"] = time.time() - t0
    del conditioner

    # --- Diffusion (MLX) ---------------------------------------------------
    dit = None
    try:
        t0 = time.time()
        dit = _load_dit(dit_snapshot, resolved.precision)
        timings["load_dit"] = time.time() - t0

        result = mlx_generate_diffusion(
            mlx_decoder=dit,
            encoder_hidden_states_np=cond.encoder_hidden_states,
            context_latents_np=cond.context_latents,
            src_latents_shape=(1, resolved.latent_frames, LATENT_CHANNELS),
            seed=resolved.seed,
            infer_method=resolved.infer_method,
            shift=resolved.shift,
            infer_steps=resolved.steps,
            guidance_scale=resolved.guidance,
            null_condition_emb_np=(
                cond.null_condition_emb if resolved.guidance > 1.0 else None
            ),
            sampler_mode=resolved.sampler,
            dcw_enabled=resolved.dcw,
            compute_dtype=resolved.compute_dtype,
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
        seed=resolved.seed,
        timings=timings,
        tags=describe(spec, resolved),
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
