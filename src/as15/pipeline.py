"""Text-to-music generation: conditioning -> MLX diffusion -> MLX VAE decode."""

from __future__ import annotations

import gc
import json
import math
import os
import time
from collections.abc import Iterator, Mapping, Sequence
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

# What the metas block will be read as a tempo. Nothing here is a hard musical
# limit -- 30 is a very slow largo and 300 is past where anyone still counts
# beats -- but outside them a number is not a tempo somebody meant: `--bpm 1`
# and `--bpm 10000` are a typo and a sample rate, and both used to be written
# into the conditioning verbatim for the text encoder to make what it could of.
MIN_BPM = 30
MAX_BPM = 300

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

    The musical metas are narrower here than on the request for the same reason:
    a tempo may be *said* as ``120`` or ``"120"`` because a request can come from
    a config file as easily as from the CLI, but by this point it is one integer,
    and every stage that writes it down -- the metas block, the planner's
    reasoning, the file's own tags -- writes the same characters.
    """

    # What to generate. Stripped, and blank where a value would be meaningless:
    # an unset key and a blank one condition identically, so they are the same
    # value here rather than two that ``describe`` then reports differently.
    style_prompt: str
    lyrics: str
    language: str
    bpm: int | None
    key_scale: str | None
    time_signature: int | None

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


def _whole(value: int | str | float) -> int | None:
    """The whole number *value* denotes, or None if it denotes none.

    Spelling is not the point: ``4``, ``"4"`` and ``" 4.0 "`` all denote four,
    and all three reach here because a request can be built from a config file
    or a form as easily as from the CLI. The *value* is. ``"4/4"``, ``"quickly"``
    and ``inf`` denote nothing the metas block can carry, and each of them used
    to be written into it verbatim -- ``float("inf") > 0`` is true, and a
    non-numeric string skipped the numeric check entirely.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    # Not an int and not parseable as a float: nothing that denotes a number.
    if not isinstance(value, float):
        return None
    # A tempo of 128.5 is not one the metas block has a spelling for, and
    # ``float('nan').is_integer()`` is False, so both fall out here.
    return int(value) if math.isfinite(value) and value.is_integer() else None


def _resolve_bpm(bpm: int | str | None) -> int | None:
    """The tempo *bpm* names, as the integer the metas block is written with.

    Raises:
        ValueError: if it names no tempo, or one outside :data:`MIN_BPM` to
            :data:`MAX_BPM`. Both used to pass: ``bpm or 'N/A'`` renders 0 as
            *unset*, and everything else -- a negative tempo, an infinity, the
            word "quickly" -- went into the conditioning as itself.
    """
    if bpm is None:
        return None
    value = _whole(bpm)
    if value is None:
        raise ValueError(
            f"bpm must be a whole number of beats per minute, got {bpm!r}. "
            f"Leave it unset for N/A."
        )
    if not MIN_BPM <= value <= MAX_BPM:
        raise ValueError(f"bpm must be between {MIN_BPM} and {MAX_BPM}, got {value}.")
    return value


def check_seed(seed: int | None) -> None:
    """Reject a seed the diffusion loop cannot draw from.

    ``mx.random.key`` takes a uint64: outside that range it raises TypeError out
    of the binding, minutes into a run, with no mention of the seed. Shared by
    :func:`resolve_request` and by :meth:`GenerationSession.takes`, which checks
    every seed in a batch before the first one loads anything.
    """
    if seed is not None and not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}, got {seed}.")


def _resolve_time_signature(time_signature: str | int | None) -> int | None:
    """The time signature *time_signature* names, as its numerator.

    Raises:
        ValueError: if the metas block was not trained on it. ``"4"`` used to
            pass the check as the number four and then be written into the block
            as the string it arrived as, which is the same characters; ``"4.0"``
            passed the same way and was written as ``4.0``.
    """
    if time_signature is None:
        return None
    value = _whole(time_signature)
    if value not in VALID_TIME_SIGNATURES:
        allowed = ", ".join(str(t) for t in VALID_TIME_SIGNATURES)
        raise ValueError(
            f"time_signature must be one of {allowed} -- the numerator on its "
            f"own, so 3 for a waltz and 6 for 6/8 -- got {time_signature!r}."
        )
    return value


def _one_line(what: str, value: str) -> str:
    """*value* stripped, rejected if it would not fit the line it is written on.

    Both fields this guards are written into a line of a block whose shape is
    part of the trained prompt format: ``- keyscale: {}`` in the metas block, and
    the ``# Languages`` header above the lyrics. A newline in either does not
    make a longer value, it makes a different block -- the rest of the metas end
    up inside the key, or the lyric sheet gains a header the model reads as one.
    """
    if "\n" in value or "\r" in value:
        raise ValueError(f"{what} must be a single line, got {value!r}.")
    return value.strip()


def _resolve_key_scale(key_scale: str | None) -> str | None:
    """The key *key_scale* names, or None where it names none.

    Free text otherwise: the block takes whatever it is given, and which
    spellings the model knows is not something this can rule on.

    Raises:
        ValueError: if it would break the metas block it is written into.
    """
    if key_scale is None:
        return None
    # Blank is unset rather than a value. The metas block already read it that
    # way -- ``key_scale or 'N/A'`` -- while ``describe`` wrote an empty AS15_KEY
    # beside it, so the file claimed a key that was never conditioned on.
    return _one_line("key_scale", key_scale) or None


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

    check_seed(request.seed)

    # The musical metas are settled here rather than checked here: a bpm of
    # ``"120"`` and one of ``120`` are the same tempo, and leaving them
    # different meant the metas block, the planner's reasoning and the file's
    # tags each stringified whatever they were handed. Free text is stripped
    # for the same reason -- ``" en "`` passed the blank check and went into
    # the trained ``# Languages`` header with its spaces still on.
    #
    # The lyrics are the one field left exactly as given: their line breaks are
    # the structure the model reads, not whitespace around a value.
    style_prompt = request.style_prompt.strip()
    if not style_prompt:
        raise ValueError(
            "style_prompt must say what to generate; it is the whole of the "
            "musical direction, and blank asks for nothing in particular."
        )

    # Lowercased because the header was trained in lowercase and because `EN`
    # and `en` should not be two different takes of the same song.
    language = _one_line("language", request.language).lower()
    if not language:
        raise ValueError("language must be a code such as 'en', not blank.")

    bpm = _resolve_bpm(request.bpm)
    key_scale = _resolve_key_scale(request.key_scale)
    time_signature = _resolve_time_signature(request.time_signature)

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
        style_prompt=style_prompt,
        lyrics=request.lyrics,
        language=language,
        bpm=bpm,
        key_scale=key_scale,
        time_signature=time_signature,
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
    # make an unset tempo indistinguishable from one deliberately given. A blank
    # one is already None by the time it gets here, so `--key ""` no longer
    # writes an AS15_KEY that names no key.
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


class GenerationSession:
    """One song, several takes, with each model loaded exactly once.

    A seed is the whole difference between two takes of the same song. The
    prompt, the lyrics, the metas and the plan are identical; what moves is the
    noise the diffusion starts from. Finding a song is therefore mostly a matter
    of hearing a few of them and keeping one -- and running ``as15 sing`` that
    many times is the wrong way to do it. Each run conditions the same words
    again under torch, loads the same 8.3 GB of DiT weights again and loads the
    VAE again, redoing about a minute of setup whose inputs did not move.

    A session does each of those once, in the order a single generation was
    already careful about, held across the whole batch rather than one take::

        plan            -> release the planner
        condition       -> release torch
        load the DiT    -> every seed  -> release the DiT
        load the VAE    -> every take  -> release the VAE

    so the peak is still whichever single stage is largest, whether the batch is
    one take or ten. What waits in between is the latents, at 3.8 MB each for a
    ten-minute song -- three orders of magnitude under the models this avoids
    reloading, which is what makes the phases orderable this way at all.

    Takes are yielded as they decode rather than returned together, so a caller
    writing them out holds one take's audio at a time. Ten minutes of stereo
    float32 is 230 MB, and a batch of them would be a serious fraction of the
    memory the rest of the run is arranged around.
    """

    def __init__(
        self,
        spec: ModelSpec,
        request: GenerationRequest,
        device: str = "auto",
        progress: bool = True,
    ):
        # Resolved and thrown away: what it is for is the rejection. A request
        # that cannot be honoured fails here, before the session has fetched a
        # byte, rather than after the snapshots and the conditioner. Nothing
        # below reads *request* again either -- every stage is handed a resolved
        # one, so none of them can settle a default or a duration differently.
        resolve_request(spec, request)

        self.spec = spec
        self.device = device
        self.progress = progress
        # What the batch as a whole cost: the stages that run once however many
        # takes there are. Diffusion and decode are per take and ride on the
        # takes themselves.
        self.timings: dict[str, float] = {}

        self._request = request
        self._snapshots: tuple[Snapshot, Snapshot] | None = None
        self._conditioning = None

    def takes(self, seeds: Sequence[int | None]) -> Iterator[GenerationResult]:
        """Generate one take per seed, yielding each as it finishes decoding.

        The seeds are the only thing that differs between them; everything else
        is the request the session was built with.

        Raises:
            ValueError: if a seed is outside the range the diffusion loop can
                draw from. Every seed is checked before the first one runs, so a
                batch cannot render three takes and then find the fourth
                unusable.
        """
        seeds = list(seeds)
        for seed in seeds:
            check_seed(seed)
        if not seeds:
            return

        # Process-global and never decaying, so it is reset per batch: a second
        # batch would otherwise report whichever earlier one -- or the weight
        # conversion -- happened to allocate the most.
        mx.reset_peak_memory()

        self._plan()
        requests = [
            resolve_request(self.spec, replace(self._request, seed=seed))
            for seed in seeds
        ]
        # One conditioning for the batch, built from the first take because they
        # are the same song: the seed reaches the diffusion loop and nothing
        # else. Passing a resolved request rather than its fields is what keeps
        # the window conditioned on and the window generated the same length.
        cond = self._condition(requests[0])
        latents = self._diffuse(requests, cond)
        yield from self._decode(requests, latents)

    def __enter__(self) -> GenerationSession:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Drop the conditioning the takes were generated against.

        The models are released by the phases that load them, success or
        failure; what a session goes on holding afterwards is the conditioning
        -- around 8 MB for a ten-minute song, and stale the moment anything
        about the request would change.
        """
        self._conditioning = None

    # --- Planning (MLX), released before anything else loads ---------------

    def _plan(self) -> None:
        """Write the audio-code plan, once for the batch, if one was asked for.

        First because it is the largest single stage after the DiT -- the 4B
        planner is 8.4 GB -- and every later stage needs its plan. Once for the
        batch because a plan is a property of the song rather than of a take:
        holding it fixed while the seed moves is what makes a batch a comparison
        of renders rather than of arrangements.

        The plan goes back into the *request* and is resolved again, rather than
        being patched into the resolved one, so resolution stays a pure function
        of the request and the LM's own output goes through the same length
        check a supplied plan gets.
        """
        resolved = self._resolve()
        if resolved.planner is None:
            return
        from .planner import write_plan

        t0 = time.time()
        plan = write_plan(
            planner_path(resolve_planner(resolved.planner)),
            resolved,
            resolved.latent_frames,
            resolved.planner_seed,
            progress=self.progress,
        )
        self.timings["plan"] = time.time() - t0
        self._request = replace(
            self._request,
            audio_codes=plan.codes,
            planner=None,
            planned_by=plan.planner,
        )

    def _resolve(self) -> ResolvedGenerationRequest:
        return resolve_request(self.spec, self._request)

    def _snapshot_pair(self) -> tuple[Snapshot, Snapshot]:
        if self._snapshots is None:
            t0 = time.time()
            self._snapshots = _resolve_snapshots(self.spec)
            self.timings["resolve"] = time.time() - t0
        return self._snapshots

    # --- Conditioning (PyTorch), released before the DiT is loaded ---------

    def _condition(self, resolved: ResolvedGenerationRequest):
        from .conditioning import Conditioner

        if self._conditioning is not None:
            return self._conditioning
        dit_snapshot, base_snapshot = self._snapshot_pair()

        t0 = time.time()
        conditioner = Conditioner(
            dit_snapshot.path, base_snapshot.path, device=self.device
        )
        self.timings["load_conditioner"] = time.time() - t0

        # Every stage from here on hands its memory back on the way out, failure
        # included. Only the successful path used to, which is enough for the
        # CLI -- the process exits either way -- but leaves a caller that
        # catches the failure and retries holding the whole dead attempt, so the
        # retry dies of an out-of-memory naming some entirely different stage.
        t0 = time.time()
        with conditioner:
            self._conditioning = conditioner.build(resolved)
            self.timings["condition"] = time.time() - t0
        return self._conditioning

    # --- Diffusion (MLX) ---------------------------------------------------

    def _diffuse(
        self, requests: Sequence[ResolvedGenerationRequest], cond
    ) -> list[tuple[np.ndarray, dict[str, float]]]:
        """Run every take's diffusion loop against one loaded DiT.

        All of them before any decoding, because the DiT and the VAE must not be
        resident together: interleaving would either reload one of them per take
        or hold both, and holding both is what the whole ordering exists to
        avoid.
        """
        from .mlx.sampler import mlx_generate_diffusion

        dit = None
        drawn: list[tuple[np.ndarray, dict[str, float]]] = []
        try:
            t0 = time.time()
            dit = _load_dit(self._snapshot_pair()[0], requests[0].precision)
            self.timings["load_dit"] = time.time() - t0

            for resolved in requests:
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
                    disable_tqdm=not self.progress,
                )
                drawn.append((result["target_latents"], dict(result["time_costs"])))
        finally:
            # Same ordering argument as Conditioner.release(): collect first so
            # that every dead array has returned its buffer to MLX's cache, then
            # clear the cache so the buffers go back to the OS.
            del dit
            gc.collect()
            mx.clear_cache()
        return drawn

    # --- Decode (MLX) ------------------------------------------------------

    def _decode(
        self,
        requests: Sequence[ResolvedGenerationRequest],
        drawn: Sequence[tuple[np.ndarray, dict[str, float]]],
    ) -> Iterator[GenerationResult]:
        vae = audio = None
        try:
            t0 = time.time()
            vae = _load_vae(self._snapshot_pair()[1])
            self.timings["load_vae"] = time.time() - t0

            for resolved, (latents, timings) in zip(requests, drawn, strict=True):
                t0 = time.time()
                audio = tiled_decode(vae, mx.array(latents).astype(mx.float32))
                mx.eval(audio)
                audio_np = np.array(audio[0])  # [samples, channels]
                # Dropped before the take is handed over, not on the next pass
                # round the loop: the caller writes a file in between, and a
                # whole decode's buffers should not sit there while it does.
                audio = None

                timings["decode"] = time.time() - t0
                # The batch's peak so far rather than this take's: the counter
                # is a high-water mark over the process, and by design every
                # take passes through the same largest stage.
                timings["peak_memory_gb"] = mx.get_peak_memory() / 1e9

                yield GenerationResult(
                    audio=audio_np,
                    sample_rate=SAMPLE_RATE,
                    seed=resolved.seed,
                    timings=timings,
                    tags=describe(self.spec, resolved),
                )
        finally:
            del vae, audio
            gc.collect()
            mx.clear_cache()


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


def take_paths(out: Path, seeds: Sequence[int | None]) -> list[Path]:
    """Where a batch of takes is written, given the ``--out`` that was asked for.

    A single take goes to *out* itself: one take is an ordinary generation and
    should not have a name invented for it. More than one and each name carries
    both its place in the batch and the seed that drew it -- the place because
    that is the order they were generated in and the order a listing should sort
    them into, the seed because it is what regenerates the one you keep. Both
    are in the file's own tags too; a filename is what you have while choosing
    between them.
    """
    if len(seeds) == 1:
        return [out]
    # Wide enough that the whole batch sorts lexically, which is how a file
    # listing will order them.
    width = max(2, len(str(len(seeds))))
    return [
        out.with_name(f"{out.stem}-{index:0{width}d}-seed-{seed}{out.suffix}")
        for index, seed in enumerate(seeds, start=1)
    ]


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
