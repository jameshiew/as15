"""Text-to-music generation: conditioning -> MLX diffusion -> MLX VAE decode."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from .convert import NULL_COND_KEY, convert_dit, convert_vae
from .models import (
    BASE_REPO,
    LATENT_CHANNELS,
    LATENT_FPS,
    SAMPLE_RATE,
    VAE_HOP,
    ModelSpec,
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
        parts.append(audio[:, head : audio.shape[1] - tail, :] if tail else audio[:, head:, :])
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
    timings: dict = field(default_factory=dict)


def _resolve_snapshots(spec: ModelSpec) -> tuple[Path, Path]:
    dit_snapshot = ensure_snapshot(spec.repo_id)
    base_snapshot = ensure_snapshot(BASE_REPO, allow_patterns=BASE_PATTERNS)
    return dit_snapshot, base_snapshot


def _load_vae(base_snapshot: Path):
    from .mlx.vae import MLXAutoEncoderOobleck

    path = convert_vae(base_snapshot / "vae")
    cfg = json.loads((base_snapshot / "vae" / "config.json").read_text())
    vae = MLXAutoEncoderOobleck(
        encoder_hidden_size=cfg["encoder_hidden_size"],
        downsampling_ratios=cfg["downsampling_ratios"],
        channel_multiples=cfg["channel_multiples"],
        decoder_channels=cfg["decoder_channels"],
        decoder_input_channels=cfg["decoder_input_channels"],
        audio_channels=cfg["audio_channels"],
    )
    vae.load_weights(str(path))
    mx.eval(vae.parameters())
    return vae


def _load_dit(dit_snapshot: Path, spec: ModelSpec, precision: str):
    from .mlx.dit import MLXDiTDecoder

    path = convert_dit(dit_snapshot, spec.cache_name, precision)
    config = load_dit_config(dit_snapshot)
    weights = mx.load(str(path))
    weights.pop(NULL_COND_KEY, None)
    dit = MLXDiTDecoder.from_config(config)
    dit.load_weights(list(weights.items()))
    mx.eval(dit.parameters())
    dit.materialize_static_buffers()
    return dit


def generate(
    spec: ModelSpec,
    request: GenerationRequest,
    device: str = "auto",
    progress: bool = True,
) -> GenerationResult:
    from .conditioning import Conditioner
    from .mlx.sampler import mlx_generate_diffusion

    timings: dict[str, float] = {}
    steps = request.steps if request.steps is not None else spec.steps
    guidance = request.guidance if request.guidance is not None else spec.guidance
    shift = request.shift if request.shift is not None else spec.shift
    dcw = request.dcw if request.dcw is not None else spec.dcw
    if not spec.supports_cfg and guidance > 1.0:
        # Distilled checkpoints are trained to run without a null branch;
        # forcing CFG on them doubles cost and degrades output.
        guidance = 1.0

    # --- Conditioning (PyTorch), released before the DiT is loaded ----------
    t0 = time.time()
    dit_snapshot, base_snapshot = _resolve_snapshots(spec)
    timings["resolve"] = time.time() - t0

    t0 = time.time()
    conditioner = Conditioner(dit_snapshot, base_snapshot, device=device)
    timings["load_conditioner"] = time.time() - t0

    t0 = time.time()
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

    conditioner.release()
    del conditioner
    gc.collect()

    # --- Diffusion (MLX) ---------------------------------------------------
    t0 = time.time()
    dit = _load_dit(dit_snapshot, spec, request.precision)
    timings["load_dit"] = time.time() - t0

    compute_dtype = "bfloat16" if request.precision == "bf16" else "float32"
    result = mlx_generate_diffusion(
        mlx_decoder=dit,
        encoder_hidden_states_np=cond.encoder_hidden_states,
        context_latents_np=cond.context_latents,
        src_latents_shape=(1, cond.latent_frames, LATENT_CHANNELS),
        seed=request.seed,
        infer_method=request.infer_method,
        shift=shift,
        infer_steps=steps,
        guidance_scale=guidance,
        null_condition_emb_np=cond.null_condition_emb if guidance > 1.0 else None,
        sampler_mode=request.sampler,
        dcw_enabled=dcw,
        compute_dtype=compute_dtype,
        disable_tqdm=not progress,
    )
    timings.update(result["time_costs"])
    latents = result["target_latents"]

    del dit
    mx.clear_cache()
    gc.collect()

    # --- Decode (MLX) ------------------------------------------------------
    t0 = time.time()
    vae = _load_vae(base_snapshot)
    audio = tiled_decode(vae, mx.array(latents).astype(mx.float32))
    mx.eval(audio)
    timings["decode"] = time.time() - t0

    audio_np = np.array(audio[0])  # [samples, channels]
    del vae, audio
    mx.clear_cache()

    timings["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    return GenerationResult(
        audio=audio_np,
        sample_rate=SAMPLE_RATE,
        seed=request.seed,
        timings=timings,
    )


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write audio, peak-limiting only if the decode overshot full scale."""
    import soundfile as sf

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.999:
        # Leave a shade of headroom so 16-bit conversion cannot wrap.
        audio = audio * (0.999 / peak)

    path.parent.mkdir(parents=True, exist_ok=True)
    subtype = "PCM_16" if path.suffix.lower() in {".wav", ".flac"} else None
    sf.write(str(path), audio, sample_rate, subtype=subtype)


def latent_frames_for(duration: float) -> int:
    return max(1, int(round(duration * LATENT_FPS)))
