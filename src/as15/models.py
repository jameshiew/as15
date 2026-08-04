"""Model registry, checkpoint resolution and cache paths."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Shared assets: VAE + Qwen3 text encoder (+ the 5Hz LM, which we do not use).
BASE_REPO = "ACE-Step/Ace-Step1.5"
# Every repo is pinned to a commit. Upstream force-pushes weights under the same
# repo ID, and a swapped checkpoint is silent -- it still generates audio, just
# worse -- so an unpinned fetch would quietly invalidate both the cached MLX
# weights and the sampling defaults that were tuned against these commits.
BASE_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"

# Latent geometry. The Oobleck VAE downsamples by prod([2,4,4,6,10]) = 1920,
# so at 48 kHz one latent frame is 40 ms -> 25 frames per second.
SAMPLE_RATE = 48_000
VAE_HOP = 1920
LATENT_FPS = SAMPLE_RATE // VAE_HOP  # 25
LATENT_CHANNELS = 64


def check_vae_geometry(cfg: Mapping[str, Any]) -> None:
    """Fail if a VAE checkpoint contradicts the constants above.

    Those constants are the runtime's contract, not a cached copy of the
    config: conditioning sizes the latent window from ``LATENT_FPS`` before the
    VAE is ever loaded, and the DiT was trained against that rate. Deriving the
    hop from whatever config is on disk would therefore turn a swapped
    checkpoint into wrong-rate audio rather than an error -- so check it
    instead. The check is cheap and runs on the JSON, before any weights.
    """
    problems = []
    hop = prod(cfg["downsampling_ratios"])
    if hop != VAE_HOP:
        problems.append(
            f"hop {hop} (ratios {list(cfg['downsampling_ratios'])}) != {VAE_HOP}"
        )
    if cfg["sampling_rate"] != SAMPLE_RATE:
        problems.append(f"sampling rate {cfg['sampling_rate']} != {SAMPLE_RATE}")
    if cfg["decoder_input_channels"] != LATENT_CHANNELS:
        problems.append(
            f"latent channels {cfg['decoder_input_channels']} != {LATENT_CHANNELS}"
        )
    if problems:
        raise RuntimeError(
            "VAE checkpoint contradicts the runtime latent geometry: "
            + "; ".join(problems)
        )


@dataclass(frozen=True)
class ModelSpec:
    """A DiT checkpoint variant and its recommended sampling defaults."""

    key: str
    repo_id: str
    revision: str
    steps: int
    guidance: float
    supports_cfg: bool
    # DCW (wavelet-domain correction) was tuned for the distilled models. On
    # the non-distilled checkpoints it makes output mushy and distorted --
    # upstream issue #1259, where it is the single cause of the "garbled audio
    # on Apple Silicon" reports. Turbo wants shift=3.0, non-turbo wants 1.0.
    dcw: bool
    shift: float
    description: str


MODELS: dict[str, ModelSpec] = {
    "xl-sft": ModelSpec(
        key="xl-sft",
        repo_id="ACE-Step/acestep-v15-xl-sft",
        revision="d06de46b4622f781cf07f4a013a67d591ca52819",
        steps=50,
        guidance=7.0,
        supports_cfg=True,
        dcw=False,
        shift=1.0,
        description="4B DiT, supervised fine-tuned. Highest quality; 50 steps with CFG.",
    ),
    "xl-turbo": ModelSpec(
        key="xl-turbo",
        repo_id="ACE-Step/acestep-v15-xl-turbo",
        revision="d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee",
        steps=8,
        guidance=1.0,
        supports_cfg=False,
        dcw=True,
        shift=3.0,
        description="4B DiT, distilled. 8 steps, no CFG - roughly 6x faster.",
    ),
}

DEFAULT_MODEL = "xl-sft"


def cache_root() -> Path:
    """Directory holding converted MLX weights."""
    env = os.environ.get("AS15_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "as15"


@dataclass(frozen=True)
class Snapshot:
    """A downloaded checkpoint, identified by the commit it resolved to."""

    repo_id: str
    revision: str
    path: Path


def ensure_snapshot(
    repo_id: str, revision: str, allow_patterns: list[str] | None = None
) -> Snapshot:
    """Return a local snapshot of *repo_id* at *revision*, downloading if needed."""
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=repo_id, revision=revision, allow_patterns=allow_patterns
        )
    )
    # snapshot_download always lays snapshots out as ``snapshots/<commit sha>``,
    # so the directory name is the resolved commit even when *revision* is a
    # branch or tag. Prefer it over *revision* -- it is what the bytes on disk
    # actually are, which is what the converted-weight cache must be keyed on.
    return Snapshot(repo_id=repo_id, revision=path.name or revision, path=path)


def load_dit_config(snapshot: Path) -> SimpleNamespace:
    """Load a DiT ``config.json`` as a plain attribute bag.

    ``MLXDiTDecoder.from_config`` only does attribute access, so we avoid
    pulling in ``transformers.PretrainedConfig`` here.
    """
    cfg = json.loads((snapshot / "config.json").read_text())
    ns = SimpleNamespace(**cfg)
    # Fields the MLX decoder reads that may be absent from older configs.
    ns.head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    ns.sliding_window = cfg.get("sliding_window") or 128
    ns.rope_theta = cfg.get("rope_theta", 1_000_000.0)
    ns.max_position_embeddings = cfg.get("max_position_embeddings", 32768)
    ns.patch_size = cfg.get("patch_size", 2)
    return ns


def resolve(name: str) -> ModelSpec:
    """The registered checkpoint called *name*.

    Raises:
        ValueError: If no checkpoint goes by that name. This used to be a
            SystemExit, which is only ever right at the outermost CLI boundary:
            anything else embedding the package had its process torn down by an
            ordinary bad argument, with nothing to catch and no traceback.
    """
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model {name!r}. Choose one of: {', '.join(MODELS)}"
        ) from None
