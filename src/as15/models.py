"""Model registry, checkpoint resolution and cache paths."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

# Shared assets: VAE + Qwen3 text encoder (+ the 5Hz LM, which we do not use).
BASE_REPO = "ACE-Step/Ace-Step1.5"

# Latent geometry. The Oobleck VAE downsamples by prod([2,4,4,6,10]) = 1920,
# so at 48 kHz one latent frame is 40 ms -> 25 frames per second.
SAMPLE_RATE = 48_000
VAE_HOP = 1920
LATENT_FPS = SAMPLE_RATE // VAE_HOP  # 25
LATENT_CHANNELS = 64


@dataclass(frozen=True)
class ModelSpec:
    """A DiT checkpoint variant and its recommended sampling defaults."""

    key: str
    repo_id: str
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

    @property
    def cache_name(self) -> str:
        return self.repo_id.split("/")[-1]


MODELS: dict[str, ModelSpec] = {
    "xl-sft": ModelSpec(
        key="xl-sft",
        repo_id="ACE-Step/acestep-v15-xl-sft",
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


def ensure_snapshot(repo_id: str, allow_patterns: list[str] | None = None) -> Path:
    """Return a local snapshot dir for *repo_id*, downloading if needed."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns)
    )


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
    try:
        return MODELS[name]
    except KeyError:
        raise SystemExit(
            f"Unknown model {name!r}. Choose one of: {', '.join(MODELS)}"
        ) from None
