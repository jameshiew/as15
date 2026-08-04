"""Convert published PyTorch checkpoints into MLX-native weight caches.

The DiT checkpoints are published as fp32 safetensors shards (~20 GB for the
XL variants) that also carry the condition encoder and the FSQ tokenizer /
detokenizer. Only the ``decoder.*`` subtree runs on MLX, and the config
declares bfloat16, so we extract that subtree once, cast it, and cache it as a
single safetensors file (~8.3 GB).

Nothing here imports torch. Weights are read with ``mx.load``, which parses
safetensors into MLX arrays directly -- ``safetensors.safe_open`` cannot be
used here because even its "mlx" framework decodes via numpy, and numpy has no
bfloat16 dtype (the VAE ships bf16).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
from tqdm import tqdm

from .models import Snapshot, cache_root

# Precision labels, mapped to the ``mlx.core`` dtype name they select. Names
# rather than dtype objects: the sampler takes its compute dtype as a string
# and resolves it with getattr, so one table serves both and the two cannot
# drift apart.
PRECISIONS = {"bf16": "bfloat16", "fp32": "float32"}


def resolve_precision(precision: str) -> str:
    """Return the ``mlx.core`` dtype name for *precision*.

    Indexing :data:`PRECISIONS` directly surfaces a bare ``KeyError`` from
    wherever the label is first used, which for ``as15 download --precision
    typo`` was a traceback out of the converter rather than a usage error.
    """
    try:
        return PRECISIONS[precision]
    except KeyError:
        raise ValueError(
            f"Unknown precision {precision!r}. Choose one of: {', '.join(PRECISIONS)}"
        ) from None


# Emitted alongside the DiT weights; needed to build the CFG null branch.
NULL_COND_KEY = "null_condition_emb"

# Bump when a converter's output stops matching what the previous version
# wrote -- renamed keys, a changed axis permutation, a different dtype policy.
# The version is part of the cache path, so a bump orphans the old file rather
# than silently loading weights the current MLX models cannot interpret. The
# two converters version independently: the DiT cache is ~8.3 GB and there is
# no reason to rebuild it when only the 337 MB VAE conversion changes.
DIT_CONVERTER_VERSION = 1
VAE_CONVERTER_VERSION = 1


def _shard_files(snapshot: Path) -> list[Path]:
    """Return the safetensors shards of a checkpoint, in index order."""
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        names = sorted(set(weight_map.values()))
        return [snapshot / n for n in names]
    single = snapshot / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"No safetensors weights found in {snapshot}")


def _convert_dit_key(key: str) -> tuple[str, str] | None:
    """Map a checkpoint key to an ``(mlx_key, layout)`` pair, or None to skip.

    ``layout`` is one of ``"conv"``, ``"conv_t"`` or ``"plain"`` and selects the
    axis permutation applied to the value.
    """
    if not key.startswith("decoder."):
        return None
    key = key[len("decoder.") :]

    # RoPE tables are recomputed by the MLX model.
    if "rotary_emb" in key:
        return None

    # Upstream wraps the patch conv in Sequential(Lambda, Conv1d, Lambda) to
    # transpose around it; MLX is channels-last so the Lambdas disappear and
    # the conv moves from index 1 to the top level.
    if key.startswith("proj_in.1."):
        new = key.replace("proj_in.1.", "proj_in.")
        return new, ("conv" if new.endswith(".weight") else "plain")
    if key.startswith("proj_out.1."):
        new = key.replace("proj_out.1.", "proj_out.")
        return new, ("conv_t" if new.endswith(".weight") else "plain")

    return key, "plain"


def _apply_layout(value: mx.array, layout: str) -> mx.array:
    if layout == "conv":
        # PyTorch Conv1d [out, in, K] -> MLX [out, K, in]
        return mx.swapaxes(value, 1, 2)
    if layout == "conv_t":
        # PyTorch ConvTranspose1d [in, out, K] -> MLX [out, K, in]
        return mx.transpose(value, (1, 2, 0))
    return value


# ---------------------------------------------------------------------------
# Cache identity
#
# A converted file is only reusable for the exact (repo, commit, converter,
# precision) it was produced from. Upstream republishes weights under the same
# repo ID, so keying on the repo name alone would silently serve a cache built
# from different bytes -- and a mismatched-but-loadable DiT generates degraded
# audio rather than failing. The tuple goes in the path so distinct inputs
# cannot collide, and in a sidecar manifest so a reused file is checked rather
# than trusted.
# ---------------------------------------------------------------------------


def _path_safe(component: str) -> str:
    """Flatten a revision into a single path component."""
    return component.replace("/", "-").replace("\\", "-") or "unknown"


def _manifest_path(weights: Path) -> Path:
    return weights.with_suffix(".json")


def _cache_hit(weights: Path, manifest: Mapping[str, object]) -> bool:
    """True if *weights* exists and was written from exactly this input."""
    if not weights.exists():
        return False
    try:
        recorded = json.loads(_manifest_path(weights).read_text())
    except OSError, ValueError:
        return False  # never written, half-written, or hand-edited
    return recorded == manifest


def _write_cache(
    weights: dict[str, mx.array], out: Path, manifest: Mapping[str, object]
) -> None:
    """Write *weights* then its manifest, so a crash leaves no valid cache."""
    tmp = out.with_suffix(".tmp.safetensors")
    mx.save_safetensors(str(tmp), weights)
    tmp.replace(out)
    _manifest_path(out).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def dit_cache_path(cache_name: str, revision: str, precision: str) -> Path:
    return (
        cache_root()
        / cache_name
        / _path_safe(revision)
        / f"dit-v{DIT_CONVERTER_VERSION}-{precision}.safetensors"
    )


def convert_dit(
    snapshot: Snapshot,
    precision: str = "bf16",
    force: bool = False,
) -> Path:
    """Extract and cache the DiT decoder weights in MLX layout.

    Returns the path to the cached safetensors file.
    """
    # Before the cache path is built: an unknown label would otherwise name a
    # file that can never hit, and only fail once the conversion reached it.
    dtype = getattr(mx, resolve_precision(precision))

    out = dit_cache_path(snapshot.cache_name, snapshot.revision, precision)
    manifest: dict[str, object] = {
        "asset": "dit",
        "converter_version": DIT_CONVERTER_VERSION,
        "precision": precision,
        "repo_id": snapshot.repo_id,
        "revision": snapshot.revision,
    }
    if _cache_hit(out, manifest) and not force:
        return out

    out.parent.mkdir(parents=True, exist_ok=True)

    weights: dict[str, mx.array] = {}
    shards = _shard_files(snapshot.path)
    for shard in tqdm(shards, desc=f"Converting DiT -> {precision}", unit="shard"):
        source = mx.load(str(shard))
        for key, value in source.items():
            if key == NULL_COND_KEY:
                weights[NULL_COND_KEY] = value.astype(dtype)
                continue
            mapped = _convert_dit_key(key)
            if mapped is None:
                continue
            new_key, layout = mapped
            weights[new_key] = _apply_layout(value, layout).astype(dtype)
        # Materialise this shard's conversions before dropping the source dict,
        # so the fp32 buffers are released before the next shard is opened.
        mx.eval(list(weights.values()))
        del source
        mx.clear_cache()

    if NULL_COND_KEY not in weights:
        raise RuntimeError(
            f"{NULL_COND_KEY!r} missing from {snapshot.path}; CFG cannot be built."
        )

    _write_cache(weights, out, manifest)
    return out


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


def _fuse_weight_norm(g: mx.array, v: mx.array, eps: float = 1e-9) -> mx.array:
    """Fuse ``weight_norm``'s ``w = g * v / ||v||`` into a single tensor.

    ``g`` is a per-output-channel scale shaped ``[C, 1, 1]``; the norm is taken
    over every axis of ``v`` except the first.
    """
    v = v.astype(mx.float32)
    norm = mx.sqrt((v * v).sum(axis=(1, 2), keepdims=True))
    return g.astype(mx.float32) * v / (norm + eps)


def vae_cache_path(revision: str) -> Path:
    return (
        cache_root()
        / "vae"
        / _path_safe(revision)
        / f"vae-v{VAE_CONVERTER_VERSION}-fp32.safetensors"
    )


def convert_vae(base: Snapshot, force: bool = False) -> Path:
    """Fuse weight-norm and re-layout the Oobleck VAE for MLX.

    The VAE stays fp32: it is only 337 MB and runs once per generation.
    """
    out = vae_cache_path(base.revision)
    manifest: dict[str, object] = {
        "asset": "vae",
        "converter_version": VAE_CONVERTER_VERSION,
        "precision": "fp32",
        "repo_id": base.repo_id,
        "revision": base.revision,
    }
    if _cache_hit(out, manifest) and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    vae_dir = base.path / "vae"
    source = mx.load(str(vae_dir / "diffusion_pytorch_model.safetensors"))
    weights: dict[str, mx.array] = {}
    for key in tqdm(sorted(source), desc="Converting VAE", unit="tensor"):
        if key.endswith(".weight_v"):
            continue  # consumed with its .weight_g partner
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            v_key = base + ".weight_v"
            if v_key not in source:
                raise RuntimeError(f"{key} has no matching {v_key}")
            w = _fuse_weight_norm(source[key], source[v_key])
            # conv_t1 is the only ConvTranspose1d in this model.
            w = (
                mx.transpose(w, (1, 2, 0))
                if "conv_t1" in base
                else mx.swapaxes(w, 1, 2)
            )
            weights[base + ".weight"] = w
        elif key.endswith((".alpha", ".beta")):
            # Snake1d params: [1, C, 1] -> [C]
            weights[key] = source[key].astype(mx.float32).squeeze()
        else:
            weights[key] = source[key].astype(mx.float32)
    mx.eval(list(weights.values()))

    _write_cache(weights, out, manifest)
    return out
