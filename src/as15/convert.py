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

import fcntl
import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
from tqdm import tqdm

from .atomic import publish
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
#
# Remembering to bump it is the whole of the protection, so it is not left to
# memory: the converters are fingerprinted in tests/test_regressions.py, which
# fails on an unaccompanied edit to either of them.
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


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _path_safe(component: str) -> str:
    """Flatten one identity component into a single path component.

    Aggressive on purpose: a revision is normally a hex commit but may be a
    branch or tag, and a repo ID is whatever the caller passed. Leading dots
    go too, so no component can be ``..`` and walk out of the cache root.
    """
    return _UNSAFE.sub("-", component).lstrip(".") or "unknown"


def _repo_dir(repo_id: str) -> str:
    """The cache directory for *repo_id*, and for no other repo.

    The path used to carry only the last component of the repo ID, on the
    reasoning that it reads better and the manifest catches a mismatch anyway.
    It does catch it -- but catching it is a ~8.3 GB reconversion, and
    ``org-a/model`` and ``org-b/model`` at the same commit and precision named
    one file, so two such checkpoints in alternation would invalidate and
    overwrite each other forever rather than each keeping a cache. Worse, the
    manifest check and the load are not one operation: another process can
    replace the file between them, so what is opened is not what was checked.

    Flattening the full ID is lossy in the other direction -- ``a-b/c`` and
    ``a/b-c`` flatten alike -- so a digest of the exact string goes on the end.
    Four bytes is far more than enough to separate a handful of repos, and a
    deliberate collision only puts two repos back in the directory they shared
    before, where the manifest still stands between them and a bad cache hit.
    """
    digest = hashlib.blake2b(repo_id.encode(), digest_size=4).hexdigest()
    return f"{_path_safe(repo_id)}-{digest}"


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


@contextmanager
def _cache_lock(target: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock over *target* for the block.

    Two processes reaching a cold cache together would otherwise each run the
    whole ~8.3 GB DiT conversion and race to publish the result. Locking
    around the conversion rather than around the write makes the second one
    wait and then hit the cache the first wrote.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Said only once we know we will actually block, because the wait
            # is as long as a conversion and looks like a hang otherwise.
            print(f"Waiting for another process to write {target.name} ...")
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    # The lock is released by the close above; the empty lock file stays, as
    # unlinking it would let a waiter lock a path nobody else can see.


def _write_cache(
    weights: dict[str, mx.array], out: Path, manifest: Mapping[str, object]
) -> None:
    """Publish *weights* then its manifest, so a crash leaves no valid cache.

    That order because :func:`_cache_hit` gates on the manifest: a reader
    arriving between the two sees weights it will not trust, which costs a
    reconversion. The reverse order would hand it a manifest vouching for
    weights that are not there yet.
    """
    publish(out, lambda tmp: mx.save_safetensors(str(tmp), weights))
    publish(
        _manifest_path(out),
        lambda tmp: tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True)),
    )


def _asset_path(repo_id: str, revision: str, filename: str) -> Path:
    return cache_root() / _repo_dir(repo_id) / _path_safe(revision) / filename


def dit_cache_path(repo_id: str, revision: str, precision: str) -> Path:
    return _asset_path(
        repo_id, revision, f"dit-v{DIT_CONVERTER_VERSION}-{precision}.safetensors"
    )


def _convert_dit_shard(
    source: Mapping[str, mx.array], dtype: mx.Dtype
) -> dict[str, mx.array]:
    """Map one checkpoint shard onto the subset the MLX DiT loads.

    Split out from :func:`convert_dit` because this, with the two functions
    above, is the whole of what ends up in the cache -- the rest of that
    function is where the file goes and who is allowed to write it. Keeping
    the two apart is what lets ``test_the_dit_converter_cannot_change_layout``
    fire on a change of output and stay quiet on a change of plumbing.
    """
    out: dict[str, mx.array] = {}
    for key, value in source.items():
        if key == NULL_COND_KEY:
            out[NULL_COND_KEY] = value.astype(dtype)
            continue
        mapped = _convert_dit_key(key)
        if mapped is None:
            continue
        new_key, layout = mapped
        out[new_key] = _apply_layout(value, layout).astype(dtype)
    return out


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

    out = dit_cache_path(snapshot.repo_id, snapshot.revision, precision)
    manifest: dict[str, object] = {
        "asset": "dit",
        "converter_version": DIT_CONVERTER_VERSION,
        "precision": precision,
        "repo_id": snapshot.repo_id,
        "revision": snapshot.revision,
    }
    if _cache_hit(out, manifest) and not force:
        return out

    with _cache_lock(out):
        # Whoever held the lock while we waited may have just written exactly
        # this cache, in which case converting again would only rewrite it.
        if _cache_hit(out, manifest) and not force:
            return out

        weights: dict[str, mx.array] = {}
        shards = _shard_files(snapshot.path)
        for shard in tqdm(shards, desc=f"Converting DiT -> {precision}", unit="shard"):
            source = mx.load(str(shard))
            weights.update(_convert_dit_shard(source, dtype))
            # Materialise this shard's conversions before dropping the source
            # dict, so the fp32 buffers are released before the next shard is
            # opened.
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


def vae_cache_path(repo_id: str, revision: str) -> Path:
    """Where the converted VAE for *repo_id* at *revision* is cached.

    The VAE lives in the shared base repo and every checkpoint decodes with
    it, so this is one file for all of them -- but it is one file *per base
    repo*, named the same way as the DiT rather than under a fixed ``vae/``
    directory, so a second base repo cannot land on it.
    """
    return _asset_path(
        repo_id, revision, f"vae-v{VAE_CONVERTER_VERSION}-fp32.safetensors"
    )


def _convert_vae_weights(source: Mapping[str, mx.array]) -> dict[str, mx.array]:
    """Fuse and re-layout every VAE tensor. See :func:`_convert_dit_shard`."""
    weights: dict[str, mx.array] = {}
    for key in tqdm(sorted(source), desc="Converting VAE", unit="tensor"):
        if key.endswith(".weight_v"):
            continue  # consumed with its .weight_g partner
        if key.endswith(".weight_g"):
            stem = key[: -len(".weight_g")]
            v_key = stem + ".weight_v"
            if v_key not in source:
                raise RuntimeError(f"{key} has no matching {v_key}")
            w = _fuse_weight_norm(source[key], source[v_key])
            # conv_t1 is the only ConvTranspose1d in this model.
            w = (
                mx.transpose(w, (1, 2, 0))
                if "conv_t1" in stem
                else mx.swapaxes(w, 1, 2)
            )
            weights[stem + ".weight"] = w
        elif key.endswith((".alpha", ".beta")):
            # Snake1d params: [1, C, 1] -> [C]
            weights[key] = source[key].astype(mx.float32).squeeze()
        else:
            weights[key] = source[key].astype(mx.float32)
    return weights


def convert_vae(base: Snapshot, force: bool = False) -> Path:
    """Fuse weight-norm and re-layout the Oobleck VAE for MLX.

    The VAE stays fp32: it is only 337 MB and runs once per generation.
    """
    out = vae_cache_path(base.repo_id, base.revision)
    manifest: dict[str, object] = {
        "asset": "vae",
        "converter_version": VAE_CONVERTER_VERSION,
        "precision": "fp32",
        "repo_id": base.repo_id,
        "revision": base.revision,
    }
    if _cache_hit(out, manifest) and not force:
        return out

    with _cache_lock(out):
        if _cache_hit(out, manifest) and not force:
            return out

        vae_dir = base.path / "vae"
        source = mx.load(str(vae_dir / "diffusion_pytorch_model.safetensors"))
        weights = _convert_vae_weights(source)
        mx.eval(list(weights.values()))

        _write_cache(weights, out, manifest)
    return out
