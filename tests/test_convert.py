"""Weight conversion and the cache it is published into.

A conversion that goes wrong does not fail: the weights still load, the model
still runs, and what comes out is worse audio. So the layouts are pinned, the
converters are fingerprinted against their declared version, and the cache is
made to prove that a file it hands back was built from the bytes it claims.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np
import pytest

from as15 import atomic, conditioning, convert
from as15.models import BASE_REPO, BASE_REVISION, MODELS, Snapshot
from helpers import flat_parameters, randomised, small_decoder

# --- layouts --------------------------------------------------------------


def test_conv_layout_swaps_in_and_kernel_axes():
    # PyTorch Conv1d [out, in, K] -> MLX [out, K, in]
    w = mx.arange(2 * 3 * 5).reshape(2, 3, 5)
    out = convert._apply_layout(w, "conv")
    assert out.shape == (2, 5, 3)
    assert out[1, 4, 2].item() == w[1, 2, 4].item()


def test_conv_transpose_layout_moves_in_channels_last():
    # PyTorch ConvTranspose1d [in, out, K] -> MLX [out, K, in]
    w = mx.arange(2 * 3 * 5).reshape(2, 3, 5)
    out = convert._apply_layout(w, "conv_t")
    assert out.shape == (3, 5, 2)
    assert out[2, 4, 1].item() == w[1, 2, 4].item()


def test_dit_key_mapping_unwraps_sequential_and_drops_rope():
    assert convert._convert_dit_key("decoder.proj_in.1.weight") == (
        "proj_in.weight",
        "conv",
    )
    assert convert._convert_dit_key("decoder.proj_out.1.weight") == (
        "proj_out.weight",
        "conv_t",
    )
    assert convert._convert_dit_key("decoder.proj_in.1.bias") == (
        "proj_in.bias",
        "plain",
    )
    # RoPE tables are recomputed by the MLX model.
    assert (
        convert._convert_dit_key("decoder.layers.0.self_attn.rotary_emb.inv_freq")
        is None
    )
    # Only the DiT subtree is converted; the FSQ codec never runs for text2music.
    assert (
        convert._convert_dit_key("encoder.lyric_encoder.layers.0.mlp.up_proj.weight")
        is None
    )
    assert convert._convert_dit_key("tokenizer.quantizer.weight") is None


def test_the_dit_cache_holds_only_what_the_model_loads():
    """The converter used to also copy in the CFG null embedding.

    Nothing read it there. The DiT has no parameter by that name, so the
    loader popped it back out first -- and a pop is indiscriminate, so it
    would equally have hidden a key the converter emitted by mistake from
    ``load_weights``, which is otherwise strict about exactly this. The null
    embedding is read from the checkpoint instead, where CFG needs it before
    the conversion has necessarily happened at all.
    """
    source = {
        conditioning.NULL_COND_KEY: mx.zeros((1, 1, 4)),
        "decoder.layers.0.mlp.up_proj.weight": mx.zeros((2, 3)),
        "decoder.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((4,)),
        "encoder.lyric_encoder.layers.0.mlp.up_proj.weight": mx.zeros((2, 3)),
        "tokenizer.quantizer.weight": mx.zeros((2, 2)),
    }
    converted = convert._convert_dit_shard(source, mx.bfloat16)
    assert set(converted) == {"layers.0.mlp.up_proj.weight"}


def test_weight_norm_fusion_matches_definition():
    g = mx.array(np.array([[[2.0]], [[3.0]]], dtype=np.float32))
    v = mx.random.normal((2, 4, 3), key=mx.random.key(0))
    fused = np.array(convert._fuse_weight_norm(g, v))
    v_np = np.array(v)
    norms = np.linalg.norm(v_np.reshape(2, -1), axis=1).reshape(2, 1, 1)
    expected = np.array(g).reshape(2, 1, 1) * v_np / norms
    assert np.allclose(fused, expected, atol=1e-5)


# --- a whole DiT, converted and run --------------------------------------


def _checkpoint_key(mlx_key: str) -> str:
    """The published name for the parameter the MLX model calls *mlx_key*.

    Upstream wraps both patch convolutions in a ``Sequential`` with the
    transposes MLX does not need, so they are published one level down --
    which is the part of the mapping a round trip built from the MLX names
    alone would otherwise never exercise.
    """
    for projection in ("proj_in", "proj_out"):
        if mlx_key.startswith(f"{projection}."):
            return f"decoder.{projection}.1.{mlx_key[len(projection) + 1 :]}"
    return f"decoder.{mlx_key}"


def _checkpoint_layout(mlx_key: str, value: mx.array) -> mx.array:
    """Put *value* back in the axis order the checkpoint stores it in.

    Written from PyTorch's shapes rather than by inverting ``_apply_layout``:
    a ``Conv1d`` weight is ``[out, in, K]`` and a ``ConvTranspose1d`` weight
    is ``[in, out, K]``, against MLX's ``[out, K, in]`` for both.
    """
    if mlx_key == "proj_in.weight":
        return mx.transpose(value, (0, 2, 1))
    if mlx_key == "proj_out.weight":
        return mx.transpose(value, (2, 0, 1))
    return value


def test_a_converted_dit_computes_what_the_checkpoint_did():
    """The conversion, end to end, on a decoder small enough to run twice.

    The pieces are pinned individually above, but the mapping is a table and
    a table is where a key goes missing: a parameter the converter never
    emits leaves the model at its initialisation, and one it emits under a
    name the model does not have is the reason ``load_weights`` is strict.
    Only running both models says the values also arrived the right way up.

    The comparison is exact. Nothing in the conversion is arithmetic -- it
    permutes axes and casts -- so at fp32 there is nothing to round.
    """
    reference_model = small_decoder()
    weights = randomised(reference_model)

    published = {
        _checkpoint_key(key): _checkpoint_layout(key, value)
        for key, value in weights.items()
    }
    # The rest of the checkpoint, which is most of it by size and none of it
    # by use: the condition encoder, the FSQ codec, the recomputed RoPE table
    # and the null embedding conditioning reads for itself.
    published.update(
        {
            "encoder.lyric_encoder.layers.0.mlp.up_proj.weight": mx.zeros((2, 3)),
            "tokenizer.quantizer.weight": mx.zeros((2, 2)),
            "decoder.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((4,)),
            conditioning.NULL_COND_KEY: mx.zeros((1, 1, 6)),
        }
    )

    converted = convert._convert_dit_shard(published, mx.float32)
    assert set(converted) == set(weights)

    loaded = small_decoder()
    loaded.load_weights(list(converted.items()))
    mx.eval(loaded.parameters())

    def _forward(model):
        out, _ = model(
            hidden_states=mx.random.normal((1, 7, 8), key=mx.random.key(101)),
            timestep=mx.full((1,), 0.6),
            timestep_r=mx.full((1,), 0.3),
            encoder_hidden_states=mx.random.normal((1, 3, 6), key=mx.random.key(102)),
            context_latents=mx.random.normal((1, 7, 8), key=mx.random.key(103)),
        )
        return np.array(out)

    assert np.array_equal(_forward(loaded), _forward(reference_model))


def test_a_checkpoint_the_model_has_no_parameter_for_is_rejected():
    """The other half of the test above: strictness is what makes it a check.

    A converter that stopped emitting a key, or renamed one, would otherwise
    give a model that runs with half of it left at initialisation.
    """
    model = small_decoder()
    weights = randomised(model)

    del weights["layers.0.mlp.up_proj.weight"]
    with pytest.raises(ValueError):
        small_decoder().load_weights(list(weights.items()))

    weights["layers.0.mlp.up_proj.weight"] = mx.zeros((64, 32))
    weights["layers.0.mlp.sideways_proj.weight"] = mx.zeros((64, 32))
    with pytest.raises(ValueError):
        small_decoder().load_weights(list(weights.items()))


def test_the_precision_label_reaches_every_converted_tensor():
    """The DiT is converted once and read back for every run afterwards.

    A tensor left at the checkpoint's fp32 doubles what that layer costs to
    read for the life of the cache, and one cast further than asked for
    quietly changes what ``--precision fp32`` means.
    """
    published = {
        _checkpoint_key(key): _checkpoint_layout(key, value)
        for key, value in flat_parameters(small_decoder()).items()
    }

    for label, dtype in ((mx.bfloat16, mx.bfloat16), (mx.float32, mx.float32)):
        converted = convert._convert_dit_shard(published, label)
        assert {value.dtype for value in converted.values()} == {dtype}


def test_an_unknown_precision_is_a_value_error_not_a_key_error():
    """``as15 download --precision typo`` used to traceback out of the converter."""
    with pytest.raises(ValueError, match="Unknown precision"):
        convert.resolve_precision("typo")
    assert convert.resolve_precision("bf16") == "bfloat16"
    assert convert.resolve_precision("fp32") == "float32"


# --- converted-weight cache identity -------------------------------------


def test_cache_paths_separate_every_identity_component(monkeypatch, tmp_path):
    """Repo, commit, converter version and precision must not alias."""
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    paths = {
        convert.dit_cache_path("ACE-Step/acestep-v15-xl-sft", "aaaa", "bf16"),
        convert.dit_cache_path("ACE-Step/acestep-v15-xl-sft", "aaaa", "fp32"),
        convert.dit_cache_path("ACE-Step/acestep-v15-xl-sft", "bbbb", "bf16"),
        convert.dit_cache_path("ACE-Step/acestep-v15-xl-turbo", "aaaa", "bf16"),
        convert.vae_cache_path("ACE-Step/Ace-Step1.5", "aaaa"),
        convert.vae_cache_path("ACE-Step/Ace-Step1.5", "bbbb"),
    }
    assert len(paths) == 6
    # The converter version is in the filename, so a bump orphans the old file
    # rather than reusing weights in a layout the MLX models no longer expect.
    assert f"-v{convert.DIT_CONVERTER_VERSION}-" in (
        convert.dit_cache_path("m", "aaaa", "bf16").name
    )
    assert f"-v{convert.VAE_CONVERTER_VERSION}-" in (
        convert.vae_cache_path("m", "aaaa").name
    )


def test_the_whole_repo_id_names_the_cache_directory(monkeypatch, tmp_path):
    """Two repos that share a last component must not share a cache.

    They did: the directory was ``repo_id.split("/")[-1]``, so at the same
    commit and precision these two named one file. The manifest kept either
    from loading the other's weights, but only by reconverting ~8.3 GB every
    time the other was used -- and it is checked, not held, so a concurrent
    run can still swap the file between the check and the open.
    """
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    a = convert.dit_cache_path("org-a/model", "aaaa", "bf16")
    b = convert.dit_cache_path("org-b/model", "aaaa", "bf16")
    assert a.parent.parent != b.parent.parent

    # Flattening alone would not have been enough either.
    assert convert._repo_dir("a-b/c") != convert._repo_dir("a/b-c")
    # The readable part survives, so the directory is still greppable.
    assert convert._repo_dir("org-a/model").startswith("org-a-model-")


def test_a_revision_is_flattened_into_one_path_component(monkeypatch, tmp_path):
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    path = convert.dit_cache_path("m", "refs/pr/7", "bf16")
    assert path.parent.name == "refs-pr-7"
    assert path.parent.parent.name == convert._repo_dir("m")


def test_no_identity_component_can_escape_the_cache_root(monkeypatch, tmp_path):
    """Both components are attacker-adjacent: they come from a repo ID."""
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    path = convert.dit_cache_path("../../etc", "../..", "bf16")
    assert tmp_path in path.resolve().parents
    assert convert._path_safe("") == "unknown"


# --- converter fingerprints ----------------------------------------------
#
# The cache is keyed on a converter version that a human has to remember to
# bump, and forgetting is the one failure the cache cannot catch: a stale file
# still loads, and a DiT in a layout the MLX model half-understands generates
# degraded audio rather than raising. So the code that decides what goes in
# the cache is fingerprinted here, and an edit to it fails the suite.
#
# Add a row per version; do not edit one in place unless the output really is
# unchanged (a rename, a comment moved into code), and say so in the commit.
CONVERTER_DIGESTS = {
    ("dit", 1): "1bacd5368711b479",
    ("dit", 2): "84c5bcb0a6657909",  # stopped copying null_condition_emb in
    ("vae", 1): "485e8044cc390bb1",
    ("vae", 2): "d8c7def703249a82",  # stopped converting the encoder subtree
}


def _layout_digest(*functions) -> str:
    """Digest what *functions* do, ignoring how they are presented.

    Through the AST, so a comment, a reworded docstring or a reflow does not
    fire the gate, and ``ast.dump`` omits positions by default, so neither
    does moving a function within the file. Names are kept: a renamed weight
    key is exactly the change this is here to catch, and a renamed local is
    cheap enough to re-fingerprint.
    """
    import ast
    import hashlib
    import inspect
    import textwrap

    chunks = []
    for function in functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list) or not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]  # ty: ignore[unresolved-attribute]
        chunks.append(ast.dump(tree))
    return hashlib.blake2b("\n".join(chunks).encode(), digest_size=8).hexdigest()


_BUMP = (
    "The {asset} converter's output changed. Bump {const} in src/as15/convert.py "
    "and add ({asset!r}, <new version>): {digest!r} to CONVERTER_DIGESTS, so that "
    "weights already converted by the old one are orphaned rather than loaded "
    "into a model that no longer agrees with them. If the output is provably "
    "unchanged, update the existing row instead and say why in the commit."
)


def test_the_dit_converter_cannot_change_layout_without_a_version_bump():
    digest = _layout_digest(
        convert._convert_dit_key,
        convert._apply_layout,
        convert._convert_dit_shard,
    )
    assert digest == CONVERTER_DIGESTS[("dit", convert.DIT_CONVERTER_VERSION)], (
        _BUMP.format(asset="dit", const="DIT_CONVERTER_VERSION", digest=digest)
    )


def test_the_vae_converter_cannot_change_layout_without_a_version_bump():
    digest = _layout_digest(convert._fuse_weight_norm, convert._convert_vae_weights)
    assert digest == CONVERTER_DIGESTS[("vae", convert.VAE_CONVERTER_VERSION)], (
        _BUMP.format(asset="vae", const="VAE_CONVERTER_VERSION", digest=digest)
    )


# The same conversion, documented two ways. A gate that fires on prose is one
# people learn to re-fingerprint without reading the diff, which is the same
# as not having it.
def _documented_one_way():
    def convert(value, layout):
        """Permute a checkpoint tensor into MLX layout."""
        return mx.swapaxes(value, 1, 2)

    return convert


def _documented_another_way():
    def convert(value, layout):
        """PyTorch Conv1d [out, in, K] -> MLX [out, K, in].

        Reworded entirely, and with a comment the other one does not have.
        """
        # The axes, not the values.
        return mx.swapaxes(value, 1, 2)

    return convert


def test_the_fingerprint_fires_on_a_changed_permutation_and_not_on_prose():
    """A gate is only worth having if it catches the mistake it is for."""

    def transposed_the_other_way(value, layout):
        return mx.swapaxes(value, 0, 1)

    one_way = _documented_one_way()
    assert _layout_digest(one_way) != _layout_digest(transposed_the_other_way)
    assert _layout_digest(one_way) == _layout_digest(_documented_another_way())


# --- publishing a cache ---------------------------------------------------


def _dit_manifest(snapshot: Snapshot, precision: str = "bf16") -> dict:
    return {
        "asset": "dit",
        "converter_version": convert.DIT_CONVERTER_VERSION,
        "precision": precision,
        "repo_id": snapshot.repo_id,
        "revision": snapshot.revision,
    }


def test_cache_is_reused_only_when_the_manifest_matches(monkeypatch, tmp_path):
    """A weights file with no matching manifest must not be trusted.

    Loading a DiT converted from different bytes does not fail -- it generates
    audibly worse audio -- so the check has to happen before reuse.
    """
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    snapshot = Snapshot("ACE-Step/acestep-v15-xl-sft", "a" * 40, tmp_path / "snap")
    out = convert.dit_cache_path(snapshot.repo_id, snapshot.revision, "bf16")
    out.parent.mkdir(parents=True)
    out.write_bytes(b"not really safetensors")

    manifest = _dit_manifest(snapshot)
    assert not convert._cache_hit(out, manifest)  # no manifest written yet

    convert._manifest_path(out).write_text(json.dumps(manifest))
    assert convert._cache_hit(out, manifest)

    # A manifest describing anything else is a miss, not a silent reuse.
    stale = Snapshot(snapshot.repo_id, "b" * 40, snapshot.path)
    assert not convert._cache_hit(out, _dit_manifest(stale))
    assert not convert._cache_hit(out, _dit_manifest(snapshot, precision="fp32"))
    convert._manifest_path(out).write_text("{ truncated")
    assert not convert._cache_hit(out, manifest)


def test_only_one_process_at_a_time_may_write_a_cache(monkeypatch, tmp_path):
    """Two cold starts used to each run the same ~8.3 GB DiT conversion.

    They also raced on one fixed temporary name, so either could rename the
    other's half-written file over the real cache.
    """
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    out = convert.dit_cache_path("m", "a" * 40, "bf16")
    lock = out.with_suffix(".lock")

    with convert._cache_lock(out):
        assert lock.exists()  # and the parent directory was created for it
        # flock is held per open file description, so a second handle behaves
        # exactly like a second process.
        with lock.open("w") as rival, pytest.raises(BlockingIOError):
            fcntl.flock(rival, fcntl.LOCK_EX | fcntl.LOCK_NB)

    # Released on the way out, so the next writer proceeds immediately.
    with lock.open("w") as rival:
        fcntl.flock(rival, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_concurrent_writers_never_share_a_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    out = convert.dit_cache_path("m", "a" * 40, "bf16")
    out.parent.mkdir(parents=True)

    seen: list[Path] = []

    def record(tmp: Path) -> None:
        seen.append(tmp)
        tmp.write_bytes(b"")

    atomic.publish(out, record)
    atomic.publish(out, record)

    assert seen[0] != seen[1]
    # Same directory, so replace() is a rename and not a copy across devices.
    assert all(tmp.parent == out.parent for tmp in seen)
    # mx.save_safetensors dispatches on the extension and fails with a bare
    # FileNotFoundError if the temporary does not carry it.
    assert all(tmp.name.endswith(".safetensors") for tmp in seen)
    assert out.read_bytes() == b""


def test_a_failed_write_leaves_neither_a_partial_cache_nor_debris(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    out = convert.vae_cache_path(BASE_REPO, "a" * 40)
    out.parent.mkdir(parents=True)

    def explode(tmp: Path) -> None:
        tmp.write_bytes(b"half a tensor")
        raise RuntimeError("no space left on device")

    with pytest.raises(RuntimeError):
        atomic.publish(out, explode)

    assert not out.exists()
    assert list(out.parent.iterdir()) == []


def test_shared_vae_cache_is_keyed_on_the_base_repo(monkeypatch, tmp_path):
    """The VAE is shared across checkpoints, so only the base repo pins it."""
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    out = convert.vae_cache_path(BASE_REPO, BASE_REVISION)
    out.parent.mkdir(parents=True)
    out.write_bytes(b"")
    manifest = {
        "asset": "vae",
        "converter_version": convert.VAE_CONVERTER_VERSION,
        "precision": "fp32",
        "repo_id": BASE_REPO,
        "revision": BASE_REVISION,
    }
    convert._manifest_path(out).write_text(json.dumps(manifest))
    assert convert._cache_hit(out, manifest)
    assert not convert._cache_hit(convert.vae_cache_path(BASE_REPO, "c" * 40), manifest)
    # Both assets are named the same way, so the DiT and the VAE of one repo
    # share a directory and differ by filename rather than by scheme.
    assert (
        convert.vae_cache_path(BASE_REPO, BASE_REVISION).parent
        == convert.dit_cache_path(BASE_REPO, BASE_REVISION, "bf16").parent
    )


def test_a_registered_checkpoint_converts_to_a_path_under_the_cache_root(
    monkeypatch, tmp_path
):
    """Every registered model has to name a cache file, not just the two here."""
    monkeypatch.setenv("AS15_CACHE", str(tmp_path))
    for spec in MODELS.values():
        path = convert.dit_cache_path(spec.repo_id, spec.revision, "bf16")
        assert tmp_path in path.parents
        assert path.suffix == ".safetensors"
