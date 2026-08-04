"""Stand-ins and fixtures used by more than one test module.

Everything here is sized so a test costs milliseconds: the sampler tests drive
a decoder that predicts a constant, and the requests are the smallest ones
``resolve_settings`` accepts. Nothing loads a checkpoint.
"""

from __future__ import annotations

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np


def request(**kwargs):
    """A request that resolve_settings accepts, before *kwargs* spoils it."""
    from as15.pipeline import GenerationRequest

    return GenerationRequest(style_prompt="a song", lyrics="", **kwargs)


# --- reading a FLAC container ---------------------------------------------
#
# Written out here rather than imported from as15.flac, which is what these
# check: a round trip through a writer and its own inverse agrees with itself
# about any consistent misreading of the format. Kept together because the
# pipeline's tests want the comments and the container's tests want the blocks
# they sit in.


def flac_blocks(data: bytes) -> list[tuple[int, bool, bytes]]:
    """Every metadata block as ``(type, is_last, payload)``."""
    assert data[:4] == b"fLaC"
    found, pos = [], 4
    while True:
        kind, last = data[pos] & 0x7F, bool(data[pos] & 0x80)
        length = int.from_bytes(data[pos + 1 : pos + 4], "big")
        found.append((kind, last, data[pos + 4 : pos + 4 + length]))
        pos += 4 + length
        if last:
            return found


def flac_frames(data: bytes) -> bytes:
    """Everything after the metadata: the encoded audio."""
    pos = 4
    while True:
        last = bool(data[pos] & 0x80)
        pos += 4 + int.from_bytes(data[pos + 1 : pos + 4], "big")
        if last:
            return data[pos:]


def flac_stream(blocks: list[tuple[int, bytes]], frames: bytes) -> bytes:
    """The inverse of :func:`flac_blocks`, for building a stream to feed in."""
    out = bytearray(b"fLaC")
    for index, (kind, payload) in enumerate(blocks):
        out.append((0x80 if index == len(blocks) - 1 else 0) | kind)
        out += len(payload).to_bytes(3, "big") + payload
    return bytes(out) + frames


def _comment_payload(data: bytes) -> bytes | None:
    return next((b for kind, _, b in flac_blocks(data) if kind == 4), None)


def flac_vendor(data: bytes) -> bytes | None:
    """The vendor string the comment block opens with, or None if there is none."""
    payload = _comment_payload(data)
    if payload is None:
        return None
    return payload[4 : 4 + int.from_bytes(payload[:4], "little")]


def flac_comments(data: bytes) -> dict[str, str]:
    """The comment block's ``NAME=value`` entries, parsed little-endian."""
    payload = _comment_payload(data)
    if payload is None:
        return {}

    def take(pos: int) -> tuple[bytes, int]:
        size = int.from_bytes(payload[pos : pos + 4], "little")
        return payload[pos + 4 : pos + 4 + size], pos + 4 + size

    _vendor, pos = take(0)
    count = int.from_bytes(payload[pos : pos + 4], "little")
    pos += 4

    tags = {}
    for _ in range(count):
        entry, pos = take(pos)
        name, _, value = entry.decode().partition("=")
        tags[name] = value
    assert pos == len(payload), "the block's own length disagrees with its entries"
    return tags


class ConstantDecoder:
    """Stand-in for the DiT that records the timestep of every evaluation.

    Predicts a velocity of 1.0, offset by batch row so that the conditional
    and unconditional halves of a CFG batch differ.
    """

    def __init__(self) -> None:
        self.timesteps: list[float] = []

    def __call__(
        self,
        *,
        hidden_states,
        timestep,
        timestep_r,
        encoder_hidden_states,
        context_latents,
        cache,
        use_cache,
    ):
        self.timesteps.append(float(np.array(timestep.astype(mx.float32))[0]))
        rows = mx.arange(hidden_states.shape[0]).reshape(-1, 1, 1)
        # A real decoder answers in its own dtype; an int32 ``arange`` here
        # would promote the result and hide the dtype leaks under test.
        rows = rows.astype(hidden_states.dtype)
        return mx.ones_like(hidden_states) + 0.1 * rows, cache


# shift=1.0 and infer_steps=3 give the schedule [1, 2/3, 1/3]; the interval
# that ends at t=0 is the one the loop used to special-case.
SCHEDULE = [1.0, 2 / 3, 1 / 3]
STEPS = len(SCHEDULE)
NOISE_SHAPE = (1, 4, 8)


def run_sampler(
    decoder,
    compute_dtype: str = "float32",
    seed: int | None = 0,
    **kwargs,
):
    from as15.mlx.sampler import mlx_generate_diffusion

    b, t, c = NOISE_SHAPE
    return mlx_generate_diffusion(
        decoder,
        encoder_hidden_states_np=np.zeros((b, 3, 6), dtype=np.float32),
        context_latents_np=np.zeros((b, t, c), dtype=np.float32),
        src_latents_shape=(b, t, c),
        seed=seed,
        infer_steps=STEPS,
        shift=1.0,
        dcw_enabled=False,
        disable_tqdm=True,
        compute_dtype=compute_dtype,
        **kwargs,
    )


def cfg_kwargs():
    return {
        "guidance_scale": 7.0,
        "null_condition_emb_np": np.zeros((1, 1, 6), dtype=np.float32),
    }


def small_decoder(num_hidden_layers: int = 2):
    """A decoder sized to the shapes :func:`run_sampler` feeds it.

    ``NOISE_SHAPE`` is (1, 4, 8), so the latents carry 8 channels and the
    context another 8, and the conditioning is 6-wide.
    """
    from as15.mlx.dit import MLXDiTDecoder

    return MLXDiTDecoder(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        in_channels=16,
        audio_acoustic_hidden_dim=8,
        patch_size=2,
        sliding_window=4,
        max_position_embeddings=64,
        encoder_hidden_size=6,
    )


def flat_parameters(module) -> dict[str, mx.array]:
    """A module's parameter tree as the flat ``name -> array`` map it loads from.

    Which is also the map a checkpoint is: the tests that convert weights
    build one of these, put it back in the layout it is published in, and
    convert it forward again.
    """
    from mlx.utils import tree_flatten

    return dict(tree_flatten(module.parameters()))


def randomised(module, scale: float = 0.3) -> dict[str, mx.array]:
    """Give *module* deterministic non-zero weights, and return their flat map.

    A model left at its initialisation has zeros in enough places -- every
    Snake ``alpha``/``beta``, the AdaLN table -- that a permuted or dropped
    weight can still produce the right answer. Seeded off the parameter's
    position so a run is reproducible without a global seed.
    """
    weights = {
        key: scale * mx.random.normal(value.shape, key=mx.random.key(i))
        for i, (key, value) in enumerate(flat_parameters(module).items())
    }
    module.load_weights(list(weights.items()))
    mx.eval(module.parameters())
    return weights
