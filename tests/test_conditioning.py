"""The one stage that still runs under PyTorch.

The checkpoint's own ``trust_remote_code`` modules do the encoding, so what is
worth pinning is everything around them: the trained prompt format, the token
budgets, the context-latent block this port assembles by hand, and which of
the two text-encoder entry points each stream goes through. All of it is
wrong-audio-shaped rather than crash-shaped -- the run succeeds either way.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
import numpy as np
import pytest
import torch

from as15 import conditioning, models

# --- the trained text format ---------------------------------------------


def test_chunk_mask_is_one_not_two():
    """Upstream's `chunk_masks_tensor[i] = 2.0` lands in a *bool* tensor.

    It therefore saturates to True and reaches the DiT as 1.0. Copying the
    literal 2.0 into the context channel pushes it outside the trained range
    and garbles the output.
    """
    assert conditioning.CHUNK_MASK_FULL == 1.0


def test_bool_assignment_saturates():
    """Pin the torch semantics the constant above is derived from."""
    mask = torch.stack([torch.ones(4, dtype=torch.bool)])
    mask[0] = 2.0
    assert mask.to(torch.float32).max().item() == 1.0


def test_metas_block_format():
    text = conditioning.format_metas(110, "C major", 4, 30.0)
    assert text == (
        "- bpm: 110\n- timesignature: 4\n- keyscale: C major\n- duration: 30 seconds\n"
    )
    # Unset fields must render as N/A, not None or an empty string.
    assert conditioning.format_metas(None, None, None, 12.7) == (
        "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 12 seconds\n"
    )


# --- token budget ---------------------------------------------------------


class _CharTokenizer:
    """A tokenizer whose tokens are characters -- only the count matters.

    Records how it was called, so the tests can also assert on what was *not*
    asked for.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.texts: list[str] = []

    def __call__(self, text: str, **kwargs):
        self.calls.append(kwargs)
        self.texts.append(text)
        ids = torch.arange(len(text), dtype=torch.long).reshape(1, -1)
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))


def _conditioner_with(tokenizer) -> conditioning.Conditioner:
    """A Conditioner holding a tokenizer and nothing else.

    Built with ``__new__``, so it has no models: reaching the end of the
    budget check would raise AttributeError rather than pass, which is the
    other half of what these tests pin -- input is rejected before the 1.2 B
    torch parameters are run, not after.
    """
    conditioner = conditioning.Conditioner.__new__(conditioning.Conditioner)
    conditioner.tokenizer = tokenizer
    return conditioner


def test_lyrics_the_encoder_cannot_read_are_rejected_not_truncated():
    """Upstream tokenises with ``truncation=True``.

    Lyrics over budget were cut there, and the run then succeeded: the song
    came back missing its last verses with nothing to say so, at the full cost
    of a generation. ``Conditioning.lyrics_text`` kept the whole sheet, so even
    printing what was conditioned on showed the input intact.
    """
    tokenizer = _CharTokenizer()

    with pytest.raises(conditioning.InputTooLong, match="lyric sheet"):
        _conditioner_with(tokenizer).build(
            style_prompt="dream pop", lyrics="l" * 4000, duration=30.0
        )

    assert tokenizer.calls, "the budget was checked without tokenising"
    assert all("truncation" not in kwargs for kwargs in tokenizer.calls)


def test_a_style_prompt_the_encoder_cannot_read_is_rejected():
    """The caption shares its 256 tokens with the instruction and metas lines.

    So the budget is smaller than it looks, and the message has to count what
    the encoder counts rather than what the user typed.
    """
    with pytest.raises(conditioning.InputTooLong, match="style prompt"):
        _conditioner_with(_CharTokenizer()).build(
            style_prompt="p" * 400, lyrics="", duration=30.0
        )


@pytest.mark.parametrize("tokens", [0, 1, 255, 256])
def test_input_that_fits_is_left_alone(tokens):
    """The bound must not reject input the encoder reads in full."""
    conditioning.check_token_budget("the style prompt", "x" * tokens, tokens, 256)


def test_the_rejection_says_how_much_to_cut():
    """Nobody can eyeball where 2048 tokens ends in their lyrics."""
    with pytest.raises(conditioning.InputTooLong) as exc:
        conditioning.check_token_budget("the lyrics", "y" * 4000, 2200, 2048)

    message = str(exc.value)
    assert "2200 tokens" in message
    assert "2048" in message
    assert "152 tokens" in message  # 2200 - 2048
    assert "276 characters" in message  # 152 of 2200 tokens, at 4000 characters


def test_the_budgets_are_the_lengths_upstream_tokenises_to():
    """These are trained lengths, not a limit this port chose."""
    assert conditioning.MAX_PROMPT_TOKENS == 256
    assert conditioning.MAX_LYRIC_TOKENS == 2048


# --- what build() assembles ----------------------------------------------
#
# The encoders are the checkpoint's own modules and are stubbed out here; what
# is left is what this port writes itself, and every one of these is a value
# the DiT was trained to expect rather than one it would reject.

SILENCE_FRAMES = 1000
ENCODER_WIDTH = 8


class _FakeTextEncoder:
    """Qwen3 with the two entry points conditioning uses, and no weights.

    Captions go through the full stack; lyrics take only the embedding table.
    Both are recorded, so a stream that started using the wrong one -- which
    costs a 0.6 B forward per generation, or drops the contextualisation the
    caption needs -- is visible.
    """

    def __init__(self) -> None:
        self.stack_calls: list[torch.Tensor] = []
        self.embed_calls: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor):
        self.stack_calls.append(input_ids)
        hidden = torch.full(
            (*input_ids.shape, ENCODER_WIDTH), 0.25, dtype=torch.float32
        )
        return SimpleNamespace(last_hidden_state=hidden)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.embed_calls.append(input_ids)
        return torch.full((*input_ids.shape, ENCODER_WIDTH), 0.5, dtype=torch.float32)


class _FakeConditionEncoder:
    """The checkpoint's AceStepConditionEncoder, reduced to its signature."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return torch.full((1, 3, ENCODER_WIDTH), 0.75, dtype=torch.bfloat16), None


def _write_checkpoint(directory: Path, shards: dict[str, dict]) -> Path:
    """Write a checkpoint of *shards*, indexed only if there is more than one.

    Which is how the real ones are published: a sharded checkpoint carries
    ``model.safetensors.index.json``, a small one is a bare
    ``model.safetensors`` with no index at all.
    """
    directory.mkdir(parents=True)
    for name, weights in shards.items():
        mx.save_safetensors(str(directory / name), weights)
    if len(shards) > 1:
        weight_map = {key: name for name, w in shards.items() for key in w}
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
    return directory


def _stub_conditioner(tmp_path: Path) -> conditioning.Conditioner:
    """A Conditioner with the two encoders and the silence latent stubbed.

    Everything replaced here is either the checkpoint's own code or a tensor
    read off disk; everything left is this port's.
    """
    snapshot = _write_checkpoint(
        tmp_path / "dit",
        {
            "model.safetensors": {
                conditioning.NULL_COND_KEY: mx.full((1, 1, ENCODER_WIDTH), 0.125)
            }
        },
    )

    conditioner = conditioning.Conditioner.__new__(conditioning.Conditioner)
    conditioner.device = torch.device("cpu")
    conditioner.dtype = torch.bfloat16
    conditioner.dit_snapshot = snapshot
    conditioner.tokenizer = _CharTokenizer()
    conditioner.text_encoder = _FakeTextEncoder()
    conditioner.encoder = _FakeConditionEncoder()
    # Distinct per frame and per channel, so a slice taken from the wrong axis
    # or at the wrong offset does not land on an equal value.
    conditioner.silence_latent = torch.arange(
        SILENCE_FRAMES * models.LATENT_CHANNELS, dtype=torch.float32
    ).reshape(1, SILENCE_FRAMES, models.LATENT_CHANNELS)
    return conditioner


def test_the_prompt_is_assembled_in_the_trained_format(tmp_path):
    """These strings are the format the condition encoder was trained on.

    They are copied verbatim from ``acestep.constants``, so a paraphrase --
    a dropped blank line, a reordered section -- is not a cosmetic change; it
    conditions the model on something it has not seen.
    """
    conditioner = _stub_conditioner(tmp_path)
    cond = conditioner.build(
        style_prompt="dream pop, warm tape",
        lyrics="[verse]\nCity lights",
        duration=20.0,
        language="fr",
        bpm=110,
        key_scale="C major",
        time_signature=4,
    )

    assert cond.text_prompt == (
        "# Instruction\n"
        "Fill the audio semantic mask based on the given conditions:\n\n"
        "# Caption\n"
        "dream pop, warm tape\n\n"
        "# Metas\n"
        "- bpm: 110\n"
        "- timesignature: 4\n"
        "- keyscale: C major\n"
        "- duration: 20 seconds\n"
        "<|endoftext|>\n"
    )
    assert (
        cond.lyrics_text
        == "# Languages\nfr\n\n# Lyric\n[verse]\nCity lights<|endoftext|>"
    )
    # Exactly what was tokenised, so nothing is conditioned on that the
    # returned strings do not show.
    assert conditioner.tokenizer.texts == [cond.text_prompt, cond.lyrics_text]


def test_an_instruction_without_its_colon_gets_one(tmp_path):
    """The trained instruction ends in a colon; a caller's need not."""
    cond = _stub_conditioner(tmp_path).build(
        style_prompt="x", lyrics="", duration=20.0, instruction="Write a song"
    )
    assert "# Instruction\nWrite a song:\n" in cond.text_prompt


def test_the_context_block_is_the_silence_latent_beside_a_full_chunk_mask(tmp_path):
    """``context_latents`` is [src | mask] on the channel axis, in that order.

    Both halves are this port's own construction: the source is the
    checkpoint's silence latent windowed to the requested length, and the mask
    says every frame is to be generated. Swapping the halves, or sizing either
    off the wrong axis, leaves a tensor of exactly the right shape.
    """
    conditioner = _stub_conditioner(tmp_path)
    cond = conditioner.build(style_prompt="x", lyrics="", duration=20.0)

    frames = 20 * models.LATENT_FPS
    assert cond.latent_frames == frames
    assert cond.context_latents.shape == (1, frames, 2 * models.LATENT_CHANNELS)

    source, mask = np.split(cond.context_latents, 2, axis=-1)
    expected = conditioner.silence_latent[:, :frames, :].numpy()
    assert np.array_equal(source, expected)
    assert np.array_equal(mask, np.full_like(mask, conditioning.CHUNK_MASK_FULL))


def test_the_lyrics_take_the_embedding_table_and_the_caption_the_whole_stack(tmp_path):
    """The lyric encoder does the contextualising, so lyrics skip Qwen3's body.

    Running them through the full stack instead would be a second 0.6 B
    forward per generation over the longer of the two streams, and would hand
    the lyric encoder inputs it was not trained on.
    """
    conditioner = _stub_conditioner(tmp_path)
    lyrics = "[verse]\nCity lights"
    cond = conditioner.build(style_prompt="dream pop", lyrics=lyrics, duration=20.0)

    text_encoder = conditioner.text_encoder
    assert len(text_encoder.stack_calls) == 1
    assert len(text_encoder.embed_calls) == 1
    assert text_encoder.stack_calls[0].shape[1] == len(cond.text_prompt)
    assert text_encoder.embed_calls[0].shape[1] == len(cond.lyrics_text)


def test_the_condition_encoder_is_given_the_window_and_dtype_it_expects(tmp_path):
    """Everything here is fixed for text2music and comes from nowhere else.

    There is no reference audio, so the timbre encoder is fed a fixed silence
    window purely so it still produces its aggregate token; the attention
    masks have to arrive as bool, not as the 0/1 ints the tokenizer returns;
    and the hidden states are cast to the conditioner's dtype so the torch
    stage runs at the width it was loaded at.
    """
    conditioner = _stub_conditioner(tmp_path)
    conditioner.build(style_prompt="x", lyrics="y", duration=20.0)

    kwargs = conditioner.encoder.kwargs
    refer = kwargs["refer_audio_acoustic_hidden_states_packed"]
    assert refer.shape == (1, conditioning.SILENCE_REFER_FRAMES, models.LATENT_CHANNELS)
    assert torch.equal(refer, conditioner.silence_latent[:, : refer.shape[1], :])

    order_mask = kwargs["refer_audio_order_mask"]
    assert order_mask.dtype == torch.long
    assert torch.equal(order_mask, torch.zeros(1, dtype=torch.long))

    assert kwargs["text_attention_mask"].dtype == torch.bool
    assert kwargs["lyric_attention_mask"].dtype == torch.bool
    assert kwargs["text_hidden_states"].dtype == conditioner.dtype
    assert kwargs["lyric_hidden_states"].dtype == conditioner.dtype


def test_the_diffusion_loop_is_handed_float32_numpy(tmp_path):
    """MLX takes numpy, and numpy has no bfloat16.

    The torch stage runs in bf16, so every tensor crossing into the sampler
    has to be widened on the way out -- ``np.array`` on a bf16 tensor raises
    rather than rounding, so this is a hard boundary rather than a preference.
    """
    cond = _stub_conditioner(tmp_path).build(style_prompt="x", lyrics="", duration=20.0)

    for array in (
        cond.encoder_hidden_states,
        cond.context_latents,
        cond.null_condition_emb,
    ):
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32

    assert np.allclose(cond.encoder_hidden_states, 0.75)
    assert np.allclose(cond.null_condition_emb, 0.125)


@pytest.mark.parametrize(
    ("duration", "frames"),
    [(20.0, 500), (0.0, 1), (0.01, 1), (20.04, 501)],
)
def test_the_latent_window_is_the_duration_at_the_latent_frame_rate(
    tmp_path, duration, frames
):
    """Rounded, and never empty: a zero-frame latent has nothing to decode.

    ``resolve_settings`` rejects the short durations well before this, so the
    floor is a second line rather than the only one -- but it is the one that
    holds for a caller driving the Conditioner directly.
    """
    cond = _stub_conditioner(tmp_path).build(
        style_prompt="x", lyrics="", duration=duration
    )
    assert cond.latent_frames == frames
    assert cond.context_latents.shape[1] == frames


def test_a_window_longer_than_the_silence_latent_is_tiled(tmp_path):
    """Ten minutes is 15000 frames, and the stored silence is far shorter."""
    conditioner = _stub_conditioner(tmp_path)
    available = conditioner.silence_latent.shape[1]

    tiled = conditioner.silence_slice(available + 3)
    assert tiled.shape[1] == available + 3
    assert torch.equal(tiled[:, :available, :], conditioner.silence_latent)
    assert torch.equal(tiled[:, available:, :], conditioner.silence_latent[:, :3, :])

    exact = conditioner.silence_slice(available)
    assert torch.equal(exact, conditioner.silence_latent)


# --- the null embedding CFG guides against --------------------------------


def test_the_null_embedding_is_read_from_whichever_shard_holds_it(tmp_path):
    """It used to be looked up through the index, and only through the index.

    Every other reader falls back to a single ``model.safetensors``, so a
    checkpoint published without an index loaded everywhere except here,
    which died on an index file that checkpoint never had.
    """
    sharded = _write_checkpoint(
        tmp_path / "sharded",
        {
            "a.safetensors": {"decoder.layers.0.weight": mx.zeros((2, 2))},
            "b.safetensors": {conditioning.NULL_COND_KEY: mx.full((1, 1, 4), 0.5)},
        },
    )
    single = _write_checkpoint(
        tmp_path / "single",
        {
            "model.safetensors": {
                "decoder.layers.0.weight": mx.zeros((2, 2)),
                conditioning.NULL_COND_KEY: mx.full((1, 1, 4), 0.5),
            }
        },
    )
    assert models.shard_files(single) == [single / "model.safetensors"]

    for snapshot in (sharded, single):
        emb = conditioning._load_null_condition_emb(snapshot)
        # fp32 from the checkpoint, not the bf16 the converter would have cast
        # it to: CFG is one branch of the sampler and does not want rounding
        # the DiT's own precision imposed on it.
        assert emb.dtype == np.float32
        assert emb.shape == (1, 1, 4)
        assert np.allclose(emb, 0.5)


def test_a_checkpoint_that_cannot_do_cfg_says_which_tensor_is_missing(tmp_path):
    snapshot = _write_checkpoint(
        tmp_path / "no-null",
        {"model.safetensors": {"decoder.layers.0.weight": mx.zeros((2, 2))}},
    )
    with pytest.raises(RuntimeError, match="CFG cannot be built"):
        conditioning._load_null_condition_emb(snapshot)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No safetensors weights"):
        models.shard_files(empty)


# --- lifetime -------------------------------------------------------------


def test_leaving_the_conditioner_block_releases_it(monkeypatch):
    """The pipeline delegates the torch stage's lifetime to the with-block.

    Built with ``__new__`` so the test costs nothing: release() is stubbed and
    a conditioner that never loaded a model has nothing to release anyway.
    """
    released: list[bool] = []
    conditioner = conditioning.Conditioner.__new__(conditioning.Conditioner)
    monkeypatch.setattr(conditioner, "release", lambda: released.append(True))

    with conditioner as entered:
        assert entered is conditioner
    assert released == [True]
