"""Text/lyric conditioning for the DiT.

This is the one stage that still runs under PyTorch. It covers the 0.61 B
``encoder.*`` subtree of the checkpoint (text projector + lyric encoder +
timbre encoder) plus the 0.6 B Qwen3 text encoder, and it runs exactly once
per generation -- the 4.17 B DiT that runs 8-50 times is pure MLX.

Using the checkpoint's own ``trust_remote_code`` modules here means the
conditioning is bit-for-bit the reference implementation rather than a
reimplementation that could silently drift.

The FSQ ``tokenizer``/``detokenizer`` subtrees are deliberately never
instantiated: they only matter for cover/audio-code tasks, and for
text2music their output is discarded by ``torch.where(is_covers > 0, ...)``.
"""

from __future__ import annotations

import copy
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .models import latent_frames_for, shard_files

# The CFG null branch's condition embedding, stored at the top level of the DiT
# checkpoint. Read here and nowhere else: the converter used to copy it into
# the MLX cache as well, where nothing ever read it -- the loader popped it
# straight back out -- so the cache carried a second, bf16-rounded copy of a
# tensor whose only consumer takes the fp32 original.
NULL_COND_KEY = "null_condition_emb"

# Verbatim from acestep.constants -- these strings are part of the trained
# prompt format, so they must not be paraphrased.
DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""

# Reference-audio window used when no reference audio is supplied.
SILENCE_REFER_FRAMES = 750

# What the two conditioning streams fit, in Qwen3 tokens. Upstream passes
# these as `max_length` with `truncation=True`; they are the lengths the
# condition encoder was trained against, so they cannot simply be raised.
MAX_PROMPT_TOKENS = 256
MAX_LYRIC_TOKENS = 2048

# Chunk mask for whole-song text2music: every frame is generated.
#
# Upstream's "auto" mode looks like it writes 2.0 into the mask
# (`chunk_masks_tensor[i] = 2.0`), but that tensor is built with
# dtype=torch.bool, so the assignment saturates to True and reaches the model
# as 1.0. Feeding an actual 2.0 here puts the context channel far outside the
# range the DiT was trained on and garbles the output.
CHUNK_MASK_FULL = 1.0


def format_metas(
    bpm: int | str | None,
    key_scale: str | None,
    time_signature: str | int | None,
    duration: float,
) -> str:
    """Render the ``# Metas`` block in the trained format."""
    return (
        f"- bpm: {bpm or 'N/A'}\n"
        f"- timesignature: {time_signature or 'N/A'}\n"
        f"- keyscale: {key_scale or 'N/A'}\n"
        f"- duration: {int(duration)} seconds\n"
    )


def format_lyrics(lyrics: str, language: str) -> str:
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"


class InputTooLong(ValueError):
    """Conditioning input longer than the encoder reads."""


def check_token_budget(what: str, text: str, tokens: int, limit: int) -> None:
    """Reject input the encoder would only ever see the start of.

    Upstream tokenises with ``truncation=True``, so a caption or a lyric sheet
    over budget is silently cut and the run succeeds: the song comes back
    missing its last verses, and nothing in the output says so. The full
    strings are still on ``Conditioning``, so even a debug print of what was
    conditioned on shows the whole input.

    The character figure is an estimate -- it assumes the rest of *text*
    tokenises at the same rate as the part measured -- and is there because
    nobody can eyeball where 263 tokens ends in their lyrics.

    Raises:
        InputTooLong: naming the stream, its size and the budget.
    """
    if tokens <= limit:
        return
    over = tokens - limit
    raise InputTooLong(
        f"{what} is {tokens} tokens; the conditioning encoder reads at most "
        f"{limit}. Cut about {over} tokens (~{round(over * len(text) / tokens)} "
        f"characters)."
    )


@dataclass
class Conditioning:
    """Everything the MLX diffusion loop needs, as numpy arrays."""

    encoder_hidden_states: np.ndarray  # [B, L, encoder_hidden_size]
    context_latents: np.ndarray  # [B, T, 2 * latent_channels]
    null_condition_emb: np.ndarray  # [1, 1, encoder_hidden_size]
    latent_frames: int
    text_prompt: str
    lyrics_text: str


def _pick_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Conditioner:
    """Loads the text encoder + condition encoder and builds DiT conditioning."""

    def __init__(
        self,
        dit_snapshot: Path,
        base_snapshot: Path,
        device: str | None = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ):
        from transformers import AutoModel, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.device = _pick_device(device)
        self.dtype = dtype
        self.dit_snapshot = dit_snapshot

        config_json = json.loads((dit_snapshot / "config.json").read_text())
        remote_module = config_json["auto_map"]["AutoModel"].split(".")[0]

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(dit_snapshot, trust_remote_code=True)

        # XL checkpoints size the encoder independently of the decoder.
        encoder_config = copy.deepcopy(config)
        encoder_config.hidden_size = config.encoder_hidden_size
        encoder_config.intermediate_size = config.encoder_intermediate_size
        encoder_config.num_attention_heads = config.encoder_num_attention_heads
        encoder_config.num_key_value_heads = config.encoder_num_key_value_heads

        encoder_cls = get_class_from_dynamic_module(
            f"{remote_module}.AceStepConditionEncoder", str(dit_snapshot)
        )
        # Built on CPU rather than the meta device: the rotary embeddings keep
        # `inv_freq` as a *non-persistent* buffer, so it is absent from the
        # state dict and only ever gets its value from __init__. A meta build
        # would leave it uninitialised.
        encoder = encoder_cls(encoder_config)
        encoder.load_state_dict(
            _load_encoder_state_dict(dit_snapshot, dtype), assign=True, strict=True
        )
        self.encoder = encoder.to(self.device).eval()

        text_dir = base_snapshot / "Qwen3-Embedding-0.6B"
        # `from_pretrained` is typed as possibly returning None; checking here
        # turns a missing/broken tokenizer into a clear error at load time
        # rather than a `NoneType is not callable` inside `build`.
        tokenizer = AutoTokenizer.from_pretrained(text_dir)
        if tokenizer is None:
            raise RuntimeError(f"No usable tokenizer in {text_dir}")
        self.tokenizer = tokenizer
        self.text_encoder = (
            AutoModel.from_pretrained(text_dir, dtype=dtype).to(self.device).eval()
        )

        silence = torch.load(
            dit_snapshot / "silence_latent.pt", map_location="cpu", weights_only=True
        )
        # Stored [1, 64, T]; everything downstream of here is [1, T, 64].
        self.silence_latent = silence.transpose(1, 2).to(self.device, dtype)

    def silence_slice(self, frames: int) -> torch.Tensor:
        """Return ``frames`` frames of silence latent, tiling if necessary."""
        available = self.silence_latent.shape[1]
        if frames <= available:
            return self.silence_latent[:, :frames, :]
        reps = -(-frames // available)  # ceil
        return self.silence_latent.repeat(1, reps, 1)[:, :frames, :]

    @torch.inference_mode()
    def build(
        self,
        style_prompt: str,
        lyrics: str,
        duration: float,
        language: str = "en",
        bpm: int | str | None = None,
        key_scale: str | None = None,
        time_signature: str | int | None = None,
        instruction: str = DEFAULT_DIT_INSTRUCTION,
    ) -> Conditioning:
        if not instruction.endswith(":"):
            instruction += ":"

        metas = format_metas(bpm, key_scale, time_signature, duration)
        text_prompt = SFT_GEN_PROMPT.format(instruction, style_prompt, metas)
        lyrics_text = format_lyrics(lyrics, language)

        # Tokenised without `truncation=True`, so that over-budget input can be
        # rejected below instead of quietly becoming a shorter song. For input
        # that fits, this is the same tensor the truncating call returned:
        # truncation only rewrites an encoding once it is longer than
        # max_length, verified at the boundary (256 tokens in, 256 out,
        # identical ids; 257 in, 256 out, last token replaced).
        text = self.tokenizer(text_prompt, return_tensors="pt")
        lyric = self.tokenizer(lyrics_text, return_tensors="pt")
        check_token_budget(
            "the style prompt (with the instruction and metas lines)",
            text_prompt,
            text.input_ids.shape[1],
            MAX_PROMPT_TOKENS,
        )
        check_token_budget(
            "the lyric sheet (with the language header)",
            lyrics_text,
            lyric.input_ids.shape[1],
            MAX_LYRIC_TOKENS,
        )

        text_ids = text.input_ids.to(self.device)
        lyric_ids = lyric.input_ids.to(self.device)
        text_mask = text.attention_mask.to(self.device).bool()
        lyric_mask = lyric.attention_mask.to(self.device).bool()

        # Captions go through the full Qwen3 stack; lyrics use only its
        # embedding table (the lyric encoder does the contextualising).
        text_hidden = self.text_encoder(input_ids=text_ids).last_hidden_state
        lyric_hidden = self.text_encoder.embed_tokens(lyric_ids)

        # No reference audio for text2music: upstream feeds a fixed silence
        # window so the timbre encoder still produces its aggregate token.
        refer = self.silence_slice(SILENCE_REFER_FRAMES)
        refer_order_mask = torch.zeros(1, dtype=torch.long, device=self.device)

        encoder_hidden_states, _ = self.encoder(
            text_hidden_states=text_hidden.to(self.dtype),
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hidden.to(self.dtype),
            lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=refer,
            refer_audio_order_mask=refer_order_mask,
        )

        frames = latent_frames_for(duration)
        src_latents = self.silence_slice(frames)
        chunk_masks = torch.full_like(src_latents, CHUNK_MASK_FULL)
        context_latents = torch.cat([src_latents, chunk_masks], dim=-1)

        null_emb = _load_null_condition_emb(self.dit_snapshot)

        return Conditioning(
            encoder_hidden_states=_to_numpy(encoder_hidden_states),
            context_latents=_to_numpy(context_latents),
            null_condition_emb=null_emb,
            latent_frames=frames,
            text_prompt=text_prompt,
            lyrics_text=lyrics_text,
        )

    def __enter__(self) -> Conditioner:
        """Use as a context manager to bound the torch stage's lifetime.

        The 4.17 B MLX DiT is loaded straight after this, so the ~2.4 GB of
        torch models held here have to go back whether conditioning succeeded
        or raised -- a caller that catches the failure and retries otherwise
        starts the next attempt that much closer to the memory ceiling, and
        fails somewhere with no obvious connection to the first failure.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def release(self) -> None:
        """Drop the torch models so their memory is free before MLX loads.

        Terminal and idempotent: the conditioner cannot be used afterwards.
        The attributes are deleted rather than set to ``None`` so that a
        use-after-release raises an ``AttributeError`` naming the missing
        model, instead of ``NoneType is not callable`` from inside ``build``.
        """
        for name in ("encoder", "text_encoder", "silence_latent"):
            self.__dict__.pop(name, None)
        # Collect before emptying the cache rather than after: a tensor whose
        # last reference is cyclic garbage is still allocated when
        # empty_cache() runs, so its block stays checked out and the call
        # returns less than it looks like it does.
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu", torch.float32).numpy()


def _load_encoder_state_dict(snapshot: Path, dtype: torch.dtype) -> dict:
    """Read only the ``encoder.*`` tensors out of the checkpoint shards.

    Every shard is opened rather than only the ones the index lists encoder
    weights in. ``safe_open`` reads a header, and ``keys()`` no tensor data,
    so the shortcut saved nothing worth keeping a second copy of the shard
    layout for -- and that copy was the one that handled both layouts.
    """
    from safetensors.torch import safe_open

    state: dict[str, torch.Tensor] = {}
    for shard in shard_files(snapshot):
        with safe_open(shard, framework="pt") as f:
            for key in sorted(f.keys()):
                if key.startswith("encoder."):
                    state[key[len("encoder.") :]] = f.get_tensor(key).to(dtype)
    if not state:
        raise RuntimeError(f"No encoder.* weights found in {snapshot}")
    return state


def _load_null_condition_emb(snapshot: Path) -> np.ndarray:
    """Read the CFG null-condition embedding out of the DiT checkpoint.

    The checkpoint is the only place it can come from: conditioning runs
    before the DiT is loaded, which on a cold cache is before the conversion
    has happened at all. Reading the fp32 original also keeps CFG clear of
    whatever precision the DiT was converted at.

    Every shard is opened until the key turns up, rather than looked up in an
    index that a single-file checkpoint does not have. ``mx.load`` is lazy, so
    that costs a header read per shard: pulling one ``[1, 1, D]`` tensor out
    of a 1 GB shard measures at 0.000 s and no MLX allocation at all.
    """
    import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)

    for shard in shard_files(snapshot):
        weights = mx.load(str(shard))
        if NULL_COND_KEY in weights:
            return np.array(weights[NULL_COND_KEY].astype(mx.float32))
    raise RuntimeError(
        f"{NULL_COND_KEY!r} missing from {snapshot}; CFG cannot be built."
    )
