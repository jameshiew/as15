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
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .models import LATENT_FPS

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
        self.config_json = config_json
        remote_module = config_json["auto_map"]["AutoModel"].split(".")[0]

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(dit_snapshot, trust_remote_code=True)
        self.config = config

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
        # Stored [1, 64, T]; the handler convention is [1, T, 64].
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

        text = self.tokenizer(
            text_prompt, truncation=True, max_length=256, return_tensors="pt"
        )
        lyric = self.tokenizer(
            lyrics_text, truncation=True, max_length=2048, return_tensors="pt"
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

        frames = max(1, round(duration * LATENT_FPS))
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

    def release(self) -> None:
        """Drop the torch models so their memory is free before MLX loads.

        Terminal and idempotent: the conditioner cannot be used afterwards.
        The attributes are deleted rather than set to ``None`` so that a
        use-after-release raises an ``AttributeError`` naming the missing
        model, instead of ``NoneType is not callable`` from inside ``build``.
        """
        for name in ("encoder", "text_encoder", "silence_latent"):
            self.__dict__.pop(name, None)
        if self.device.type == "mps":
            torch.mps.empty_cache()


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu", torch.float32).numpy()


def _load_encoder_state_dict(snapshot: Path, dtype: torch.dtype) -> dict:
    """Read only the ``encoder.*`` tensors out of the checkpoint shards."""
    from safetensors.torch import safe_open

    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        wanted: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            if key.startswith("encoder."):
                wanted.setdefault(shard, []).append(key)
    else:
        wanted = {"model.safetensors": None}

    state: dict[str, torch.Tensor] = {}
    for shard, keys in wanted.items():
        with safe_open(snapshot / shard, framework="pt") as f:
            for key in keys if keys is not None else f.keys():
                if not key.startswith("encoder."):
                    continue
                state[key[len("encoder.") :]] = f.get_tensor(key).to(dtype)
    if not state:
        raise RuntimeError(f"No encoder.* weights found in {snapshot}")
    return state


def _load_null_condition_emb(snapshot: Path) -> np.ndarray:
    """Read the CFG null-condition embedding from the converted MLX cache."""
    import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)

    from .convert import NULL_COND_KEY

    index = snapshot / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    shard = weight_map[NULL_COND_KEY]
    value = mx.load(str(snapshot / shard))[NULL_COND_KEY]
    return np.array(value.astype(mx.float32))
