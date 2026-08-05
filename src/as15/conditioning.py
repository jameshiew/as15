"""Text/lyric conditioning for the DiT.

This is the one stage that still runs under PyTorch. It covers the 0.61 B
``encoder.*`` subtree of the checkpoint (text projector + lyric encoder +
timbre encoder) plus the 0.6 B Qwen3 text encoder, and it runs exactly once
per generation -- the 4.17 B DiT that runs 8-50 times is pure MLX.

Using the checkpoint's own ``trust_remote_code`` modules here means the
conditioning is bit-for-bit the reference implementation rather than a
reimplementation that could silently drift.

Of the FSQ ``tokenizer``/``detokenizer`` subtrees, only the quantizer's
codebook and the detokenizer are ever built, and only when a run supplies
audio codes -- see :class:`AudioCodeDecoder`. The tokenizer's attention pooler
is the half that turns *audio* into codes, which is the cover/extract task
this package does not do.
"""

from __future__ import annotations

import copy
import gc
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from .codes import CODEBOOK_SIZE, POOL_WINDOW, check_codes
from .models import shard_files

if TYPE_CHECKING:
    from .pipeline import ResolvedGenerationRequest

# The CFG null branch's condition embedding, stored at the top level of the DiT
# checkpoint. Read here and nowhere else: the converter used to copy it into
# the MLX cache as well, where nothing ever read it -- the loader popped it
# straight back out -- so the cache carried a second, bf16-rounded copy of a
# tensor whose only consumer takes the fp32 original.
NULL_COND_KEY = "null_condition_emb"

# Verbatim from acestep.constants -- these strings are part of the trained
# prompt format, so they must not be paraphrased.
#
# Which of the two a run uses is decided by whether it has an audio-code plan,
# not by anything the caller sets. Upstream reads the same way round: supplying
# codes flips its task from ``text2music`` to ``cover``, and the task picks the
# instruction (``TASK_INSTRUCTIONS`` in acestep.constants). Conditioning a
# planned run on the mask-filling instruction would describe the job to the
# text encoder as the one it is not doing.
DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
AUDIO_CODE_DIT_INSTRUCTION = (
    "Generate audio semantic tokens based on the given conditions:"
)

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
    duration: int,
) -> str:
    """Render the ``# Metas`` block in the trained format.

    *duration* is whole seconds because that is what the block was trained in,
    and it arrives already rounded rather than being rounded here: this used to
    ``int()`` the request's duration, so a 12.9 s request told the model 12
    while the latent window was sized for 12.92 -- the pacing the lyrics were
    conditioned against was a song shorter than the one being generated.
    """
    return (
        f"- bpm: {bpm or 'N/A'}\n"
        f"- timesignature: {time_signature or 'N/A'}\n"
        f"- keyscale: {key_scale or 'N/A'}\n"
        f"- duration: {duration} seconds\n"
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


def _encoder_config(config):
    """The config the encoder-side submodules are built from.

    XL checkpoints size the condition encoder independently of the decoder, and
    the FSQ tokenizer and detokenizer are built from that same encoder-sized
    config upstream -- their ``hidden_size`` is ``encoder_hidden_size``, which
    is also what ``fsq_dim`` is. Building either from the decoder's config
    gives shapes the checkpoint's own weights will not load into.
    """
    encoder_config = copy.deepcopy(config)
    encoder_config.hidden_size = config.encoder_hidden_size
    encoder_config.intermediate_size = config.encoder_intermediate_size
    encoder_config.num_attention_heads = config.encoder_num_attention_heads
    encoder_config.num_key_value_heads = config.encoder_num_key_value_heads
    return encoder_config


class AudioCodeDecoder:
    """Turns a 5 Hz audio-code plan into the 25 Hz latent hints the DiT reads.

    The planner LM emits one FSQ index per 200 ms. Two checkpoint submodules
    turn those back into something shaped like a latent: the residual-FSQ
    codebook, which looks each index up and projects it to
    ``encoder_hidden_size``, and the detokenizer, which expands every code into
    ``pool_window_size`` frames and projects those down to the 64 acoustic
    channels. The result replaces the silence that a text-only run conditions
    on, and nothing else about the conditioning changes.

    Only ``tokenizer.quantizer.*`` is loaded, not the whole audio tokenizer:
    its attention pooler is the *encoding* direction -- audio to codes -- which
    is the cover/extract task, and skipping it leaves 105 MB of weights on
    disk. The two subtrees that are loaded come to ~105 M parameters.
    """

    def __init__(
        self, snapshot: Path, config, dtype: torch.dtype, device: torch.device
    ):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from vector_quantize_pytorch import ResidualFSQ

        self.device = device
        self.dtype = dtype
        # The plan's geometry is the runtime's contract -- a plan is parsed and
        # bounds-checked by the CLI before any checkpoint is on disk -- so a
        # checkpoint that disagrees is an error rather than a plan silently
        # read at the wrong rate or against the wrong codebook.
        self.pool_window = int(config.pool_window_size)
        if self.pool_window != POOL_WINDOW:
            raise RuntimeError(
                f"checkpoint expands each audio code into {self.pool_window} "
                f"latent frames, not {POOL_WINDOW}; plans are read at the wrong rate."
            )

        # The codebook stays float32. ResidualFSQ forces float32 through its FSQ
        # layers internally and upstream's tokenizer casts around that; feeding
        # it bf16 here would be casting into a module that casts straight back.
        quantizer = ResidualFSQ(
            dim=config.fsq_dim,
            levels=list(config.fsq_input_levels),
            num_quantizers=config.fsq_input_num_quantizers,
        )
        quantizer.load_state_dict(
            _load_subtree(snapshot, "tokenizer.quantizer.", torch.float32), strict=True
        )
        self.quantizer = quantizer.to(device).eval()
        if self.codebook_size != CODEBOOK_SIZE:
            raise RuntimeError(
                f"checkpoint codebook holds {self.codebook_size} codes, not "
                f"{CODEBOOK_SIZE}; plans were bounds-checked against the wrong one."
            )

        remote_module = json.loads((snapshot / "config.json").read_text())["auto_map"][
            "AutoModel"
        ].split(".")[0]
        detokenizer_cls = get_class_from_dynamic_module(
            f"{remote_module}.AudioTokenDetokenizer", str(snapshot)
        )
        # On CPU rather than the meta device, for the same reason the condition
        # encoder is: the rotary embedding's ``inv_freq`` is a non-persistent
        # buffer, absent from the state dict and only ever set in __init__.
        detokenizer = detokenizer_cls(config)
        # ``assign=True`` for the same reason the condition encoder uses it:
        # without it the loaded tensors are *copied into* the float32
        # parameters the constructor made, so the module keeps float32 weights
        # and meets its bfloat16 input in a matmul. On MPS that is not a
        # promotion but an abort -- MPSNDArrayMatrixMultiplication asserts that
        # destination and accumulator share a datatype, and the process dies
        # with no Python traceback to say where.
        detokenizer.load_state_dict(
            _load_subtree(snapshot, "detokenizer.", dtype), assign=True, strict=True
        )
        self.detokenizer = detokenizer.to(device).eval()
        # Said out loud, because the failure it guards is a process abort with
        # no traceback rather than an exception anyone can catch.
        loaded = next(self.detokenizer.parameters()).dtype
        if loaded != dtype:
            raise RuntimeError(
                f"detokenizer weights loaded as {loaded}, not {dtype}; its input "
                f"is {dtype} and the two meet in a matmul."
            )

    @property
    def codebook_size(self) -> int:
        """How many distinct codes the quantizer can look up.

        Read off the built codebook rather than multiplied out of the config,
        because it is the bound an index is actually checked against: the
        planner's vocabulary carries 65535 ``<|audio_code_N|>`` tokens over a
        codebook of 64000, so the tokens above the codebook exist and index
        nothing.
        """
        return int(self.quantizer.codebooks.shape[-2])

    @torch.inference_mode()
    def hints(self, codes: Sequence[int], frames: int) -> torch.Tensor:
        """The [1, *frames*, 64] latent hints *codes* stand for.

        Raises:
            ValueError: if a code is outside the codebook, or there are too few
                to cover *frames*.
        """
        # Checked again here rather than trusted from the CLI, because this is
        # the boundary where a bad index stops being data and starts being a
        # lookup -- and against the codebook that was actually loaded.
        check_codes(codes, frames, self.codebook_size, self.pool_window)

        # [B, T, num_quantizers]: the residual-FSQ lookup indexes the quantizer
        # on the last axis, and a bare [B, T] fails inside einx with a shape
        # error naming neither the caller nor the argument.
        index = torch.tensor(list(codes), dtype=torch.long, device=self.device).reshape(
            1, -1, 1
        )
        hints_5hz = self.quantizer.get_output_from_indices(index)
        hints = self.detokenizer(hints_5hz.to(self.dtype))
        # Whole codes go in, so the expansion overshoots whenever the frame
        # count is not a multiple of the pool window. Upstream crops the same
        # way, to the length of the latents being generated.
        return hints[:, :frames, :]

    def release(self) -> None:
        for name in ("quantizer", "detokenizer"):
            self.__dict__.pop(name, None)


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

        encoder_config = _encoder_config(config)
        # Kept for the audio-code decoder, which is built on demand from the
        # same encoder-sized config rather than from the decoder's.
        self.encoder_config = encoder_config

        encoder_cls = get_class_from_dynamic_module(
            f"{remote_module}.AceStepConditionEncoder", str(dit_snapshot)
        )
        # Built on CPU rather than the meta device: the rotary embeddings keep
        # `inv_freq` as a *non-persistent* buffer, so it is absent from the
        # state dict and only ever gets its value from __init__. A meta build
        # would leave it uninitialised.
        encoder = encoder_cls(encoder_config)
        encoder.load_state_dict(
            _load_subtree(dit_snapshot, "encoder.", dtype), assign=True, strict=True
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
        request: ResolvedGenerationRequest,
        instruction: str | None = None,
    ) -> Conditioning:
        """Condition on *request*, which nothing here reinterprets.

        A :class:`~as15.pipeline.ResolvedGenerationRequest` rather than the
        fields it holds, because this stage and the diffusion loop have to be
        told the same song: the length arrives as a frame count and a whole
        number of seconds that the pipeline settled once, so there is no
        duration left here to round differently from the one being generated.

        *instruction* defaults to the one the request's own shape calls for --
        see :data:`AUDIO_CODE_DIT_INSTRUCTION` -- and is an argument only so a
        caller experimenting with the trained prompt format can say so.
        """
        if instruction is None:
            instruction = (
                AUDIO_CODE_DIT_INSTRUCTION
                if request.audio_codes
                else DEFAULT_DIT_INSTRUCTION
            )
        if not instruction.endswith(":"):
            instruction += ":"

        metas = format_metas(
            request.bpm,
            request.key_scale,
            request.time_signature,
            request.metas_duration,
        )
        text_prompt = SFT_GEN_PROMPT.format(instruction, request.style_prompt, metas)
        lyrics_text = format_lyrics(request.lyrics, request.language)

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

        frames = request.latent_frames
        silence = self.silence_slice(frames)
        # A text-only run conditions the context block on silence. An
        # audio-code run replaces it with the plan's latent hints, and changes
        # nothing else: the chunk mask still says "generate every frame", and
        # the encoder hidden states above never saw the codes at all.
        chunk_masks = torch.full_like(silence, CHUNK_MASK_FULL)

        if request.audio_codes:
            src_latents = self.audio_code_decoder().hints(request.audio_codes, frames)
        else:
            src_latents = silence
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

    def audio_code_decoder(self) -> AudioCodeDecoder:
        """The FSQ codebook and detokenizer, built on first use.

        Not in ``__init__`` because a text-only run never needs them, and they
        are ~105 M parameters loaded while the 4 B DiT is still to come.
        """
        decoder = self.__dict__.get("_audio_code_decoder")
        if decoder is None:
            decoder = AudioCodeDecoder(
                self.dit_snapshot, self.encoder_config, self.dtype, self.device
            )
            self._audio_code_decoder = decoder
        return decoder

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
        decoder = self.__dict__.pop("_audio_code_decoder", None)
        if decoder is not None:
            decoder.release()
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


def _load_subtree(snapshot: Path, prefix: str, dtype: torch.dtype) -> dict:
    """Read the tensors under *prefix* out of the checkpoint shards.

    Every shard is opened rather than only the ones the index lists the subtree
    in. ``safe_open`` reads a header, and ``keys()`` no tensor data, so the
    shortcut saved nothing worth keeping a second copy of the shard layout for
    -- and that copy was the one that handled both layouts.
    """
    from safetensors.torch import safe_open

    state: dict[str, torch.Tensor] = {}
    for shard in shard_files(snapshot):
        with safe_open(shard, framework="pt") as f:
            for key in sorted(f.keys()):
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = f.get_tensor(key).to(dtype)
    if not state:
        raise RuntimeError(f"No {prefix}* weights found in {snapshot}")
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
