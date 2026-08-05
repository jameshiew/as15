"""The 5 Hz planner: an LM that writes the audio-code plan a song is rendered from.

The DiT can be conditioned on a *plan* -- one FSQ code per 200 ms, sketching
how the track moves -- instead of on silence. This module is where a plan comes
from when nobody supplies one: a Qwen3 fine-tune whose vocabulary ends in 65535
``<|audio_code_N|>`` tokens, run in MLX (:mod:`as15.mlx.lm`).

It runs in the two phases upstream uses, against the same trained prompt format:

1. **Reasoning.** The LM is given the caption and lyrics and writes a ``<think>``
   block of YAML settling bpm, key, time signature, duration and a caption. Any
   of those the caller already fixed are overwritten afterwards, so the LM is
   only ever consulted about what was left open.
2. **Codes.** The same prompt, with that reasoning block appended, continued
   into a run of audio-code tokens. Nothing but a code may be drawn, and the run
   ends at exactly the length the song needs.

Upstream reaches phase 2 through a 2300-line constrained-decoding FSM, most of
which exists to force the *reasoning* block into a valid YAML shape field by
field. That machinery is not reproduced here: phase 1 is left unconstrained and
its output parsed leniently, because the model is fine-tuned to emit exactly
this block and a malformed one costs a re-read of four values that the caller
can also just supply. Phase 2's constraint *is* reproduced, because there the
constraint is the contract -- it is what makes the plan the right length and
every token in it a real code.

Memory is staged: the planner is loaded, run and released before the condition
encoder is built, which is itself released before the DiT loads. The 4B is 8.4
GB of weights, so overlapping any two of those stages is what a 32 GB machine
cannot do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .codes import CODEBOOK_SIZE, codes_for_frames

# Verbatim from acestep.constants. Part of the trained prompt format.
LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"

# The reasoning block's delimiters, and the keys it is read for. Upstream also
# emits ``genres`` but drops it before conditioning, so it is parsed and
# ignored here too rather than being mistaken for a caption.
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
COT_KEYS = ("bpm", "caption", "duration", "keyscale", "language", "timesignature")

# What upstream draws with: temperature 0.85, nucleus 0.9, no top-k, no
# repetition penalty. Reproduced rather than re-tuned -- these are the settings
# the checkpoint's own defaults name, and a plan drawn at other settings is not
# obviously better, only different.
PLANNER_TEMPERATURE = 0.85
PLANNER_TOP_P = 0.9

# Classifier-free guidance over the *codes*, at upstream's default. The
# unconditional branch is the prompt below: the trained CFG-dropout format,
# which is the literal string the planner saw in place of a caption and lyric
# sheet during training, with an empty reasoning block. Guiding away from it is
# what makes the codes follow the song rather than the model's general prior.
#
# Upstream forces guidance off for the reasoning phase, and so does this: only
# `codes` passes an unconditional prompt.
PLANNER_GUIDANCE = 2.0
NEGATIVE_PROMPT = "NO USER INPUT"
# Two newlines inside, not one. Qwen's template renders an assistant reasoning
# block as ``<think>\n{stripped}\n</think>``, so an empty one has the inner
# blank line -- and that is the prefix the model was trained against.
EMPTY_REASONING = f"{THINK_OPEN}\n\n{THINK_CLOSE}"

# Headroom over the reasoning block, which is six short YAML lines. Upstream
# allows 500; the block does not get longer with the song.
MAX_REASONING_TOKENS = 500

_CODE_TOKEN = re.compile(r"^<\|audio_code_(\d+)\|>$")


def _one_string(value: object) -> str:
    """The single string a tokenizer call returned.

    Both ``apply_chat_template`` and ``decode`` are typed as also returning
    batches, because both take batches; these calls do not. Checking says so
    once, rather than letting a list reach the model as a prompt whose
    ``str()`` is a Python repr.
    """
    if not isinstance(value, str):
        raise RuntimeError(f"expected one string from the tokenizer, got {type(value)}")
    return value


@dataclass(frozen=True)
class Plan:
    """A written plan, and what the LM decided while writing it."""

    codes: tuple[int, ...]
    # The reasoning block as the LM left it, verbatim. Recorded rather than
    # summarised: it is the only account of what the planner thought the song
    # was, and re-deriving it from the codes is not possible.
    reasoning: str
    planner: str


def format_user_prompt(caption: str, lyrics: str) -> str:
    """The user turn, in the trained format.

    Both headers are always present, including for an instrumental, whose
    ``# Lyric`` section is simply empty -- that is the shape the planner was
    trained on.
    """
    return f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"


def format_reasoning(metas: dict[str, object]) -> str:
    """Render *metas* as the ``<think>`` block phase 2 is continued from.

    Through ``yaml.dump`` with sorted keys, which is exactly what upstream
    writes here -- including folding a long caption across indented
    continuation lines at the default width. Digit strings become integers
    first, for the same reason: ``bpm: 96`` is what the block was trained with,
    and ``bpm: '96'`` is what dumping the string would produce.
    """
    import yaml

    items = {
        key: (int(value) if isinstance(value, str) and value.isdigit() else value)
        for key, value in metas.items()
        if value is not None and value != ""
    }
    dumped = yaml.dump(items, allow_unicode=True, sort_keys=True).strip()
    return f"{THINK_OPEN}\n{dumped}\n{THINK_CLOSE}"


def parse_reasoning(text: str) -> dict[str, str]:
    """The scalar fields of a ``<think>`` block, leniently.

    Lenient because phase 1 runs unconstrained, and a block that is short a
    field or carries an extra one should still yield what it does contain.
    YAML first, because the block is YAML and a long caption arrives folded
    across indented continuation lines that a line-by-line read would truncate
    at the first one; the scan is the fallback for a block the model left
    genuinely malformed.

    Only the keys that mean something downstream are kept, so a stray sentence
    cannot become a bpm.
    """
    import yaml

    body = text
    if THINK_OPEN in body:
        body = body.split(THINK_OPEN, 1)[1]
    body = body.split(THINK_CLOSE, 1)[0]

    try:
        loaded = yaml.safe_load(body)
    except yaml.YAMLError:
        loaded = None
    if isinstance(loaded, dict):
        return {
            str(key).strip().lower(): str(value).strip()
            for key, value in loaded.items()
            if str(key).strip().lower() in COT_KEYS and value is not None
        }

    found: dict[str, str] = {}
    for line in body.splitlines():
        key, sep, value = line.partition(":")
        key = key.strip().lower()
        if sep and key in COT_KEYS and key not in found:
            found[key] = value.strip()
    return found


def normalise_time_signature(value: str) -> str:
    """``"4/4"`` is the LM's spelling; the metas block wants ``"4"``."""
    return value.split("/")[0].strip() if "/" in value else value.strip()


class Planner:
    """A loaded planner LM, held only for as long as a plan takes to write."""

    def __init__(self, path: Path, key: str):
        from transformers import AutoTokenizer

        from .mlx.lm import MLXQwen3LM

        self.key = key
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer is None:
            raise RuntimeError(f"No usable tokenizer in {path}")
        self.tokenizer = tokenizer
        self.model = MLXQwen3LM.from_snapshot(path)

        self.code_token_ids, self.code_for_token = _audio_code_tokens(tokenizer)
        if not self.code_token_ids:
            raise RuntimeError(
                f"{path} has no <|audio_code_N|> tokens in its vocabulary; it is "
                f"a plain Qwen3, not a 5 Hz planner."
            )
        eos = tokenizer.eos_token_id
        if eos is None:
            raise RuntimeError(f"{path} declares no end-of-sequence token.")
        self.eos_id = int(eos)

    def _prompt(self, caption: str, lyrics: str) -> str:
        return self._turns(format_user_prompt(caption, lyrics))

    def _turns(self, user_content: str) -> str:
        """The chat prefix, with the assistant turn left open.

        Open because the codes are a *continuation* of that turn: closing it
        with an end-of-turn marker is what the model reads as "the song is
        finished" rather than "the codes go here".
        """
        return _one_string(
            self.tokenizer.apply_chat_template(
                [
                    # The trailing blank line inside the system turn is part of the
                    # trained format, not stray whitespace.
                    {
                        "role": "system",
                        "content": f"# Instruction\n{LM_INSTRUCTION}\n\n",
                    },
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def _encode(self, text: str) -> list[int]:
        # ``add_special_tokens=False``: the chat template has already written
        # every marker the model expects, and a second BOS shifts every
        # position the model was trained to read.
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def reason(self, caption: str, lyrics: str, seed: int | None) -> str:
        """Phase 1: let the LM settle the metadata it was not given."""
        from .mlx.lm import SamplingParams, generate

        prompt = self._prompt(caption, lyrics)
        close_ids = frozenset(self._encode(THINK_CLOSE))
        drawn = list(
            generate(
                self.model,
                self._encode(prompt),
                MAX_REASONING_TOKENS,
                SamplingParams(temperature=PLANNER_TEMPERATURE, top_p=PLANNER_TOP_P),
                seed=seed,
                stop_ids=frozenset({self.eos_id}) | close_ids,
            )
        )
        return _one_string(self.tokenizer.decode(drawn, skip_special_tokens=False))

    def codes(
        self,
        caption: str,
        lyrics: str,
        reasoning: str,
        count: int,
        seed: int | None,
        progress: bool = True,
        guidance: float = PLANNER_GUIDANCE,
    ) -> tuple[int, ...]:
        """Phase 2: continue the prompt into exactly *count* audio codes.

        The run is constrained rather than trusted. Every non-code token is
        made unreachable, so the plan cannot contain prose, and the length is
        fixed by the caller rather than by the model stopping where it likes --
        which is what keeps the plan matched to the song instead of leaving its
        last verse conditioned on silence.

        Guided away from the trained no-input prompt, at upstream's default
        strength. That is the second forward pass per token, and it is what
        makes the plan follow this song rather than a plausible average one.
        """
        import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)
        from tqdm import tqdm

        from .mlx.lm import SamplingParams, generate

        # Both branches end with the reasoning block and the same blank line;
        # only the user turn and the block's contents differ. The separator is
        # the one Qwen's template writes between a reasoning block and what
        # follows it, so training and inference see the same prefix right
        # before the first code.
        prompt = self._prompt(caption, lyrics) + reasoning + "\n\n"
        uncond = self._turns(NEGATIVE_PROMPT) + EMPTY_REASONING + "\n\n"
        allowed = mx.array(sorted(self.code_token_ids))

        def only_codes(logits, _drawn):
            # Built by masking everything and then reopening the codes, rather
            # than by closing the rest: the vocabulary is 217k wide and only
            # 64k of it is ever legal here.
            masked = mx.full(logits.shape, -mx.inf, dtype=mx.float32)
            return mx.put_along_axis(
                masked,
                allowed[None, :],
                logits.astype(mx.float32)[:, allowed],
                axis=-1,
            )

        drawn = generate(
            self.model,
            self._encode(prompt),
            count,
            SamplingParams(temperature=PLANNER_TEMPERATURE, top_p=PLANNER_TOP_P),
            seed=seed,
            logits_processor=only_codes,
            uncond_prompt_ids=self._encode(uncond),
            guidance=guidance,
        )
        codes = [
            self.code_for_token[token]
            for token in tqdm(drawn, total=count, desc="planning", disable=not progress)
        ]
        # The mask admits only code tokens, so this cannot fire on a drawn
        # plan; it does fire if the vocabulary and the codebook ever disagree.
        if len(codes) != count:
            raise RuntimeError(
                f"the planner wrote {len(codes)} codes where {count} were asked for."
            )
        return tuple(codes)

    def release(self) -> None:
        """Drop the weights before the next stage loads its own."""
        import gc

        import mlx.core as mx  # ty: ignore[unresolved-import]  (mlx ships no stubs)

        self.__dict__.pop("model", None)
        gc.collect()
        mx.clear_cache()


def _audio_code_tokens(tokenizer) -> tuple[list[int], dict[int, int]]:
    """The vocabulary's audio-code tokens, and what code each one stands for.

    Read out of the tokenizer rather than computed from a base id, even though
    the ids are contiguous in every published planner: the mapping is what
    makes a drawn token a codebook index, and deriving it from an offset that
    a future checkpoint moved would produce a plan of plausible wrong codes.

    Tokens above the codebook are left out. The planners carry 65535 of them
    over a codebook of 64000, so the top 1535 name nothing -- upstream clamps
    them into range after the fact; masking them away means they cannot be
    drawn in the first place.
    """
    token_ids: list[int] = []
    code_for_token: dict[int, int] = {}
    for token, token_id in tokenizer.get_vocab().items():
        match = _CODE_TOKEN.match(token)
        if match is None:
            continue
        code = int(match.group(1))
        if code < CODEBOOK_SIZE:
            token_ids.append(int(token_id))
            code_for_token[int(token_id)] = code
    return token_ids, code_for_token


def write_plan(
    path: Path,
    request,
    frames: int,
    seed: int | None,
    progress: bool = True,
) -> Plan:
    """Load the planner at *path*, write a plan for *request*, and release it.

    *request* is a :class:`~as15.pipeline.ResolvedGenerationRequest`: the plan
    has to be for the song that will actually be generated, and the frame count
    is what decides how many codes that is.
    """
    planner = Planner(path, request.planner or "")
    try:
        reasoning = planner.reason(request.style_prompt, request.lyrics, seed)
        metas = parse_reasoning(reasoning)

        # Everything the caller fixed wins over what the LM imagined, so the
        # plan is written against the song that is being generated rather than
        # the one the planner would have preferred. The duration is always the
        # caller's -- it is already a settled frame count by this point.
        metas["duration"] = str(request.metas_duration)
        if request.bpm is not None:
            metas["bpm"] = str(request.bpm)
        if request.key_scale is not None:
            metas["keyscale"] = str(request.key_scale)
        if request.time_signature is not None:
            metas["timesignature"] = str(request.time_signature)
        metas["language"] = request.language
        metas.setdefault("caption", request.style_prompt)
        if "timesignature" in metas:
            metas["timesignature"] = normalise_time_signature(metas["timesignature"])

        settled = format_reasoning(dict(metas))
        codes = planner.codes(
            request.style_prompt,
            request.lyrics,
            settled,
            codes_for_frames(frames),
            seed,
            progress=progress,
        )
        return Plan(codes=codes, reasoning=settled, planner=planner.key)
    finally:
        planner.release()
