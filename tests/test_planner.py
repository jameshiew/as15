"""The 5Hz planner's prompt format, reasoning block and code constraint.

The prompt strings here are pinned against what upstream renders, because they
are the trained format: a planner given a prompt shaped differently still emits
codes, and the codes are worse rather than wrong. Nothing in this module loads
a planner -- the parts that need one are exercised through a stand-in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from as15.planner import (
    LM_INSTRUCTION,
    Planner,
    format_reasoning,
    format_user_prompt,
    normalise_time_signature,
    parse_reasoning,
    write_plan,
)

# --- the trained prompt format --------------------------------------------


def test_the_user_turn_is_the_trained_one():
    """Verbatim from upstream's ``build_formatted_prompt``.

    Both headers are always present; an instrumental's ``# Lyric`` section is
    empty rather than absent, which is the shape the planner was trained on.
    """
    assert format_user_prompt("dream pop", "[verse]\nla") == (
        "# Caption\ndream pop\n\n# Lyric\n[verse]\nla\n"
    )
    assert format_user_prompt("techno", "") == "# Caption\ntechno\n\n# Lyric\n\n"


def test_the_instruction_is_the_code_generating_one():
    """Not the DiT's mask-filling instruction, which is a different string."""
    assert LM_INSTRUCTION == (
        "Generate audio semantic tokens based on the given conditions:"
    )


# --- the reasoning block --------------------------------------------------


def test_the_reasoning_block_is_yaml_with_sorted_keys():
    block = format_reasoning(
        {"timesignature": "4", "bpm": "96", "caption": "a song", "duration": 120}
    )
    assert block == (
        "<think>\nbpm: 96\ncaption: a song\nduration: 120\ntimesignature: 4\n</think>"
    )


def test_digit_strings_become_numbers_in_the_block():
    """``bpm: 96`` is the trained form; dumping the string gives ``bpm: '96'``."""
    assert "bpm: 96\n" in format_reasoning({"bpm": "96"})
    assert "'96'" not in format_reasoning({"bpm": "96"})


def test_unset_fields_are_left_out_rather_than_written_empty():
    block = format_reasoning({"bpm": "96", "keyscale": None, "caption": ""})
    assert block == "<think>\nbpm: 96\n</think>"


def test_a_folded_caption_survives_the_round_trip():
    """A long caption is wrapped across indented continuation lines.

    That is what ``yaml.dump`` does at its default width, and what upstream
    therefore feeds the planner. A line-by-line reader keeps only the first
    line and silently truncates the caption to its first eight words.
    """
    caption = (
        "A dreamy indie rock track featuring clean arpeggiated electric guitars "
        "shimmering with a light chorus effect and spacious reverb, with a gentle "
        "female vocal leading over a steady rhythm section"
    )
    block = format_reasoning({"caption": caption, "bpm": 96})
    assert "\n  " in block, "the caption should be folded, or this pins nothing"
    assert parse_reasoning(block)["caption"] == caption


def test_a_block_the_model_left_malformed_still_yields_what_it_holds():
    """Phase 1 is unconstrained, so the block is not guaranteed to be YAML."""
    parsed = parse_reasoning(
        "<think>\nbpm: 96\nthis line is not a field\nkeyscale: C major\n: stray\n</think>"
    )
    assert parsed["bpm"] == "96"
    assert parsed["keyscale"] == "C major"


def test_only_the_fields_that_mean_something_are_kept():
    """``genres`` is generated upstream and dropped before conditioning."""
    parsed = parse_reasoning("<think>\nbpm: 96\ngenres: folk\nnonsense: 1\n</think>")
    assert set(parsed) == {"bpm"}


def test_reasoning_is_read_without_the_tags_too():
    """The model is stopped *at* ``</think>``, so the close tag may be absent."""
    assert parse_reasoning("<think>\nbpm: 96\n")["bpm"] == "96"


@pytest.mark.parametrize(
    ("given", "expected"), [("4/4", "4"), ("6/8", "6"), ("3", "3"), (" 4 ", "4")]
)
def test_a_time_signature_is_reduced_to_its_numerator(given, expected):
    """``4/4`` is the LM's spelling and ``4`` is what the metas block takes."""
    assert normalise_time_signature(given) == expected


# --- what the caller fixes wins -------------------------------------------


class StubPlanner:
    """A planner that records what it was asked, and answers fixed codes."""

    key = "stub"

    def __init__(self, reasoning: str):
        self._reasoning = reasoning
        self.reasoning_used: str | None = None
        self.count: int | None = None

    def reason(self, _caption, _lyrics, _seed):
        return self._reasoning

    def codes(self, _caption, _lyrics, reasoning, count, _seed, progress=True):
        self.reasoning_used = reasoning
        self.count = count
        return tuple(range(count))

    def release(self):
        pass


@pytest.fixture
def planned(monkeypatch):
    """Run ``write_plan`` against a stand-in, and hand back both sides."""

    def run(reasoning, **request_kwargs):
        from helpers import resolved

        stub = StubPlanner(reasoning)
        monkeypatch.setattr("as15.planner.Planner", lambda _path, _key: stub)
        # A registered planner, because resolution rejects a name it does not
        # know -- which is the point of registering them. Nothing is loaded:
        # the stand-in replaces the class before ``write_plan`` builds one.
        request = resolved(planner="0.6b", **request_kwargs)
        # The path is never opened: the stand-in above replaces the class that
        # would have loaded from it.
        plan = write_plan(
            Path("unused"), request, request.latent_frames, seed=1, progress=False
        )
        return stub, plan

    return run


LM_BLOCK = (
    "<think>\nbpm: 150\ncaption: whatever\nduration: 300\nkeyscale: F minor\n</think>"
)


def test_what_the_caller_fixed_overrides_what_the_planner_imagined(planned):
    """The plan has to be for the song being generated, not the LM's preference.

    The planner is consulted about what was left open and overruled about what
    was not -- otherwise asking for 96 bpm and getting a plan built at 150 is a
    disagreement nothing in the output explains.
    """
    stub, _plan = planned(LM_BLOCK, bpm=96, key_scale="C major", duration=20.0)

    assert stub.reasoning_used is not None
    assert "bpm: 96" in stub.reasoning_used
    assert "keyscale: C major" in stub.reasoning_used
    assert "duration: 20" in stub.reasoning_used


def test_the_planner_fills_in_what_the_caller_left_open(planned):
    stub, _plan = planned(LM_BLOCK, duration=20.0)

    assert stub.reasoning_used is not None
    assert "bpm: 150" in stub.reasoning_used
    assert "keyscale: F minor" in stub.reasoning_used


def test_the_duration_is_always_the_callers(planned):
    """It is already a settled frame count by this point; the LM does not get a say."""
    stub, _plan = planned(LM_BLOCK, duration=20.0)
    assert stub.reasoning_used is not None
    assert "duration: 20" in stub.reasoning_used
    assert "duration: 300" not in stub.reasoning_used


def test_the_plan_is_long_enough_for_the_song(planned):
    """20 s is 500 frames is 100 codes."""
    stub, plan = planned(LM_BLOCK, duration=20.0)
    assert stub.count == 100
    assert len(plan.codes) == 100


def test_a_planner_with_no_audio_code_tokens_is_rejected(monkeypatch, tmp_path):
    """A plain Qwen3 would happily emit prose where the codes should be."""

    class PlainTokenizer:
        eos_token_id = 1

        def get_vocab(self):
            return {"hello": 0, "<|im_end|>": 1}

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        classmethod(lambda _cls, _path, **_kw: PlainTokenizer()),
    )
    monkeypatch.setattr(
        "as15.mlx.lm.MLXQwen3LM.from_snapshot",
        classmethod(lambda _cls, _path: object()),
    )
    with pytest.raises(RuntimeError, match="not a 5 Hz planner"):
        Planner(tmp_path, "stub")
