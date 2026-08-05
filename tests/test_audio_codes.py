"""Conditioning on an audio-code plan, and what a planned take records.

The numerics of the two checkpoint submodules a plan goes through -- the FSQ
codebook and the detokenizer -- are the checkpoint's own code, so what is
tested here is everything around them: that the plan reaches the context block
and nothing else, that the instruction changes with it, and that a take says
what it was planned from.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from as15 import conditioning, pipeline
from as15.models import MODELS
from helpers import resolved
from test_conditioning import _stub_conditioner

# Long enough for the 10 s songs below -- 250 frames is 50 codes -- and
# non-constant, so a plan read at the wrong offset is not the same plan.
PLAN = tuple(range(7, 7 + 50))


class _FakeCodeDecoder:
    """Stands in for the FSQ codebook and detokenizer.

    Answers hints that are recognisable per frame and per channel, so a plan
    that reached the wrong half of the context block, or the right half at the
    wrong offset, does not land on a plausible value.
    """

    def __init__(self):
        self.asked: tuple | None = None

    def hints(self, codes, frames):
        self.asked = (tuple(codes), frames)
        return torch.full((1, frames, 64), -0.5, dtype=torch.float32)

    def release(self):
        pass


@pytest.fixture
def planned(tmp_path):
    """A stubbed conditioner whose audio-code decoder is a stand-in too."""
    conditioner = _stub_conditioner(tmp_path)
    decoder = _FakeCodeDecoder()
    # Through __dict__, which is where the lazy builder keeps it and looks for
    # it -- so a stand-in installed here is found exactly as a real one is.
    conditioner.__dict__["_audio_code_decoder"] = decoder
    return conditioner, decoder


# --- what a plan changes, and what it does not ----------------------------


def test_a_plan_replaces_the_context_latents_and_nothing_else(planned):
    """The whole effect of a plan on the DiT is the first 64 context channels.

    Verified against the checkpoint: its decoder has no ``is_covers``
    parameter, the chunk mask is unchanged, and the encoder hidden states never
    see the codes. Anything else moving here would be this port inventing
    conditioning the model was not trained on.
    """
    conditioner, _decoder = planned
    request = resolved(duration=10.0)
    plain = conditioner.build(request)
    planned_cond = conditioner.build(resolved(duration=10.0, audio_codes=PLAN))

    src, mask = np.split(planned_cond.context_latents, 2, axis=-1)
    plain_src, plain_mask = np.split(plain.context_latents, 2, axis=-1)

    assert np.all(src == -0.5), "the plan should be the whole source half"
    assert not np.array_equal(src, plain_src)
    # The chunk mask still says "generate every frame".
    assert np.array_equal(mask, plain_mask)
    assert np.all(mask == conditioning.CHUNK_MASK_FULL)


def test_the_decoder_is_asked_for_exactly_the_frames_being_generated(planned):
    conditioner, decoder = planned
    request = resolved(duration=10.0, audio_codes=PLAN)
    conditioner.build(request)
    assert decoder.asked == (PLAN, request.latent_frames)


def test_a_planned_run_is_conditioned_on_the_cover_instruction(planned):
    """Supplying codes flips upstream's task from text2music to cover.

    The task picks the instruction, so a planned run describes the job to the
    text encoder as generating semantic tokens rather than filling a mask.
    Conditioning it on the other string is not a paraphrase; it is a prompt the
    model has not seen for this job.
    """
    conditioner, _decoder = planned
    plain = conditioner.build(resolved(duration=10.0))
    planned_cond = conditioner.build(resolved(duration=10.0, audio_codes=PLAN))

    assert conditioning.DEFAULT_DIT_INSTRUCTION in plain.text_prompt
    assert conditioning.AUDIO_CODE_DIT_INSTRUCTION in planned_cond.text_prompt
    assert conditioning.AUDIO_CODE_DIT_INSTRUCTION not in plain.text_prompt


def test_a_text_only_run_never_builds_the_code_decoder(tmp_path):
    """It is ~105 M parameters loaded while a 4 B DiT is still to come.

    ``_stub_conditioner`` leaves the real builder in place, so reaching it
    raises on the stub checkpoint rather than quietly loading nothing.
    """
    conditioner = _stub_conditioner(tmp_path)
    conditioner.build(resolved(duration=10.0))
    assert "_audio_code_decoder" not in conditioner.__dict__


# --- resolution -----------------------------------------------------------


def test_a_plan_too_short_for_the_song_is_rejected_before_anything_loads():
    """Resolution runs before snapshots are fetched, so this costs a second."""
    with pytest.raises(ValueError, match="a 120s song needs 600"):
        resolved(duration=120.0, audio_codes=(1, 2, 3))


def test_a_plan_and_a_planner_together_are_rejected():
    """One writes a plan and the other supplies one; both is not a setting."""
    with pytest.raises(ValueError, match="not both"):
        resolved(duration=10.0, audio_codes=(1,) * 50, planner="4b")


def test_an_unknown_planner_is_rejected_by_name():
    with pytest.raises(ValueError, match="Unknown planner"):
        resolved(duration=10.0, planner="7b")


def test_a_resolved_plan_is_a_tuple_whatever_it_arrived_as():
    """The resolved request is frozen, and a list in it would not be."""
    request = resolved(duration=10.0, audio_codes=[1] * 50)
    assert request.audio_codes == tuple([1] * 50)


# --- what a planned take records -----------------------------------------


def test_a_planned_take_carries_its_whole_plan():
    """So the take can be re-rendered, which is the point of keeping plans.

    A digest would leave the file describing a recipe it does not carry: the
    plan is the largest single thing separating this take from a text-only one
    and cannot be recovered from the audio.
    """
    request = resolved(duration=10.0, audio_codes=PLAN)
    tags = pipeline.describe(MODELS["xl-sft"], request)

    from as15.codes import parse_codes

    assert parse_codes(tags["AS15_AUDIO_CODES"]) == request.audio_codes
    assert tags["AS15_AUDIO_CODE_COUNT"] == str(len(PLAN))


def test_a_text_only_take_says_nothing_about_a_plan():
    """An empty plan field is not the same claim as an absent one."""
    tags = pipeline.describe(MODELS["xl-sft"], resolved(duration=10.0))
    assert "AS15_AUDIO_CODES" not in tags
    assert "AS15_AUDIO_CODE_COUNT" not in tags


def test_the_recorded_plan_resolves_back_to_the_same_take():
    """Every AS15_ tag is meant to be typeable straight back at the CLI."""
    from as15.codes import parse_codes

    request = resolved(duration=10.0, audio_codes=(9,) * 250)
    tags = pipeline.describe(MODELS["xl-sft"], request)
    again = resolved(duration=10.0, audio_codes=parse_codes(tags["AS15_AUDIO_CODES"]))
    assert again == request


def test_a_plan_longer_than_the_song_is_cropped_to_what_ran():
    """What is recorded has to be what conditioned the take.

    The conditioner only ever reads the frames it is generating, so a longer
    plan left whole would go into ``AS15_AUDIO_CODES`` as the recipe while a
    prefix of it was what actually ran -- and re-rendering from the tag would
    not be re-rendering the same take.
    """
    from as15.codes import codes_for_frames

    request = resolved(duration=10.0, audio_codes=tuple(range(500)))
    assert request.audio_codes is not None
    assert len(request.audio_codes) == codes_for_frames(request.latent_frames) == 50
    assert request.audio_codes == tuple(range(50))


def test_a_take_says_which_planner_wrote_its_plan():
    """Provenance the plan itself does not carry.

    The audio is reproducible from the codes alone, but nothing in them says
    whether a 0.6B or a 4B chose them -- which is the whole question when
    auditioning one against the other.
    """
    request = resolved(duration=10.0, audio_codes=PLAN, planned_by="4b", planner_seed=7)
    tags = pipeline.describe(MODELS["xl-sft"], request)
    assert tags["AS15_PLANNER"] == "4b"
    assert tags["AS15_PLANNER_SEED"] == "7"


def test_a_plan_that_arrived_in_a_file_names_no_planner():
    """Naming one this run did not use is a worse claim than saying nothing."""
    request = resolved(duration=10.0, audio_codes=PLAN, planner_seed=7)
    tags = pipeline.describe(MODELS["xl-sft"], request)
    assert "AS15_PLANNER" not in tags
    assert "AS15_PLANNER_SEED" not in tags
