"""The orchestrator's STAGES registry stays consistent with the schemas state machine it dispatches over."""

import dataclasses
import inspect

from cua.pipeline import schemas
from cua.pipeline.orchestrator import (
    DEFAULT_STAGE_COST,
    DRAIN_STAGES,
    STAGE6_COST_BY_MODE,
    STAGE_COST_ESTIMATES,
    STAGES,
    StageSpec,
    stage_cost,
    stage_workers,
)


def test_registry_covers_every_stage():
    assert sorted(STAGES) == [1, 2, 3, 4, 5, 6, 7]
    assert sorted(STAGES) == sorted(schemas.STAGE_TO_PASSED_STATUS)
    for number, spec in STAGES.items():
        assert spec.number == number
        assert spec.name and isinstance(spec.name, str)
        assert inspect.iscoroutinefunction(spec.run)


def test_stage_names_are_unique():
    names = [spec.name for spec in STAGES.values()]
    assert len(set(names)) == len(names)


def test_every_stage_but_discovery_consumes_an_input_status():
    """Stage 1 seeds the funnel from GitHub, so it is the only stage with no input status to select on."""
    assert sorted(n for n in STAGES if n != 1) == sorted(schemas.STAGE_TO_INPUT_STATUS)


def test_rejecting_stages_are_a_subset_with_finalize_excluded():
    assert set(schemas.STAGE_TO_REJECTED_STATUS) <= set(STAGES)
    assert 7 not in schemas.STAGE_TO_REJECTED_STATUS
    assert 1 not in schemas.STAGE_TO_REJECTED_STATUS


def test_drain_stages_are_the_llm_funnel_in_order():
    assert DRAIN_STAGES == (3, 4, 5, 6, 7)
    assert DRAIN_STAGES == tuple(sorted(DRAIN_STAGES))
    for stage in DRAIN_STAGES:
        assert stage in schemas.STAGE_TO_INPUT_STATUS


def test_cost_estimates_cover_the_drained_stages():
    assert STAGE_COST_ESTIMATES == {3: 0.15, 4: 0.20, 5: 0.02, 7: 0.50}
    for stage, cost in STAGE_COST_ESTIMATES.items():
        assert stage_cost(stage) == cost
    assert stage_cost(6) == STAGE6_COST_BY_MODE[schemas.CAPTURE_MODE]
    assert stage_cost(99) == DEFAULT_STAGE_COST


def test_stage_6_cost_follows_the_capture_mode():
    original = schemas.CAPTURE_MODE
    try:
        schemas.CAPTURE_MODE = "replay"
        assert stage_cost(6) == STAGE6_COST_BY_MODE["replay"]
        schemas.CAPTURE_MODE = "agent"
        assert stage_cost(6) == STAGE6_COST_BY_MODE["agent"]
        schemas.CAPTURE_MODE = "not-a-mode"
        assert stage_cost(6) == STAGE6_COST_BY_MODE["agent"]
    finally:
        schemas.CAPTURE_MODE = original


def test_container_holding_stages_cap_their_workers():
    for stage in (4, 5, 6):
        assert STAGES[stage].sandboxed
        assert STAGES[stage].workers_cap is not None
    for stage in (1, 2, 3):
        assert not STAGES[stage].sandboxed
    assert not STAGES[7].sandboxed
    assert STAGES[7].workers_cap == 3


def test_workers_never_exceed_the_stage_cap():
    """Sandboxing is cleared first: the Docker clamp needs a live daemon, and the flags themselves are pinned above."""
    for spec in STAGES.values():
        capped = stage_workers(dataclasses.replace(spec, sandboxed=False), 99)
        assert capped == (spec.workers_cap if spec.workers_cap is not None else 99)


def test_uncapped_stage_passes_the_requested_workers_through():
    spec = StageSpec(0, "probe", STAGES[3].run)
    assert stage_workers(spec, 7) == 7
