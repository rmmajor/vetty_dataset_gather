"""The teacher's tool registry: the exact tool set per autonomy, and the schema invariants supervision needs."""

from cua.pipeline.teacher.tools import (
    AUTONOMOUS_TOOLS,
    CONTROL_TOOLS,
    GUARDED_CONTROL_TOOLS,
    INPUT_TOOLS,
    NAVIGATION_TOOLS,
    OBSERVATION_TOOLS,
    THOUGHT_SCHEMA,
    TOOLS_BY_AUTONOMY,
    build_driver_mcp_server,
)

AUTONOMOUS_NAMES = [
    "click",
    "type",
    "press",
    "scroll",
    "drag",
    "hover",
    "click_xy",
    "drag_path",
    "upload",
    "select",
    "snapshot",
    "screenshot",
    "wait",
    "navigate",
    "task_complete",
]
GUARDED_NAMES = [*AUTONOMOUS_NAMES, "handoff", "confirm"]
# The four tools that are not recorded user actions, so the supervision-carrying `thought` does not apply to them.
NO_THOUGHT_NAMES = {"upload", "snapshot", "screenshot", "task_complete"}


def test_registry_holds_exactly_the_seventeen_tools_in_order():
    assert [spec.name for spec in TOOLS_BY_AUTONOMY["autonomous"]] == AUTONOMOUS_NAMES
    assert [spec.name for spec in TOOLS_BY_AUTONOMY["guarded"]] == GUARDED_NAMES
    assert len(GUARDED_NAMES) == 17 and len(set(GUARDED_NAMES)) == 17


def test_groups_partition_the_registry():
    groups = [INPUT_TOOLS, OBSERVATION_TOOLS, NAVIGATION_TOOLS, CONTROL_TOOLS, GUARDED_CONTROL_TOOLS]
    assert [spec.name for group in groups for spec in group] == GUARDED_NAMES
    assert AUTONOMOUS_TOOLS == (*INPUT_TOOLS, *OBSERVATION_TOOLS, *NAVIGATION_TOOLS, *CONTROL_TOOLS)


def test_every_recorded_action_tool_requires_a_thought():
    for spec in TOOLS_BY_AUTONOMY["guarded"]:
        required = spec.schema.get("required", [])
        properties = spec.schema.get("properties", {})
        if spec.name in NO_THOUGHT_NAMES:
            assert "thought" not in required, spec.name
            continue
        assert required[0] == "thought", spec.name
        assert properties["thought"] == THOUGHT_SCHEMA, spec.name


def test_the_server_exposes_the_guarded_tools_only_in_guarded_mode():
    for autonomy, expected in (
        ("autonomous", AUTONOMOUS_NAMES),
        ("guarded", GUARDED_NAMES),
        ("supervised", AUTONOMOUS_NAMES),
    ):
        _, allowed = build_driver_mcp_server(None, {}, "browser", autonomy)
        assert allowed == [f"mcp__browser__{name}" for name in expected], autonomy


def test_every_spec_carries_a_description_and_an_object_schema():
    for spec in TOOLS_BY_AUTONOMY["guarded"]:
        assert spec.description.strip() == spec.description and len(spec.description) > 20, spec.name
        assert spec.schema == {} or spec.schema["type"] == "object", spec.name
        assert callable(spec.handler), spec.name
