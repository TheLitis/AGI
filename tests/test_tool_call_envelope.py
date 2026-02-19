from __future__ import annotations

from interface_adapters import ToolCallEnvelope, int_action_to_tool_call, tool_call_to_int_action


def test_int_action_roundtrip_via_tool_call_envelope():
    call = int_action_to_tool_call(3)
    assert isinstance(call, ToolCallEnvelope)
    assert call.kind == "primitive"
    assert call.primitive_id == 3
    action = tool_call_to_int_action(call)
    assert action == 3


def test_tool_call_mapping_to_int_action():
    action = tool_call_to_int_action(
        {
            "kind": "tool_call",
            "tool_name": "repo.run_tests",
            "tool_args": {"mode": "quick"},
        },
        tool_name_to_action={"repo.run_tests": 3},
        default_action=0,
    )
    assert action == 3

