from __future__ import annotations

from latest_ai_development.tools import tool_registry


def test_get_tools_for_agent_resolves_registered_tools_by_agent_id():
    fake_tool = object()

    original_registry = tool_registry.REGISTERED_TOOLS.copy()
    tool_registry.REGISTERED_TOOLS.clear()
    tool_registry.REGISTERED_TOOLS["Sales Data Agentic Tools"] = fake_tool

    try:
        resolved_tools = tool_registry.get_tools_for_agent(
            [
                {
                    "id": "research",
                    "name": "Sales Data Query Agent",
                    "tools": ["Sales Data Agentic Tools"],
                }
            ],
            agent_id="research",
        )
    finally:
        tool_registry.REGISTERED_TOOLS.clear()
        tool_registry.REGISTERED_TOOLS.update(original_registry)

    assert resolved_tools == [fake_tool]


def test_get_tools_for_agent_ignores_unregistered_tools():
    resolved_tools = tool_registry.get_tools_for_agent(
        [
            {
                "id": "research",
                "name": "Sales Data Query Agent",
                "tools": ["Unknown Tool"],
            }
        ],
        agent_name="Sales Data Query Agent",
    )

    assert resolved_tools == []
