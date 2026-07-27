from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

REGISTERED_TOOLS: dict[str, Any] = {}


def get_tools_by_names(tool_names: Sequence[str] | None) -> list[Any]:
    resolved_tools: list[Any] = []

    for tool_name in tool_names or []:
        if not isinstance(tool_name, str):
            continue

        tool = REGISTERED_TOOLS.get(tool_name.strip())
        if tool is not None:
            resolved_tools.append(tool)

    return resolved_tools


def get_tools_for_agent(
    agent_plan: Sequence[Mapping[str, Any]] | None,
    *,
    agent_id: str | None = None,
    agent_name: str | None = None,
) -> list[Any]:
    if not agent_plan:
        return []

    for raw_agent in agent_plan:
        if not isinstance(raw_agent, Mapping):
            continue

        matches_id = bool(agent_id) and raw_agent.get("id") == agent_id
        matches_name = bool(agent_name) and raw_agent.get("name") == agent_name
        if not matches_id and not matches_name:
            continue

        tool_names = raw_agent.get("tools")
        if not isinstance(tool_names, Sequence) or isinstance(tool_names, (str, bytes)):
            return []

        return get_tools_by_names(
            [str(tool_name).strip() for tool_name in tool_names if str(tool_name).strip()]
        )

    return []
