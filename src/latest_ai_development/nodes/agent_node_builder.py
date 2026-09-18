from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast, get_type_hints

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, create_model

from latest_ai_development.config.settings import (
    BASE_DIR,
    get_agents_config,
)
from latest_ai_development.graph.state import State
from latest_ai_development.llm.llm_factory import LLMFactory, get_fallback_llms
from latest_ai_development.llm.model_attempt_logging import (
    LoggedModelFallbackMiddleware,
    ModelAttemptLoggingMiddleware,
    get_model_attempts,
)
from latest_ai_development.llm.model_settings import (
    get_fallback_model_settings,
    get_model_settings,
)
from latest_ai_development.logging_utils import summarize_tools
from latest_ai_development.tools.tool_registry import get_tools_for_node

logger = logging.getLogger(__name__)


def _get_agent_config(agent_id: str) -> dict[str, Any]:
    agents = get_agents_config().get("agents", {})
    agent_config = agents.get(agent_id) if isinstance(agents, Mapping) else None
    if not isinstance(agent_config, dict):
        raise ValueError(f"Agent '{agent_id}' is not defined in agents.yaml")
    return agent_config


def _load_prompt(prompt_value: str) -> str:
    """Return an inline prompt, while retaining legacy prompt-file support."""
    value = prompt_value.strip()
    if not value:
        raise ValueError("Agent prompt must not be empty")

    looks_like_file = "\n" not in value and (
        value.startswith("prompts/") or value.endswith((".md", ".txt"))
    )
    if not looks_like_file:
        return value

    prompt_path = Path(value)
    if not prompt_path.is_absolute():
        prompt_path = BASE_DIR / "src" / "latest_ai_development" / prompt_path
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Agent prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _build_user_message(state: Mapping[str, Any], fields: Sequence[str]) -> str:
    sections = []
    for field in fields:
        label = field.replace("_", " ").title()
        sections.append(f"{label}:\n{state.get(field, '')}")
    return "\n\n".join(sections)


def _extract_content(result: Any) -> Any:
    if isinstance(result, Mapping) and result.get("messages"):
        return getattr(result["messages"][-1], "content", result["messages"][-1])
    return getattr(result, "content", result)


def _build_response_format(
    agent_id: str, agent_config: Mapping[str, Any]
) -> type[BaseModel] | None:
    output_mapping = agent_config.get("output_mapping")
    if not output_mapping:
        return None
    if not isinstance(output_mapping, Mapping):
        raise ValueError("Agent output_mapping must be a mapping")
    state_types = get_type_hints(State, include_extras=True)
    fields: dict[str, Any] = {
        str(output_field): (state_types.get(str(state_field), str), ...)
        for output_field, state_field in output_mapping.items()
    }
    model_name = "".join(part.title() for part in agent_id.split("_")) + "Response"
    return create_model(model_name, **fields)


def _build_state_updates(result: Any, agent_config: Mapping[str, Any]) -> dict[str, Any]:
    output_mapping = agent_config.get("output_mapping")
    if not output_mapping:
        return {str(agent_config["output_field"]): _extract_content(result)}
    if not isinstance(output_mapping, Mapping):
        raise ValueError("Agent output_mapping must be a mapping")

    structured_content = result.get("structured_response") if isinstance(result, Mapping) else None
    if isinstance(structured_content, BaseModel):
        structured_content = structured_content.model_dump()
    if not isinstance(structured_content, Mapping):
        raise ValueError("Agent did not return its configured structured response")
    missing_fields = [field for field in output_mapping if field not in structured_content]
    if missing_fields:
        raise ValueError(f"Agent output is missing configured fields: {missing_fields}")
    return {
        str(state_field): structured_content[output_field]
        for output_field, state_field in output_mapping.items()
    }


def _build_middleware(agent_id: str, provider: str, model: str) -> list[Any]:
    primary_settings = get_model_settings(provider, model_name=model)
    fallback_settings = get_fallback_model_settings(provider)
    primary_attempt, fallback_attempts = get_model_attempts(
        agent_id,
        primary_settings,
        fallback_settings,
    )
    fallback_llms = get_fallback_llms(provider)
    if fallback_llms:
        return [
            LoggedModelFallbackMiddleware(
                primary_attempt,
                fallback_attempts,
                fallback_llms[0],
                *fallback_llms[1:],
            )
        ]
    return [ModelAttemptLoggingMiddleware(primary_attempt)]


class AgentNodeBuilder:
    """Build an agent node entirely from agents.yaml and graph_config.yaml."""

    @staticmethod
    def build(
        *,
        node_id: str,
        agent_id: str,
        execution_mode: str = "async",
    ) -> Callable[[State], Any]:
        agent_config = _get_agent_config(agent_id)
        primary_settings = get_model_settings(
            agent_config.get("provider"),
            model_name=agent_config.get("model"),
        )
        provider = primary_settings.provider
        model = primary_settings.model
        llm = LLMFactory().create(primary_settings)
        middleware = _build_middleware(agent_id, provider, model)
        prompt = _load_prompt(str(agent_config["prompt"]))
        role = str(agent_config.get("role", "")).strip()
        system_prompt = f"Role: {role}\n\n{prompt}" if role else prompt
        reads_state = [str(field) for field in agent_config.get("reads_state", [])]
        tools = get_tools_for_node(node_id)
        response_format = _build_response_format(agent_id, agent_config)
        agent_kwargs: dict[str, Any] = {
            "model": llm,
            "tools": tools,
            "system_prompt": system_prompt,
            "middleware": middleware,
        }
        if response_format is not None:
            agent_kwargs["response_format"] = response_format
        agent = create_agent(**agent_kwargs)

        def agent_node_sync(state: State, /) -> dict[str, Any]:
            logger.info(
                "Invoking configured agent | node_id=%s | agent_id=%s | tools=%s",
                node_id,
                agent_id,
                summarize_tools(tools),
            )
            result = agent.invoke(
                cast(
                    Any,
                    {"messages": [HumanMessage(content=_build_user_message(state, reads_state))]},
                )
            )
            return _build_state_updates(result, agent_config)

        async def agent_node_async(state: State, /) -> dict[str, Any]:
            logger.info(
                "Invoking configured agent | node_id=%s | agent_id=%s | tools=%s",
                node_id,
                agent_id,
                summarize_tools(tools),
            )
            result = await agent.ainvoke(
                cast(
                    Any,
                    {"messages": [HumanMessage(content=_build_user_message(state, reads_state))]},
                )
            )
            return _build_state_updates(result, agent_config)

        if execution_mode == "sync":
            return agent_node_sync
        if execution_mode == "async":
            return agent_node_async
        raise ValueError(
            f"Agent node '{node_id}' has unsupported execution_mode '{execution_mode}'"
        )
