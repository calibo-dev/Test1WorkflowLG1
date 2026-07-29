# Deprecated node, removed in favor of primary_researcher_node_builder.py
# This file is now obsolete and can be deleted.

# from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from latest_ai_development.config.settings import get_settings
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
from latest_ai_development.prompts.prompt_builder import build_research_prompt



logger = logging.getLogger(__name__)


def _get_research_agent_role(state: State) -> str:
    for agent in state.get("agent_plan", []):
        if agent.get("id") == "research":
            return agent.get("role", "")
    return ""


def _build_research_message(state: State) -> str:
    role = _get_research_agent_role(state).strip()
    topic = state["topic"]

    if not role:
        return topic

    return f"Agent Role:\n{role}\n\nTopic:\n{topic}"


class ResearchNodeBuilder:

    @staticmethod
    def build() -> Callable[[State], dict[str, Any]]:
        logger.info("Building Research Node...")
        provider = get_settings().provider.lower()
        tools: list[Any] = []
        logger.debug("Research Node tools=%s", summarize_tools(tools))

        logger.info("Creating LLM for Research Node with provider=%s...", provider)
        llm = LLMFactory().create_llm(provider=provider)
        primary_settings = get_model_settings(provider)
        fallback_settings = get_fallback_model_settings(provider)
        primary_attempt, fallback_attempts = get_model_attempts(
            "research",
            primary_settings,
            fallback_settings,
        )
        fallback_llms = get_fallback_llms(provider)
        middleware = []
        if fallback_llms:
            middleware.append(
                LoggedModelFallbackMiddleware(
                    primary_attempt,
                    fallback_attempts,
                    fallback_llms[0],
                    *fallback_llms[1:],
                )
            )
        else:
            middleware.append(ModelAttemptLoggingMiddleware(primary_attempt))

        logger.info("Building Research Prompt...")
        system_prompt = build_research_prompt()

        logger.info("Creating Research Agent...")
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware,
        )

        def research_node(state: State, /) -> dict[str, Any]:
            logger.info("Research Node started")
            logger.debug(
                "Research Node payload | topic_length=%s | agent_plan_count=%s",
                len(str(state.get("topic", "")).strip()),
                len(state.get("agent_plan", [])),
            )

            agent_input = {"messages": [HumanMessage(content=_build_research_message(state))]}
            logger.debug("Research Node invoking agent")
            result = agent.invoke(cast(Any, agent_input))

            if isinstance(result, dict) and "messages" in result:
                research_result = result["messages"][-1].content
            else:
                research_result = getattr(result, "content", str(result))

            logger.info("Research Node completed")
            logger.debug("Research Node output length=%s", len(str(research_result)))

            return {
                **state,
                "research_result": research_result,
            }

        return research_node
