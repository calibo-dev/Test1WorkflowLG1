# Deprecated node, removed in favor of primary_writer_node_builder.py
# This file is now obsolete and can be deleted.

# from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from latest_ai_development.config.settings import get_settings
from latest_ai_development.graph.state import State
from latest_ai_development.llm.llm_factory import LLMFactory
from latest_ai_development.logging_utils import summarize_tools
from latest_ai_development.prompts.prompt_builder import build_reporting_prompt
from latest_ai_development.tools.tool_registry import get_tools_for_agent

logger = logging.getLogger(__name__)


def _get_reporting_agent_role(state: State) -> str:
    for agent in state.get("agent_plan", []):
        if agent.get("id") == "reporting":
            return agent.get("role", "")
    return ""


def _build_reporting_message(state: State) -> str:
    role = _get_reporting_agent_role(state).strip()
    topic = state["topic"]
    research_result = state["research_result"]

    if not role:
        return (
            f"Topic:\n{topic}\n\n"
            f"Research Result:\n{research_result}"
        )

    return (
        f"Agent Role:\n{role}\n\n"
        f"Topic:\n{topic}\n\n"
        f"Research Result:\n{research_result}"
    )

class ReportingNodeBuilder:

    @staticmethod
    def build() -> Callable[[State], dict[str, Any]]:
        logger.info("Building Reporting Node...")
        provider = get_settings().provider.lower()

        logger.info("Creating LLM for Reporting Node with provider=%s...", provider)
        llm = LLMFactory().create_llm(provider=provider)

        logger.info("Building Reporting Prompt...")
        prompt = build_reporting_prompt()

        def reporting_node(state: State, /) -> dict[str, Any]:
            logger.info("Reporting Node started")
            tools = get_tools_for_agent(state.get("agent_plan", []), agent_id="reporting")
            logger.debug("Reporting Node tools=%s", summarize_tools(tools))
            logger.debug(
                "Reporting Node payload | topic_length=%s | research_result_present=%s",
                len(str(state.get("topic", "")).strip()),
                bool(str(state.get("research_result", "") or "").strip()),
            )
            user_message = _build_reporting_message(state)

            logger.info("Creating Reporting Agent...")
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=prompt,
            )
            agent_input = {"messages": [HumanMessage(content=user_message)]}
            logger.debug("Reporting Node invoking agent")
            result = agent.invoke(cast(Any, agent_input))

            report = result["messages"][-1].content

            logger.info("Reporting Node completed successfully")
            logger.debug("Reporting Node output length=%s", len(str(report)))

            return {
                **state,
                "report": report,
            }

        return reporting_node
