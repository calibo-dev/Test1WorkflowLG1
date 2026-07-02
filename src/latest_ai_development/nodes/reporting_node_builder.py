from __future__ import annotations

import logging
import os
from pathlib import Path
from collections.abc import Callable
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from latest_ai_development.graph.state import State
from latest_ai_development.llm.llm_factory import LLMFactory
from latest_ai_development.prompts.prompt_builder import build_reporting_prompt

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


def _write_report_markdown(report: str) -> None:
    output_path = Path("./data/report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


class ReportingNodeBuilder:

    @staticmethod
    def build() -> Callable[[State], dict[str, Any]]:
        logger.info("Building Reporting Node...")
        provider = os.getenv("PROVIDER", "ollama").lower()
        
        logger.info("Creating LLM for Reporting Node with provider=%s...", provider)
        llm = LLMFactory().create_llm(provider=provider)

        logger.info("Building Reporting Prompt...")
        prompt = build_reporting_prompt()

        logger.info("Creating Reporting Agent...")
        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=prompt,
        )

        def reporting_node(state: State, /) -> dict[str, Any]:
            logger.info(f"Reporting Node received topic: {state.get('topic')}")
            user_message = _build_reporting_message(state)

            agent_input = {"messages": [HumanMessage(content=user_message)]}
            result = agent.invoke(cast(Any, agent_input))

            report = result["messages"][-1].content
            _write_report_markdown(report)

            logger.info("Reporting Node completed successfully")
            logger.debug(f"Generated Report: {report}")

            return {
                **state,
                "report": report,
            }

        return reporting_node
