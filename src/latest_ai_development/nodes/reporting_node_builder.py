from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from latest_ai_development.llm.llm_factory import LLMFactory
from latest_ai_development.prompts.prompt_builder import build_reporting_prompt

logger = logging.getLogger(__name__)


def _write_report_markdown(report: str) -> None:
    output_path = Path("./data/report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


class ReportingNodeBuilder:

    @staticmethod
    def build():
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

        def reporting_node(state: dict) -> dict:
            logger.info(f"Reporting Node received query: {state.get('query')}")
            user_message = (
                f"Query:\n{state['query']}\n\n"
                f"Research Result:\n{state['research_result']}"
            )

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
