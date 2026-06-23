from __future__ import annotations

import logging
from langchain.agents import create_agent

from latest_ai_development.llm.llm_factory import LLMFactory
from latest_ai_development.prompts.prompt_builder import build_reporting_prompt

logger = logging.getLogger(__name__)


class ReportingNodeBuilder:

    @staticmethod
    def build():
        logger.info("Building Reporting Node...")
        
        logger.info("Creating LLM for Reporting Node...")
        llm = LLMFactory().create_llm(provider="openai")

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

            result = agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            f"""
Query:
{state['query']}

Research Result:
{state['research_result']}
""",
                        )
                    ]
                }
            )

            report = result["messages"][-1].content

            logger.info("Reporting Node completed successfully")
            logger.debug(f"Generated Report: {report}")

            return {
                **state,
                "report": report,
            }

        return reporting_node