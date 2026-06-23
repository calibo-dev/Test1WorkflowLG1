from __future__ import annotations

import logging
from langchain.agents import create_agent

from latest_ai_development.llm.llm_factory import LLMFactory
from latest_ai_development.prompts.prompt_builder import build_research_prompt



logger = logging.getLogger(__name__)


class ResearchNodeBuilder:

    @staticmethod
    def build():
        logger.info("Building Research Node...")

        logger.info("Creating LLM for Research Node...")
        llm = LLMFactory().create_llm(provider="openai")

        logger.info("Building Research Prompt...")
        system_prompt = build_research_prompt()

        logger.info("Creating Research Agent...")
        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=system_prompt,
        )

        def research_node(state: dict) -> dict:
            logger.info(f"Research Node received query: {state['query']}")

            result = agent.invoke({
                "messages": [
                    ("user", state["query"])
                ]
            })

            if isinstance(result, dict) and "messages" in result:
                research_result = result["messages"][-1].content
            else:
                research_result = getattr(result, "content", str(result))

            logger.info(f"Research Node produced result: {research_result}")

            return {
                **state,
                "research_result": research_result,
            }

        return research_node