from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from latest_ai_development.graph.state import State
from latest_ai_development.llm.llm_factory import LLMFactory
from latest_ai_development.prompts.prompt_builder import build_primary_researcher_prompt

logger = logging.getLogger(__name__)


class PrimaryResearcherNodeBuilder:

    @staticmethod
    def build() -> Callable[[State], dict[str, Any]]:
        logger.info("Building Primary Researcher Node...")
        provider = os.getenv("PROVIDER", "ollama").lower()

        logger.info("Creating LLM for Primary Researcher Node with provider=%s...", provider)
        llm = LLMFactory().create_llm(provider=provider)

        logger.info("Building Primary Researcher Prompt...")
        system_prompt = build_primary_researcher_prompt()

        logger.info("Creating Primary Researcher Agent...")
        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=system_prompt,
        )

        def primary_researcher_node(state: State, /) -> dict[str, Any]:
            logger.info(f"Primary Researcher Node received topic: {state['topic']}")

            agent_input = {"messages": [HumanMessage(content=state["topic"]) ]}
            result = agent.invoke(cast(Any, agent_input))

            if isinstance(result, dict) and "messages" in result:
                research_notes = result["messages"][-1].content
            else:
                research_notes = getattr(result, "content", str(result))

            logger.info(f"Primary Researcher Node produced research_notes: {research_notes}")

            return {
                **state,
                "research_notes": research_notes,
            }

        return primary_researcher_node
