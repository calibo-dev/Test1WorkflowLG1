from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from latest_ai_development.graph.state import State
from latest_ai_development.llm.llm_factory import LLMFactory
from latest_ai_development.prompts.prompt_builder import build_primary_writer_prompt

logger = logging.getLogger(__name__)


class PrimaryWriterNodeBuilder:

    @staticmethod
    def build() -> Callable[[State], dict[str, Any]]:
        logger.info("Building Primary Writer Node...")
        provider = os.getenv("PROVIDER", "ollama").lower()

        logger.info("Creating LLM for Primary Writer Node with provider=%s...", provider)
        llm = LLMFactory().create_llm(provider=provider)

        logger.info("Building Primary Writer Prompt...")
        system_prompt = build_primary_writer_prompt()

        logger.info("Creating Primary Writer Agent...")
        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=system_prompt,
        )

        def primary_writer_node(state: State, /) -> dict[str, Any]:
            logger.info(f"Primary Writer Node received topic: {state['topic']}")

            user_message = f"Topic:\n{state['topic']}\n\nResearch Notes:\n{state['research_notes']}"
            agent_input = {"messages": [HumanMessage(content=user_message)]}
            result = agent.invoke(cast(Any, agent_input))

            if isinstance(result, dict) and "messages" in result:
                article = result["messages"][-1].content
            else:
                article = getattr(result, "content", str(result))

            logger.info(f"Primary Writer Node produced article: {article}")

            return {
                **state,
                "article": article,
            }

        return primary_writer_node
