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
from latest_ai_development.logging_utils import summarize_tools
from latest_ai_development.tools.tool_registry import get_tools_for_agent

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

        def primary_writer_node(state: State, /) -> dict[str, Any]:
            logger.info(f"Primary Writer Node received topic: {state['topic']}")
            tools = get_tools_for_agent(state.get("agent_plan", []), agent_id="reporting")
            logger.debug("Primary Writer Node tools=%s", summarize_tools(tools))

            user_message = f"Topic:\n{state['topic']}\n\nResearch Notes:\n{state['research_notes']}"
            logger.info("Creating Primary Writer Agent...")
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
            )
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
