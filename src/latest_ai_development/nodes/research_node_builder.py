# Deprecated node, removed in favor of primary_researcher_node_builder.py
# This file is now obsolete and can be deleted.

# from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from latest_ai_development.graph.state import State
from latest_ai_development.llm.llm_factory import LLMFactory
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
        provider = os.getenv("PROVIDER", "ollama").lower()

        logger.info("Creating LLM for Research Node with provider=%s...", provider)
        llm = LLMFactory().create_llm(provider=provider)

        logger.info("Building Research Prompt...")
        system_prompt = build_research_prompt()

        logger.info("Creating Research Agent...")
        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=system_prompt,
        )

        def research_node(state: State, /) -> dict[str, Any]:
            logger.info(f"Research Node received topic: {state['topic']}")

            agent_input = {"messages": [HumanMessage(content=_build_research_message(state))]}
            result = agent.invoke(cast(Any, agent_input))

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