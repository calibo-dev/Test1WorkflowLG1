from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from latest_ai_development.config.settings import get_settings
from latest_ai_development.graph.state import State
from latest_ai_development.llm.llm_factory import LLMFactory
from latest_ai_development.logging_utils import summarize_tools
from latest_ai_development.prompts.prompt_builder import build_router_prompt


logger = logging.getLogger(__name__)


class OrchestrationNodeBuilder:
    @staticmethod
    def _default_agent_plan() -> list[dict[str, Any]]:
        return [
            {
                "id": "research",
                "enabled": True,
                "role": "Research agent",
            },
            {
                "id": "reporting",
                "enabled": True,
                "role": "Reporting agent",
            },
        ]

    @staticmethod
    def _normalize_agent_plan(raw_agent_plan: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_agent_plan, list):
            return OrchestrationNodeBuilder._default_agent_plan()

        normalized_agent_plan: list[dict[str, Any]] = []
        for raw_agent in raw_agent_plan:
            if not isinstance(raw_agent, dict):
                continue

            agent_id = raw_agent.get("id")
            if agent_id not in {"research", "reporting"}:
                continue

            normalized_agent_plan.append(
                {
                    "id": agent_id,
                    "enabled": bool(raw_agent.get("enabled", True)),
                    "role": str(raw_agent.get("role", "") or ""),
                }
            )

        if not normalized_agent_plan:
            return OrchestrationNodeBuilder._default_agent_plan()

        return normalized_agent_plan

    @staticmethod
    def _parse_router_result(raw_result: Any) -> list[dict[str, Any]]:
        if isinstance(raw_result, dict) and "messages" in raw_result:
            content = raw_result["messages"][-1].content
        else:
            content = getattr(raw_result, "content", str(raw_result))

        if not isinstance(content, str):
            return OrchestrationNodeBuilder._default_agent_plan()

        normalized_content = content.strip()
        if normalized_content.startswith("```"):
            normalized_content = normalized_content.strip("`")
            if normalized_content.startswith("json"):
                normalized_content = normalized_content[4:].strip()

        try:
            parsed = json.loads(normalized_content)
        except json.JSONDecodeError:
            return OrchestrationNodeBuilder._default_agent_plan()

        return OrchestrationNodeBuilder._normalize_agent_plan(parsed.get("agent_plan"))

    @staticmethod
    def route_after_router_agent(state: State) -> str:
        if state.get("errors"):
            logger.info("Router routing to finalize_output because errors are present")
            return "finalize_output"

        enabled_agents = {
            agent.get("id")
            for agent in state.get("agent_plan", [])
            if agent.get("enabled", True)
        }
        logger.debug("Router enabled agents=%s", sorted(enabled_agents))

        if "research" in enabled_agents:
            logger.info("Router selected research")
            return "research"

        if "reporting" in enabled_agents:
            logger.info("Router selected reporting")
            return "reporting"

        logger.info("Router selected finalize_output because no enabled agents remain")
        return "finalize_output"

    @staticmethod
    def route_after_research(state: State) -> str:
        if state.get("errors"):
            logger.info("Research router routing to finalize_output because errors are present")
            return "finalize_output"

        enabled_agents = {
            agent.get("id")
            for agent in state.get("agent_plan", [])
            if agent.get("enabled", True)
        }
        logger.debug("Research router enabled agents=%s", sorted(enabled_agents))

        if "reporting" in enabled_agents:
            logger.info("Research router selected reporting")
            return "reporting"

        logger.info("Research router selected finalize_output because reporting is disabled")
        return "finalize_output"

    @staticmethod
    def build_validate_input() -> Callable[[State], dict[str, Any]]:

        def validate_input_node(state: State, /) -> dict[str, Any]:
            logger.info("Validate Input node started")

            topic = state.get("topic", "")
            errors = list(state.get("errors", []))
            logger.debug(
                "Validate Input payload | topic_length=%s | existing_error_count=%s",
                len(topic.strip()) if isinstance(topic, str) else 0,
                len(errors),
            )

            if not isinstance(topic, str) or not topic.strip():
                errors.append("topic is required")

            logger.info("Validate Input node completed with %s error(s)", len(errors))

            return {
                **state,
                "errors": errors,
            }

        return validate_input_node

    @staticmethod
    def build_router_agent() -> Callable[[State], dict[str, Any]]:
        logger.info("Building Router Agent Node...")
        provider = get_settings().provider.lower()
        tools: list[Any] = []
        logger.debug("Router Agent tools=%s", summarize_tools(tools))

        logger.info("Creating LLM for Router Agent with provider=%s...", provider)
        llm = LLMFactory().create_llm(provider=provider)

        logger.info("Building Router Prompt...")
        system_prompt = build_router_prompt()

        logger.info("Creating Router Agent...")
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

        def router_agent_node(state: State, /) -> dict[str, Any]:
            logger.info("Router Agent node started")

            agent_plan = list(state.get("agent_plan", []))
            if agent_plan:
                logger.info("Router Agent node reused existing agent plan with %s agent(s)", len(agent_plan))
                return {
                    **state,
                    "agent_plan": OrchestrationNodeBuilder._normalize_agent_plan(agent_plan),
                }

            agent_input = {"messages": [HumanMessage(content=f"Topic:\n{state['topic']}")]}
            logger.debug(
                "Router Agent invoking model | topic_length=%s",
                len(str(state.get("topic", "")).strip()),
            )
            result = agent.invoke(agent_input)
            parsed_agent_plan = OrchestrationNodeBuilder._parse_router_result(result)

            logger.info("Router Agent node completed with %s agent(s)", len(parsed_agent_plan))
            logger.debug("Router Agent parsed plan=%s", parsed_agent_plan)

            return {
                **state,
                "agent_plan": parsed_agent_plan,
            }

        return router_agent_node

    @staticmethod
    def build_finalize_output() -> Callable[[State], dict[str, Any]]:

        def finalize_output_node(state: State, /) -> dict[str, Any]:
            logger.info("Finalize Output node started")

            errors = list(state.get("errors", []))
            report = state.get("report", "")
            logger.debug(
                "Finalize Output payload | error_count=%s | report_present=%s",
                len(errors),
                bool(str(report or "").strip()),
            )

            if errors and not report:
                report = f"Error: {'; '.join(errors)}"
                logger.info("Finalize Output synthesized report from errors")

            logger.info("Finalize Output node completed")

            return {
                **state,
                "report": report,
            }

        return finalize_output_node
