from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from latest_ai_development.config.settings import get_graph_config, get_settings
from latest_ai_development.graph.graph_builder import GraphBuilder
from latest_ai_development.graph.state import State
from latest_ai_development.logging_utils import configure_logging, summarize_state_keys
from latest_ai_development.tracking import trace_workflow_execution

logger = logging.getLogger(__name__)


class LatestAiDevelopmentWorkflow:
    """Workflow controller for the LangGraph template."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.graph_config = get_graph_config()
        self.final_output_field = str(self.graph_config.get("final_output_field", "report") or "report")
        self.save_output = bool(self.graph_config.get("execution", {}).get("save_output", False))
        configure_logging(getattr(self.settings, "log_level", "INFO"))
        self._graph = None
        logger.info(
            "Workflow initialized | entry_point=%s | final_output_field=%s | save_output=%s",
            self.graph_config.get("entry_point"),
            self.final_output_field,
            self.save_output,
        )

    def get_graph(self):
        if self._graph is None:
            logger.info("Building LangGraph workflow")
            self._graph = GraphBuilder.build()
        else:
            logger.debug("Using cached LangGraph workflow")
        return self._graph

    def build_initial_state(self, *, topic: str) -> State:
        initial_state: State = {
            "topic": topic,
            "agent_plan": [],
            "errors": [],
            "research_result": "",
            "report": "",
        }
        logger.debug(
            "Initial state prepared | state_keys=%s",
            summarize_state_keys(initial_state),
        )
        return initial_state

    def kickoff(self, inputs: dict[str, Any], flow_run_id: str | None = None) -> State:
        topic = str(inputs.get("topic", "") or "")
        flow_run_id = str(flow_run_id or inputs.get("flow_run_id") or uuid4().hex[:12]).strip()
        if not flow_run_id:
            flow_run_id = uuid4().hex[:12]

        logger.info(
            "Workflow started | flow_run_id=%s | provider=%s | topic=%s",
            flow_run_id,
            getattr(self.settings, "provider", "unknown"),
            topic,
        )

        initial_state = self.build_initial_state(topic=topic)
        result = trace_workflow_execution(
            self.get_graph().invoke,
            flow_run_id=flow_run_id,
            inputs=inputs,
            initial_state=initial_state,
        )

        report = result.get(self.final_output_field, "")
        if isinstance(report, str) and report.strip():
            self.settings.report_output_path.write_text(report, encoding="utf-8")
            logger.info(
                "Report saved | flow_run_id=%s | output_path=%s",
                flow_run_id,
                self.settings.report_output_path,
            )

        logger.info("Workflow completed | flow_run_id=%s", flow_run_id)

        return result
