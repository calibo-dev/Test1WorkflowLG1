from __future__ import annotations

import asyncio
import logging
from inspect import isawaitable
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from latest_ai_development.config.settings import get_graph_config, get_settings
from latest_ai_development.graph.context import RuntimeContext
from latest_ai_development.graph.graph_builder import GraphBuilder
from latest_ai_development.graph.state import State
from latest_ai_development.logging_utils import configure_logging, summarize_state_keys
from latest_ai_development.tools.tool_registry import close_registered_tools
from latest_ai_development.tracking import atrace_workflow_execution

logger = logging.getLogger(__name__)


class LatestAiDevelopmentWorkflow:
    """Workflow controller for the LangGraph template."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.graph_config = get_graph_config()
        self.final_output_field = str(
            self.graph_config.get("final_output_field", "final_response") or "final_response"
        )
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

    async def astart(self) -> None:
        """Compile all configured nodes and agents during application startup."""
        self.get_graph()

    async def aclose(self) -> None:
        """Release graph and tool resources owned by this workflow process."""
        graph = self._graph
        self._graph = None
        close = getattr(graph, "aclose", None) or getattr(graph, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result
        await close_registered_tools()

    def build_initial_state(self, *, user_input: str) -> State:
        initial_state: State = {"user_input": user_input, "errors": []}
        logger.debug(
            "Initial state prepared | state_keys=%s",
            summarize_state_keys(initial_state),
        )
        return initial_state

    def build_runtime_context(
        self,
        overrides: dict[str, Any] | None = None,
    ) -> RuntimeContext:
        configured_defaults = self.graph_config.get("runtime_context", {}).get("defaults", {})
        values = dict(configured_defaults)
        values.update(overrides or {})
        return {"values": values}

    async def akickoff(
        self,
        inputs: dict[str, Any],
        flow_run_id: str | None = None,
        *,
        thread_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> State:
        user_input = str(inputs.get("user_input", "") or "")
        flow_run_id = str(flow_run_id or inputs.get("flow_run_id") or uuid4().hex[:12]).strip()
        if not flow_run_id:
            flow_run_id = uuid4().hex[:12]
        thread_id = str(thread_id or inputs.get("thread_id") or flow_run_id).strip()

        logger.info(
            "Workflow started | flow_run_id=%s | thread_id=%s",
            flow_run_id,
            thread_id,
        )

        initial_state = self.build_initial_state(user_input=user_input)
        result = await self._ainvoke(
            initial_state,
            flow_run_id=flow_run_id,
            inputs=inputs,
            thread_id=thread_id,
            context=self.build_runtime_context(context or inputs.get("context")),
        )
        self._save_result(result, flow_run_id=flow_run_id)
        logger.info("Workflow completed | flow_run_id=%s", flow_run_id)
        return result

    async def aresume(
        self,
        *,
        thread_id: str,
        value: Any,
        flow_run_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> State:
        flow_run_id = str(flow_run_id or uuid4().hex[:12])
        result = await self._ainvoke(
            Command(resume=value),
            flow_run_id=flow_run_id,
            inputs={"thread_id": thread_id, "resume": value},
            thread_id=thread_id,
            context=self.build_runtime_context(context),
        )
        self._save_result(result, flow_run_id=flow_run_id)
        return result

    async def _ainvoke(
        self,
        graph_input: State | Command,
        *,
        flow_run_id: str,
        inputs: dict[str, Any],
        thread_id: str,
        context: RuntimeContext,
    ) -> State:
        config = {"configurable": {"thread_id": thread_id}}
        return await atrace_workflow_execution(
            lambda value: self.get_graph().ainvoke(value, config=config, context=context),
            flow_run_id=flow_run_id,
            inputs=inputs,
            initial_state=graph_input,
        )

    def _save_result(self, result: State, *, flow_run_id: str) -> None:
        final_response = result.get(self.final_output_field, "")
        if self.save_output and isinstance(final_response, str) and final_response.strip():
            try:
                self.settings.report_output_path.write_text(final_response, encoding="utf-8")
                logger.info(
                    "Report saved | flow_run_id=%s | output_path=%s",
                    flow_run_id,
                    self.settings.report_output_path,
                )
            except OSError as exc:
                logger.warning(
                    "Report save failed | flow_run_id=%s | output_path=%s | error=%s",
                    flow_run_id,
                    self.settings.report_output_path,
                    exc,
                )

    def kickoff(
        self,
        inputs: dict[str, Any],
        flow_run_id: str | None = None,
        *,
        thread_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> State:
        """Synchronous boundary for CLI callers; async servers should use akickoff."""
        return asyncio.run(
            self.akickoff(
                inputs,
                flow_run_id=flow_run_id,
                thread_id=thread_id,
                context=context,
            )
        )
