from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from functools import wraps
from importlib import import_module
from typing import Any, cast

from langgraph.graph import END, StateGraph

from latest_ai_development.config.settings import get_graph_config
from latest_ai_development.graph.state import State
from latest_ai_development.logging_utils import summarize_state_delta, summarize_state_keys
from latest_ai_development.tracking import trace_node_execution

logger = logging.getLogger(__name__)


class GraphBuilder:
    @staticmethod
    def _resolve_reference(reference: str):
        module_path, separator, attribute_path = reference.partition(":")
        if not separator or not module_path or not attribute_path:
            raise ValueError(
                f"Callable reference '{reference}' must use '<module>:<attribute>' format"
            )

        module = import_module(module_path)
        resolved = module
        for attribute_name in attribute_path.split("."):
            resolved = getattr(resolved, attribute_name)

        if not callable(resolved):
            raise TypeError(f"Resolved reference '{reference}' is not callable")

        return resolved

    @classmethod
    def _resolve_node_callable(cls, reference: str):
        builder = cls._resolve_reference(reference)
        node_callable = builder()
        if not callable(node_callable):
            raise TypeError(f"Builder '{reference}' did not return a callable node")
        return node_callable

    @staticmethod
    def _wrap_node(node_id: str, node_callable):
        def wrapped_node(state):
            return trace_node_execution(node_id, node_callable, state=state)

        return wrapped_node

    @staticmethod
    def _resolve_target(target: str):
        return END if target == "END" else target

    @staticmethod
    def _as_state_mapping(value: Any) -> Mapping[str, Any] | None:
        return value if isinstance(value, Mapping) else None

    @classmethod
    def _wrap_node_callable(cls, node_id: str, node_callable: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(node_callable)
        def wrapped_node(*args: Any, **kwargs: Any) -> Any:
            state = cls._as_state_mapping(args[0] if args else None)
            logger.info("Node entered | node_id=%s", node_id)
            logger.debug(
                "Node input snapshot | node_id=%s | state_keys=%s",
                node_id,
                summarize_state_keys(state),
            )

            try:
                result = node_callable(*args, **kwargs)
            except Exception:
                logger.exception("Node failed | node_id=%s", node_id)
                raise

            result_state = cls._as_state_mapping(result)
            logger.info("Node exited | node_id=%s", node_id)
            logger.debug(
                "Node output snapshot | node_id=%s | state_keys=%s | delta=%s",
                node_id,
                summarize_state_keys(result_state),
                summarize_state_delta(state, result_state),
            )
            return result

        return wrapped_node

    @classmethod
    def _wrap_router_callable(cls, source_node_id: str, router_callable: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(router_callable)
        def wrapped_router(*args: Any, **kwargs: Any) -> Any:
            state = cls._as_state_mapping(args[0] if args else None)
            logger.info("Router evaluated | source_node=%s", source_node_id)
            logger.debug(
                "Router input snapshot | source_node=%s | state_keys=%s",
                source_node_id,
                summarize_state_keys(state),
            )

            try:
                decision = router_callable(*args, **kwargs)
            except Exception:
                logger.exception("Router failed | source_node=%s", source_node_id)
                raise

            logger.info(
                "Router selected path | source_node=%s | decision=%s",
                source_node_id,
                decision,
            )
            return decision

        return wrapped_router

    @staticmethod
    def build():
        graph_cfg = get_graph_config()
        logger.info(
            "Graph compilation started | entry_point=%s | final_output_field=%s | node_count=%s",
            graph_cfg.get("entry_point"),
            graph_cfg.get("final_output_field"),
            len(graph_cfg.get("nodes", {})),
        )
        logger.debug("Graph configuration nodes=%s", sorted(graph_cfg.get("nodes", {}).keys()))
        workflow = StateGraph(State)
        for node_id, node_cfg in graph_cfg["nodes"].items():
            logger.info(
                "Creating node callable | node_id=%s | builder=%s",
                node_id,
                node_cfg["builder"],
            )
            node_callable = GraphBuilder._resolve_node_callable(node_cfg["builder"])
            workflow.add_node(
                node_id,
                cast(Any, GraphBuilder._wrap_node(node_id, node_callable)),
            )

        workflow.set_entry_point(str(graph_cfg["entry_point"]))
        logger.info("Graph entry point registered | entry_point=%s", graph_cfg["entry_point"])

        for edge_cfg in graph_cfg.get("edges", []):
            logger.debug(
                "Registering edge | from=%s | to=%s",
                edge_cfg["from"],
                edge_cfg["to"],
            )
            workflow.add_edge(
                str(edge_cfg["from"]),
                GraphBuilder._resolve_target(str(edge_cfg["to"])),
            )
        for conditional_cfg in graph_cfg.get("conditional_edges", []):
            logger.info(
                "Registering conditional edges | source_node=%s | router=%s",
                conditional_cfg["from"],
                conditional_cfg["router"],
            )
            router_callable = GraphBuilder._resolve_reference(str(conditional_cfg["router"]))
            path_map = {
                str(label): GraphBuilder._resolve_target(str(target))
                for label, target in conditional_cfg["path_map"].items()
            }
            logger.debug(
                "Conditional edge map | source_node=%s | path_map=%s",
                conditional_cfg["from"],
                path_map,
            )
            workflow.add_conditional_edges(
                str(conditional_cfg["from"]),
                cast(Any, GraphBuilder._wrap_router_callable(str(conditional_cfg["from"]), router_callable)),
                cast(Any, path_map),
            )

        compiled_graph = workflow.compile()
        logger.info("Graph compilation completed")
        return compiled_graph
