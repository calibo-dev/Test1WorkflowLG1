from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from langgraph.graph import END, StateGraph

from latest_ai_development.config.settings import get_graph_config
from latest_ai_development.graph.state import State


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
    def _resolve_target(target: str):
        return END if target == "END" else target

    @staticmethod
    def build():
        graph_cfg = get_graph_config()
        workflow = StateGraph(State)
        for node_id, node_cfg in graph_cfg["nodes"].items():
            node_callable = GraphBuilder._resolve_node_callable(node_cfg["builder"])
            workflow.add_node(node_id, cast(Any, node_callable))

        workflow.set_entry_point(str(graph_cfg["entry_point"]))

        for edge_cfg in graph_cfg.get("edges", []):
            workflow.add_edge(
                str(edge_cfg["from"]),
                GraphBuilder._resolve_target(str(edge_cfg["to"])),
            )
        for conditional_cfg in graph_cfg.get("conditional_edges", []):
            router_callable = GraphBuilder._resolve_reference(str(conditional_cfg["router"]))
            path_map = {
                str(label): GraphBuilder._resolve_target(str(target))
                for label, target in conditional_cfg["path_map"].items()
            }
            workflow.add_conditional_edges(
                str(conditional_cfg["from"]),
                cast(Any, router_callable),
                cast(Any, path_map),
            )

        return workflow.compile()
