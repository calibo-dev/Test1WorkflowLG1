from __future__ import annotations

from latest_ai_development.graph.graph_builder import GraphBuilder


def test_graph_builder_compiles_workflow_from_graph_config(monkeypatch):
    custom_graph_config = {
        "entry_point": "start",
        "final_output_field": "report",
        "nodes": {
            "start": {"builder": "builder:start"},
            "branch_a": {"builder": "builder:branch_a"},
            "finish": {"builder": "builder:finish"},
        },
        "edges": [
            {"from": "branch_a", "to": "finish"},
            {"from": "finish", "to": "END"},
        ],
        "conditional_edges": [
            {
                "from": "start",
                "router": "router:start",
                "path_map": {
                    "branch_a": "branch_a",
                },
            }
        ],
    }

    def build_start():
        def start_node(state, /):
            return {
                **state,
                "research_result": "seed",
            }

        return start_node

    def build_branch_a():
        def branch_a_node(state, /):
            return {
                **state,
                "report": f"branch from {state['research_result']}",
            }

        return branch_a_node

    def build_finish():
        def finish_node(state, /):
            return {
                **state,
                "report": f"{state['report']} -> done",
            }

        return finish_node

    def route_start(_state):
        return "branch_a"

    reference_map = {
        "builder:start": build_start,
        "builder:branch_a": build_branch_a,
        "builder:finish": build_finish,
        "router:start": route_start,
    }

    monkeypatch.setattr(
        "latest_ai_development.graph.graph_builder.get_graph_config",
        lambda: custom_graph_config,
    )
    monkeypatch.setattr(
        GraphBuilder,
        "_resolve_reference",
        staticmethod(lambda reference: reference_map[reference]),
    )

    graph = GraphBuilder.build()
    result = graph.invoke(
        {
            "topic": "LangGraph",
            "agent_plan": [],
            "errors": [],
            "research_result": "",
            "report": "",
        }
    )

    assert result["report"] == "branch from seed -> done"
