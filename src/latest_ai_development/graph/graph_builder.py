from typing import Any, cast

from langgraph.graph import StateGraph, END
from latest_ai_development.graph.state import State
from latest_ai_development.nodes.orchestration_node_builder import OrchestrationNodeBuilder
from latest_ai_development.nodes.research_node_builder import ResearchNodeBuilder
from latest_ai_development.nodes.reporting_node_builder import ReportingNodeBuilder


class GraphBuilder:

    @staticmethod
    def build():

        workflow = StateGraph(State)

        validate_input_node = OrchestrationNodeBuilder.build_validate_input()
        router_agent_node = OrchestrationNodeBuilder.build_router_agent()
        research_node = ResearchNodeBuilder.build()
        reporting_node = ReportingNodeBuilder.build()
        finalize_output_node = OrchestrationNodeBuilder.build_finalize_output()

        workflow.add_node("validate_input", cast(Any, validate_input_node))
        workflow.add_node("router_agent", cast(Any, router_agent_node))
        workflow.add_node("research", cast(Any, research_node))
        workflow.add_node("reporting", cast(Any, reporting_node))
        workflow.add_node("finalize_output", cast(Any, finalize_output_node))

        workflow.set_entry_point("validate_input")

        workflow.add_edge("validate_input", "router_agent")
        workflow.add_conditional_edges(
            "router_agent",
            OrchestrationNodeBuilder.route_after_router_agent,
            {
                "research": "research",
                "reporting": "reporting",
                "finalize_output": "finalize_output",
            },
        )
        workflow.add_conditional_edges(
            "research",
            OrchestrationNodeBuilder.route_after_research,
            {
                "reporting": "reporting",
                "finalize_output": "finalize_output",
            },
        )
        workflow.add_edge("reporting", "finalize_output")
        workflow.add_edge("finalize_output", END)

        return workflow.compile()