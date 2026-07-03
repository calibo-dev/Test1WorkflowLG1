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
        primary_researcher_node = PrimaryResearcherNodeBuilder.build()
        primary_writer_node = PrimaryWriterNodeBuilder.build()
        finalize_output_node = OrchestrationNodeBuilder.build_finalize_output()

        workflow.add_node("validate_input", cast(Any, validate_input_node))
        workflow.add_node("primary_researcher", cast(Any, primary_researcher_node))
        workflow.add_node("primary_writer", cast(Any, primary_writer_node))
        workflow.add_node("finalize_output", cast(Any, finalize_output_node))

        workflow.set_entry_point("validate_input")

        workflow.add_edge("validate_input", "primary_researcher")
        workflow.add_edge("primary_researcher", "primary_writer")
        workflow.add_edge("primary_writer", "finalize_output")
        workflow.add_edge("finalize_output", END)

        return workflow.compile()