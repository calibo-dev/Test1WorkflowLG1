from langgraph.graph import StateGraph, END

from state import GraphState
from nodes.research_node_builder import ResearchNodeBuilder
from nodes.reporting_noder_builder import ReportingNodeBuilder


class GraphBuilder:

    @staticmethod
    def build():

        workflow = StateGraph(GraphState)

        research_node = ResearchNodeBuilder.build()
        reporting_node = ReportingNodeBuilder.build()

        workflow.add_node("research", research_node)
        workflow.add_node("reporting", reporting_node)

        workflow.set_entry_point("research")

        workflow.add_edge("research", "reporting")
        workflow.add_edge("reporting", END)

        return workflow.compile()