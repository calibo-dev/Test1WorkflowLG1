from langgraph.graph import StateGraph, END
from latest_ai_development.graph.state import GraphState
from latest_ai_development.nodes.research_node_builder import ResearchNodeBuilder
from latest_ai_development.nodes.reporting_noder_builder import ReportingNodeBuilder


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