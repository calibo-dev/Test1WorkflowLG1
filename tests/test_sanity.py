from latest_ai_development.config.settings import get_graph_config
from latest_ai_development.workflow import LatestAiDevelopmentWorkflow


def test_langgraph_template_builds_initial_state():
    workflow = LatestAiDevelopmentWorkflow()
    initial_state = workflow.build_initial_state(topic="LangGraph")

    assert initial_state == {
        "topic": "LangGraph",
        "agent_plan": [],
        "errors": [],
        "research_result": "",
        "report": "",
    }


def test_langgraph_template_has_expected_graph_structure():
    graph_config = get_graph_config()

    assert graph_config["entry_point"] == "validate_input"
    assert graph_config["final_output_field"] == "report"
    assert set(graph_config["nodes"]) == {
        "validate_input",
        "router_agent",
        "research",
        "reporting",
        "finalize_output",
    }
    assert graph_config["edges"][-1] == {
        "from": "finalize_output",
        "to": "END",
    }
