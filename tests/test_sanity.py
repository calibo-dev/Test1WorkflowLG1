from latest_ai_development.config.settings import get_graph_config
from latest_ai_development.graph.state import OUTPUT_FIELDS
from latest_ai_development.workflow import LatestAiDevelopmentWorkflow


def test_langgraph_template_exposes_runtime_contract():
    workflow = LatestAiDevelopmentWorkflow()
    assert set(workflow.required_inputs()).issubset(workflow.input_fields())
    assert workflow.final_output_field in workflow.output_fields()


def test_langgraph_template_has_expected_graph_structure():
    graph_config = get_graph_config()

    nodes = set(graph_config["nodes"])
    assert graph_config["entry_point"] in nodes
    assert graph_config["final_output_field"] in OUTPUT_FIELDS

    for edge in graph_config.get("edges", []):
        assert edge["from"] in nodes
        assert edge["to"] == "END" or edge["to"] in nodes
