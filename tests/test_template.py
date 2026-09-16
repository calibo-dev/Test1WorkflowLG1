from latest_ai_development.config import validators


def test_template_configuration_is_valid() -> None:
    validators._validate_tools_config()
    validators._validate_mcp_config()
    validators._validate_agents_config()
    validators._validate_graph_config()
