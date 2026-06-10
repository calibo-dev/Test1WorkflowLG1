from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.llm.llm_factory import get_llm
from latest_ai_development.prompts.prompt_builder import build_research_prompt
from latest_ai_development.tools.tool_registry import get_tools


def build_research_node():
    """
    Build the research stage LangGraph node.

    Pipeline:
        Prompt → LLM (optional tool calling) → Output Parser
    """

    # Load stage configuration
    stages_cfg = get_stages_config()
    stage_cfg = stages_cfg.get("research")

    if not stage_cfg:
        raise ValueError("Missing 'research' configuration in stages.yaml")

    # Build prompt
    prompt = build_research_prompt(stage_cfg)

    # Configuration
    use_tools = stage_cfg.get("use_tools", True)
    temperature_override = stage_cfg.get("temperature_override")

    # Load LLM
    llm = (
        get_llm(temperature=temperature_override)
        if temperature_override is not None
        else get_llm()
    )

    # Tool binding
    tools = get_tools()

    if use_tools and tools:
        llm = llm.bind_tools(tools)

    # Parser
    parser = StrOutputParser()

    # Chain
    chain = prompt | llm | parser

    def research_node(state: dict) -> dict:
        research_result = chain.invoke(
            {
                "query": state["query"]
            }
        )

        return {
            **state,
            "research_result": research_result,
        }

    return research_node