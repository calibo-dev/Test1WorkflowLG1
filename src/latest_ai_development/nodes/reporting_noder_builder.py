from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from latest_ai_development.config.settings import get_stages_config
from latest_ai_development.llm.llm_factory import get_llm
from latest_ai_development.prompts.prompt_builder import build_reporting_prompt


def build_reporting_node():
    """
    Build the reporting stage LangGraph node.

    Pipeline:
        Prompt → LLM → Output Parser
    """

    # Load stage configuration
    stages_cfg = get_stages_config()
    stage_cfg = stages_cfg.get("reporting")

    if not stage_cfg:
        raise ValueError("Missing 'reporting' configuration in stages.yaml")

    # Build prompt
    prompt = build_reporting_prompt(stage_cfg)

    # Configuration
    temperature_override = stage_cfg.get("temperature_override")

    # Load LLM
    llm = (
        get_llm(temperature=temperature_override)
        if temperature_override is not None
        else get_llm()
    )

    # Parser
    parser = StrOutputParser()

    # Chain
    chain = prompt | llm | parser

    def reporting_node(state: dict) -> dict:
        report = chain.invoke(
            {
                "query": state["query"],
                "research_result": state["research_result"],
            }
        )

        return {
            **state,
            "report": report,
        }

    return reporting_node