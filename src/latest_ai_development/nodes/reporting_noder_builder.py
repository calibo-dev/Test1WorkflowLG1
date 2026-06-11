from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from latest_ai_development.llm.llm_factory import LLMFactory
from latest_ai_development.prompts.prompt_builder import build_reporting_prompt


class ReportingNodeBuilder:

    @staticmethod
    def build():
        prompt = build_reporting_prompt()

        llm = LLMFactory().create_llm(provider="openai")

        chain = prompt | llm | StrOutputParser()

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