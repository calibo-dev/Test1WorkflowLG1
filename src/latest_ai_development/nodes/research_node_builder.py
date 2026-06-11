from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from latest_ai_development.llm.llm_factory import LLMFactory
from latest_ai_development.prompts.prompt_builder import build_research_prompt


class ResearchNodeBuilder:

    @staticmethod
    def build():
        prompt = build_research_prompt()

        llm = LLMFactory().create_llm(provider="openai")

        chain = prompt | llm | StrOutputParser()

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