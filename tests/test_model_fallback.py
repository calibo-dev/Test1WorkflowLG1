from __future__ import annotations

from latest_ai_development.llm.model_attempt_logging import (
    LoggedModelFallbackMiddleware,
    ModelAttemptLoggingMiddleware,
)
from latest_ai_development.nodes.research_node_builder import ResearchNodeBuilder


def test_research_node_uses_fallback_middleware_when_configured(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setenv("PROVIDER", "OPENAI")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setattr(
        "latest_ai_development.nodes.research_node_builder.build_research_prompt",
        lambda: "prompt",
    )
    monkeypatch.setattr(
        "latest_ai_development.nodes.research_node_builder.LLMFactory.create_llm",
        lambda self, provider=None: f"primary:{provider}",
    )
    monkeypatch.setattr(
        "latest_ai_development.nodes.research_node_builder.get_fallback_llms",
        lambda provider=None: ["openai:gpt-4.1-nano"],
    )

    def fake_create_agent(*, model, tools, system_prompt, middleware):
        captured["model"] = model
        captured["middleware"] = middleware

        class FakeAgent:
            def invoke(self, _agent_input):
                return {"messages": [type("M", (), {"content": "research output"})()]}

        return FakeAgent()

    monkeypatch.setattr(
        "latest_ai_development.nodes.research_node_builder.create_agent",
        fake_create_agent,
    )

    node = ResearchNodeBuilder.build()
    node({"topic": "LangGraph", "agent_plan": [], "errors": [], "research_result": "", "report": ""})

    middleware = captured["middleware"]
    assert captured["model"] == "primary:openai"
    assert isinstance(middleware, list)
    assert len(middleware) == 1
    assert isinstance(middleware[0], LoggedModelFallbackMiddleware)


def test_research_node_uses_primary_logging_when_no_fallback(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setenv("PROVIDER", "OPENAI")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setattr(
        "latest_ai_development.nodes.research_node_builder.build_research_prompt",
        lambda: "prompt",
    )
    monkeypatch.setattr(
        "latest_ai_development.nodes.research_node_builder.LLMFactory.create_llm",
        lambda self, provider=None: f"primary:{provider}",
    )
    monkeypatch.setattr(
        "latest_ai_development.nodes.research_node_builder.get_fallback_llms",
        lambda provider=None: [],
    )
    monkeypatch.setattr(
        "latest_ai_development.nodes.research_node_builder.get_fallback_model_settings",
        lambda provider=None: [],
    )

    def fake_create_agent(*, model, tools, system_prompt, middleware):
        captured["middleware"] = middleware

        class FakeAgent:
            def invoke(self, _agent_input):
                return {"messages": [type("M", (), {"content": "research output"})()]}

        return FakeAgent()

    monkeypatch.setattr(
        "latest_ai_development.nodes.research_node_builder.create_agent",
        fake_create_agent,
    )

    node = ResearchNodeBuilder.build()
    node({"topic": "LangGraph", "agent_plan": [], "errors": [], "research_result": "", "report": ""})

    middleware = captured["middleware"]
    assert isinstance(middleware, list)
    assert len(middleware) == 1
    assert isinstance(middleware[0], ModelAttemptLoggingMiddleware)
