from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langgraph.types import Command, Send, interrupt

from latest_ai_development.config import validators
from latest_ai_development.config.settings import (
    get_agents_config,
    get_graph_config,
    get_mcp_config,
    get_models_config,
    get_settings,
    get_tools_config,
)
from latest_ai_development.graph.graph_builder import GraphBuilder
from latest_ai_development.nodes import agent_node_builder
from latest_ai_development.nodes.agent_node_builder import AgentNodeBuilder
from latest_ai_development.nodes.orchestration_node_builder import (
    FeedbackOrchestrationNodeBuilder,
    route_after_evaluation,
)
from latest_ai_development.tools import tool_registry
from latest_ai_development.workflow import LatestAiDevelopmentWorkflow


@pytest.fixture(autouse=True)
def clear_runtime_state(monkeypatch):
    monkeypatch.setenv("TRACING_BACKEND", "NONE")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    get_settings.cache_clear()
    get_models_config.cache_clear()
    get_graph_config.cache_clear()
    get_agents_config.cache_clear()
    get_tools_config.cache_clear()
    get_mcp_config.cache_clear()
    tool_registry.REGISTERED_TOOLS.clear()
    yield
    get_settings.cache_clear()
    get_models_config.cache_clear()
    get_graph_config.cache_clear()
    get_agents_config.cache_clear()
    get_tools_config.cache_clear()
    get_mcp_config.cache_clear()
    tool_registry.REGISTERED_TOOLS.clear()


def test_generated_configuration_satisfies_template_contract():
    validators._validate_tools_config()
    validators._validate_mcp_config()
    validators._validate_agents_config()
    validators._validate_graph_config()


def test_agent_configuration_allows_deterministic_only_workflows(monkeypatch):
    monkeypatch.setattr(validators, "get_agents_config", lambda: {"agents": {}})

    validators._validate_agents_config()


def test_structured_agent_output_uses_declared_state_types():
    response_model = agent_node_builder._build_response_format(
        "scoring_agent",
        {"output_mapping": {"score": "urgency_score"}},
    )

    assert response_model is not None
    assert response_model.model_fields["score"].annotation is int


def test_graph_builder_compiles_direct_and_conditional_routes(monkeypatch):
    graph_config = {
        "entry_point": "start",
        "final_output_field": "final_response",
        "nodes": {
            "start": {"type": "deterministic", "builder": "test:start"},
            "work": {"type": "deterministic", "builder": "test:work"},
            "finish": {"type": "deterministic", "builder": "test:finish"},
        },
        "edges": [
            {"from": "work", "to": "finish"},
            {"from": "finish", "to": "END"},
        ],
        "conditional_edges": [
            {
                "from": "start",
                "decision_field": "validation_status",
                "path_map": {"continue": "work"},
            }
        ],
    }

    def node(update):
        return lambda state: update(state)

    builders = {
        "test:start": lambda: node(lambda _state: {"validation_status": "continue"}),
        "test:work": lambda: node(lambda _state: {"draft_response": "ready"}),
        "test:finish": lambda: node(
            lambda state: {"final_response": f"result: {state['draft_response']}"}
        ),
    }
    monkeypatch.setattr(
        "latest_ai_development.graph.graph_builder.get_graph_config",
        lambda: graph_config,
    )
    monkeypatch.setattr(
        GraphBuilder,
        "_resolve_reference",
        staticmethod(lambda reference: builders[reference]),
    )

    result = asyncio.run(GraphBuilder.build().ainvoke({"user_input": "request"}))

    assert result["final_response"] == "result: ready"


def test_graph_builder_passes_runtime_context_to_nodes(monkeypatch):
    graph_config = {
        "entry_point": "context_node",
        "end_nodes": ["context_node"],
        "final_output_field": "final_response",
        "runtime_context": {"defaults": {"prefix": "default"}},
        "nodes": {
            "context_node": {"type": "deterministic", "builder": "test:context"},
        },
        "edges": [{"from": "context_node", "to": "END"}],
        "conditional_edges": [],
    }

    def build_context_node():
        def context_node(state, runtime):
            return {
                "final_response": f"{runtime.context['values']['prefix']}: {state['user_input']}"
            }

        return context_node

    monkeypatch.setattr(
        "latest_ai_development.graph.graph_builder.get_graph_config",
        lambda: graph_config,
    )
    monkeypatch.setattr(
        GraphBuilder,
        "_resolve_reference",
        staticmethod(lambda _reference: build_context_node),
    )

    result = asyncio.run(
        GraphBuilder.build().ainvoke(
            {"user_input": "request"},
            context={"values": {"prefix": "tenant-a"}},
        )
    )

    assert result["final_response"] == "tenant-a: request"


def test_graph_builder_supports_command_routing(monkeypatch):
    graph_config = {
        "entry_point": "route",
        "end_nodes": ["finish"],
        "final_output_field": "final_response",
        "nodes": {
            "route": {
                "type": "deterministic",
                "builder": "test:route",
                "command_destinations": ["finish"],
            },
            "finish": {"type": "deterministic", "builder": "test:finish"},
        },
        "edges": [{"from": "finish", "to": "END"}],
        "conditional_edges": [],
    }
    builders = {
        "test:route": lambda: (
            lambda _state: Command(
                update={"draft_response": "routed"},
                goto="finish",
            )
        ),
        "test:finish": lambda: lambda state: {"final_response": state["draft_response"]},
    }
    monkeypatch.setattr(
        "latest_ai_development.graph.graph_builder.get_graph_config",
        lambda: graph_config,
    )
    monkeypatch.setattr(
        GraphBuilder,
        "_resolve_reference",
        staticmethod(lambda reference: builders[reference]),
    )

    result = asyncio.run(GraphBuilder.build().ainvoke({"user_input": "request"}))

    assert result["final_response"] == "routed"


def test_graph_builder_supports_dynamic_send_fanout(monkeypatch):
    graph_config = {
        "entry_point": "dispatch",
        "end_nodes": ["worker"],
        "final_output_field": "final_response",
        "nodes": {
            "dispatch": {"type": "deterministic", "builder": "test:dispatch"},
            "worker": {"type": "deterministic", "builder": "test:worker"},
        },
        "edges": [{"from": "worker", "to": "END"}],
        "conditional_edges": [
            {
                "from": "dispatch",
                "router": "test:fanout",
                "send_destinations": ["worker"],
            }
        ],
    }
    builders = {
        "test:dispatch": lambda: lambda _state: {},
        "test:worker": lambda: lambda state: {"errors": [state["user_input"]]},
        "test:fanout": lambda state: [
            Send("worker", {"user_input": item.strip()}) for item in state["user_input"].split(",")
        ],
    }
    monkeypatch.setattr(
        "latest_ai_development.graph.graph_builder.get_graph_config",
        lambda: graph_config,
    )
    monkeypatch.setattr(
        GraphBuilder,
        "_resolve_reference",
        staticmethod(lambda reference: builders[reference]),
    )

    result = asyncio.run(GraphBuilder.build().ainvoke({"user_input": "one,two"}))

    assert sorted(result["errors"]) == ["one", "two"]


@pytest.mark.parametrize(
    ("agent_config", "content", "expected"),
    [
        (
            {
                "output_field": "draft_response",
                "writes_state": ["draft_response"],
            },
            "plain result",
            {"draft_response": "plain result"},
        ),
        (
            {
                "output_mapping": {
                    "classification": "sentiment",
                    "reason": "sentiment_reason",
                },
                "writes_state": ["sentiment", "sentiment_reason"],
            },
            {"classification": "positive", "reason": "structured result"},
            {"sentiment": "positive", "sentiment_reason": "structured result"},
        ),
    ],
)
def test_agent_node_maps_model_output(monkeypatch, agent_config, content, expected):
    configured_agent = {
        "role": "Worker",
        "prompt": "prompts/worker.md",
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "reads_state": ["user_input"],
        "tools": [],
        **agent_config,
    }
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        agent_node_builder,
        "get_agents_config",
        lambda: {"agents": {"worker": configured_agent}},
    )
    monkeypatch.setattr(agent_node_builder, "_load_prompt", lambda _reference: "Prompt")
    monkeypatch.setattr(agent_node_builder, "_build_middleware", lambda *_args: [])
    monkeypatch.setattr(
        agent_node_builder.LLMFactory,
        "create_llm",
        lambda self, provider=None, model=None, section="primary": (
            captured.update(
                selected_provider=provider,
                selected_model=model,
            )
            or "llm"
        ),
    )
    monkeypatch.setattr(agent_node_builder, "get_tools_for_node", lambda _node_id: ["tool"])

    class FakeAgent:
        async def ainvoke(self, agent_input):
            captured["input"] = agent_input
            if configured_agent.get("output_mapping"):
                return {"structured_response": content}
            message = type("Message", (), {"content": content})()
            return {"messages": [message]}

    def create_fake_agent(**kwargs):
        captured.update(kwargs)
        captured["create_count"] = int(captured.get("create_count", 0)) + 1
        return FakeAgent()

    monkeypatch.setattr(agent_node_builder, "create_agent", create_fake_agent)

    node_callable = AgentNodeBuilder.build(node_id="worker_node", agent_id="worker")
    result = asyncio.run(node_callable({"user_input": "request"}))
    asyncio.run(node_callable({"user_input": "second request"}))

    assert result == expected
    assert captured["selected_provider"] == "openai"
    assert captured["selected_model"] == "gpt-4.1-mini"
    assert captured["model"] == "llm"
    assert captured["tools"] == ["tool"]
    assert captured["create_count"] == 1
    if configured_agent.get("output_mapping"):
        assert captured["response_format"].__name__ == "WorkerResponse"


def test_tool_registry_resolves_tools_from_agent_configuration(monkeypatch):
    tool = object()
    monkeypatch.setattr(
        tool_registry,
        "get_graph_config",
        lambda: {"nodes": {"worker_node": {"type": "agent", "agent": "worker"}}},
    )
    monkeypatch.setattr(
        tool_registry,
        "get_agents_config",
        lambda: {"agents": {"worker": {"tools": ["lookup"]}}},
    )
    monkeypatch.setattr(tool_registry, "get_tools_config", lambda: {"tools": {}})
    tool_registry.register_tool("lookup", tool)

    assert tool_registry.get_tools_for_node("worker_node") == [tool]


def test_tool_registry_passes_factory_configuration_and_caches_tool(monkeypatch):
    calls: list[dict[str, object]] = []
    tool = object()

    def create_tool(**kwargs):
        calls.append(kwargs)
        return tool

    monkeypatch.setattr(
        tool_registry,
        "get_tools_config",
        lambda: {
            "tools": {
                "country_lookup": {
                    "factory": "example.tools:create_tool",
                    "factory_kwargs": {"table_name": "ak_country", "read_only": True},
                }
            }
        },
    )
    monkeypatch.setattr(
        tool_registry,
        "import_module",
        lambda _module_name: SimpleNamespace(create_tool=create_tool),
    )

    assert tool_registry.get_tools_by_names(["country_lookup"]) == [tool]
    assert tool_registry.get_tools_by_names(["country_lookup"]) == [tool]
    assert calls == [{"table_name": "ak_country", "read_only": True}]


def test_tool_registry_supports_many_tools_on_one_node(monkeypatch):
    first_tool = object()
    second_tool = object()
    monkeypatch.setattr(
        tool_registry,
        "get_graph_config",
        lambda: {"nodes": {"worker_node": {"type": "agent", "agent": "worker"}}},
    )
    monkeypatch.setattr(
        tool_registry,
        "get_agents_config",
        lambda: {
            "agents": {
                "worker": {"tools": ["first", "second", "first"]},
            }
        },
    )
    monkeypatch.setattr(tool_registry, "get_tools_config", lambda: {"tools": {}})
    tool_registry.register_tool("first", first_tool)
    tool_registry.register_tool("second", second_tool)

    assert tool_registry.get_tools_for_node("worker_node") == [first_tool, second_tool]


def test_tool_registry_reuses_one_tool_across_multiple_nodes(monkeypatch):
    calls: list[str] = []
    shared_tool = object()

    def create_tool():
        calls.append("created")
        return shared_tool

    monkeypatch.setattr(
        tool_registry,
        "get_graph_config",
        lambda: {
            "nodes": {
                "first_node": {"type": "agent", "agent": "first_agent"},
                "second_node": {"type": "agent", "agent": "second_agent"},
            }
        },
    )
    monkeypatch.setattr(
        tool_registry,
        "get_agents_config",
        lambda: {
            "agents": {
                "first_agent": {"tools": ["shared"]},
                "second_agent": {"tools": ["shared"]},
            }
        },
    )
    monkeypatch.setattr(
        tool_registry,
        "get_tools_config",
        lambda: {"tools": {"shared": {"factory": "example.tools:create_tool"}}},
    )
    monkeypatch.setattr(
        tool_registry,
        "import_module",
        lambda _module_name: SimpleNamespace(create_tool=create_tool),
    )

    assert tool_registry.get_tools_for_node("first_node") == [shared_tool]
    assert tool_registry.get_tools_for_node("second_node") == [shared_tool]
    assert calls == ["created"]


def test_tool_resources_close_once_at_shutdown():
    class ClosableTool:
        def __init__(self):
            self.close_count = 0

        async def aclose(self):
            self.close_count += 1

    tool = ClosableTool()
    tool_registry.register_tool("first", tool)
    tool_registry.register_tool("alias", tool)

    asyncio.run(tool_registry.close_registered_tools())

    assert tool.close_count == 1
    assert tool_registry.REGISTERED_TOOLS == {}


def test_fastapi_lifespan_starts_and_closes_one_workflow(monkeypatch):
    from latest_ai_development import main

    calls: list[str] = []

    class FakeWorkflow:
        async def astart(self):
            calls.append("start")

        async def aclose(self):
            calls.append("close")

    runtime = FakeWorkflow()
    monkeypatch.setattr(main, "initialize_langsmith_tracing", lambda: None)
    monkeypatch.setattr(main, "_initialize_workflow", lambda: runtime)

    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        assert main.app.state.workflow is runtime

    assert calls == ["start", "close"]
    assert not hasattr(main.app.state, "workflow")


def test_graph_validation_rejects_mixed_routing(monkeypatch):
    graph_config = deepcopy(get_graph_config())
    conditional_source = graph_config["conditional_edges"][0]["from"]
    graph_config["edges"].append({"from": conditional_source, "to": graph_config["end_nodes"][0]})
    monkeypatch.setattr(validators, "get_graph_config", lambda: graph_config)

    with pytest.raises(ValueError, match="cannot mix direct and conditional"):
        validators._validate_graph_config()


def test_graph_validation_accepts_command_and_send_routes(monkeypatch):
    graph_config = {
        "entry_point": "route",
        "end_nodes": ["worker"],
        "final_output_field": "final_response",
        "runtime_context": {"defaults": {"tenant": "default"}},
        "nodes": {
            "route": {
                "type": "deterministic",
                "builder": "test:route",
                "command_destinations": ["dispatch"],
            },
            "dispatch": {"type": "deterministic", "builder": "test:dispatch"},
            "worker": {"type": "deterministic", "builder": "test:worker"},
        },
        "edges": [{"from": "worker", "to": "END"}],
        "conditional_edges": [
            {
                "from": "dispatch",
                "router": "test:fanout",
                "send_destinations": ["worker"],
            }
        ],
    }
    monkeypatch.setattr(validators, "get_graph_config", lambda: graph_config)
    monkeypatch.setattr(validators, "get_agents_config", lambda: {"agents": {}})
    monkeypatch.setattr(validators, "_resolve_reference", lambda _reference: lambda: None)

    validators._validate_graph_config()


@pytest.mark.parametrize(
    ("state", "expected_route"),
    [
        (
            {"sentiment": "positive", "confidence": "0.95"},
            "appreciation",
        ),
        (
            {"sentiment": "mixed", "confidence": "0.90"},
            "clarification",
        ),
        (
            {"sentiment": "negative", "confidence": "0.90", "urgency_score": 35},
            "service_recovery",
        ),
        (
            {"sentiment": "positive", "confidence": "0.90", "policy_risk_score": 70},
            "human_review",
        ),
    ],
)
def test_complex_feedback_router_uses_multiple_state_signals(state, expected_route):
    assert route_after_evaluation(state) == expected_route


def test_weighted_deterministic_evaluator_combines_rules():
    evaluator = FeedbackOrchestrationNodeBuilder.build_evaluate_feedback()

    result = evaluator(
        {
            "user_input": "request",
            "normalized_feedback": "This unsafe product caused an injury. I need an urgent refund.",
            "sentiment": "negative",
            "confidence": "0.95",
        }
    )

    assert result == {"urgency_score": 75, "policy_risk_score": 100}


def test_graph_interrupt_resumes_with_persisted_thread(monkeypatch):
    graph_config = {
        "entry_point": "approval",
        "end_nodes": ["approval"],
        "final_output_field": "final_response",
        "persistence": {"enabled": True, "backend": "memory"},
        "nodes": {
            "approval": {"type": "deterministic", "builder": "test:approval"},
        },
        "edges": [{"from": "approval", "to": "END"}],
        "conditional_edges": [],
    }

    def build_approval_node():
        def approval_node(_state):
            decision = interrupt("Approve this result?")
            return {"final_response": str(decision)}

        return approval_node

    monkeypatch.setattr(
        "latest_ai_development.graph.graph_builder.get_graph_config",
        lambda: graph_config,
    )
    monkeypatch.setattr(
        GraphBuilder,
        "_resolve_reference",
        staticmethod(lambda _reference: build_approval_node),
    )
    graph = GraphBuilder.build()
    config = {"configurable": {"thread_id": "thread-1"}}

    paused = asyncio.run(graph.ainvoke({"user_input": "request"}, config=config))
    resumed = asyncio.run(graph.ainvoke(Command(resume="approved"), config=config))

    assert paused["__interrupt__"][0].value == "Approve this result?"
    assert resumed["final_response"] == "approved"


def test_workflow_builds_one_canonical_input_state():
    workflow = LatestAiDevelopmentWorkflow()
    state = workflow.build_initial_state({"user_input": "request", "ignored": "value"})

    assert state["user_input"] == "request"
    assert state["errors"] == []
    assert set(state) == {"user_input", "errors"}
    assert workflow.required_inputs() == ("user_input",)
    assert workflow.output_fields() == ("final_response",)
