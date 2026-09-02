from __future__ import annotations

import os
from types import SimpleNamespace

import latest_ai_development.tracking as tracking_module


def test_initialize_langsmith_tracing_is_idempotent(monkeypatch):
    monkeypatch.setenv("TRACING_BACKEND", "LANGSMITH")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    monkeypatch.setattr(tracking_module, "_LANGSMITH_INITIALIZED", False)

    tracking_module.initialize_langsmith_tracing()
    tracking_module.initialize_langsmith_tracing()

    assert os.environ["LANGSMITH_API_KEY"] == "test-key"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "test-project"


def test_initialize_langsmith_tracing_disables_when_backend_is_none(monkeypatch):
    monkeypatch.setenv("TRACING_BACKEND", "NONE")
    monkeypatch.setattr(tracking_module, "_LANGSMITH_INITIALIZED", False)

    tracking_module.initialize_langsmith_tracing()

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_initialize_langsmith_tracing_falls_back_when_missing_credentials(monkeypatch):
    monkeypatch.setenv("TRACING_BACKEND", "LANGSMITH")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY_SECRET", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY_SECRET", raising=False)
    monkeypatch.setattr(tracking_module, "_LANGSMITH_INITIALIZED", False)
    monkeypatch.setattr(
        tracking_module,
        "get_settings",
        lambda: SimpleNamespace(
            langsmith_api_key="",
            langsmith_api_key_secret="",
            langsmith_project="",
            langsmith_endpoint="",
        ),
    )

    tracking_module.initialize_langsmith_tracing()

    assert os.environ["TRACING_BACKEND"] == "NONE"
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_workflow_trace_spec_uses_graph_config_override(monkeypatch):
    monkeypatch.setattr(
        tracking_module,
        "get_graph_config",
        lambda: {
            "tracing": {
                "workflow": {
                    "name": "Custom Workflow Trace",
                    "tags": ["workflow", "custom"],
                    "input_fields": {
                        "topic": "inputs.topic",
                    },
                }
            }
        },
    )

    spec = tracking_module._workflow_trace_spec()

    assert spec["name"] == "Custom Workflow Trace"
    assert spec["tags"] == ["workflow", "custom"]
    assert spec["input_fields"] == {"topic": "inputs.topic"}


def test_node_trace_spec_uses_node_override(monkeypatch):
    monkeypatch.setattr(
        tracking_module,
        "get_graph_config",
        lambda: {
            "tracing": {
                "nodes": {
                    "research": {
                        "name": "Research Trace",
                        "metadata": {"agent_name": "research_agent"},
                    }
                }
            }
        },
    )

    spec = tracking_module._node_trace_spec("research")

    assert spec["name"] == "Research Trace"
    assert spec["metadata"]["component_type"] == "node"
    assert spec["metadata"]["node_id"] == "research"
    assert spec["metadata"]["agent_name"] == "research_agent"
