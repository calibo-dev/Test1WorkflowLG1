from __future__ import annotations

import importlib
import sys

from fastapi.testclient import TestClient


def _load_main_module(monkeypatch):
    monkeypatch.setenv("PROVIDER", "OLLAMA")
    monkeypatch.setenv("CLOUD_PROVIDER", "")
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "")
    monkeypatch.delenv("OLLAMA_API_KEY_SECRET", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_SECRET", raising=False)
    monkeypatch.delenv("ANTHROPICAI_API_KEY_SECRET", raising=False)
    monkeypatch.delenv("GEMINIAI_API_KEY_SECRET", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    sys.modules.pop("latest_ai_development.main", None)
    return importlib.import_module("latest_ai_development.main")


def test_ask_returns_report(monkeypatch):
    main_module = _load_main_module(monkeypatch)
    monkeypatch.setattr(main_module, "initialize_langsmith_tracing", lambda: None)
    monkeypatch.setattr(
        main_module.workflow,
        "kickoff",
        lambda inputs, flow_run_id=None: {
            "topic": inputs["topic"],
            "report": "generated report",
        },
    )

    with TestClient(main_module.app) as client:
        response = client.post("/ask", json={"topic": "LangGraph"})

    assert response.status_code == 200
    assert response.json() == {
        "topic": "LangGraph",
        "report": "generated report",
    }


def test_ask_returns_http_500_on_workflow_failure(monkeypatch):
    main_module = _load_main_module(monkeypatch)
    monkeypatch.setattr(main_module, "initialize_langsmith_tracing", lambda: None)

    def _raise_failure(inputs, flow_run_id=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module.workflow, "kickoff", _raise_failure)

    with TestClient(main_module.app) as client:
        response = client.post("/ask", json={"topic": "LangGraph"})

    assert response.status_code == 500
    assert "Workflow execution failed." in response.json()["detail"]
