from __future__ import annotations

from fastapi.testclient import TestClient

import latest_ai_development.main as main_module


def test_ask_returns_report(monkeypatch):
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
    monkeypatch.setattr(main_module, "initialize_langsmith_tracing", lambda: None)

    def _raise_failure(inputs, flow_run_id=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module.workflow, "kickoff", _raise_failure)

    with TestClient(main_module.app) as client:
        response = client.post("/ask", json={"topic": "LangGraph"})

    assert response.status_code == 500
    assert "Workflow execution failed." in response.json()["detail"]
