from __future__ import annotations

import os

import latest_ai_development.tracking as tracking_module


def test_initialize_langsmith_tracing_is_idempotent(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    monkeypatch.setattr(tracking_module, "_LANGSMITH_INITIALIZED", False)

    tracking_module.initialize_langsmith_tracing()
    tracking_module.initialize_langsmith_tracing()

    assert os.environ["LANGSMITH_API_KEY"] == "test-key"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "test-project"
