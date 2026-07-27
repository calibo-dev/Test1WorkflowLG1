from __future__ import annotations

import os

import latest_ai_development.tracking as tracking_module


def test_initialize_langsmith_tracing_is_idempotent(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    monkeypatch.setattr(tracking_module, "_LANGSMITH_INITIALIZED", False)

    tracking_module.initialize_langsmith_tracing()
    tracking_module.initialize_langsmith_tracing()

    assert os.environ["LANGSMITH_API_KEY"] == "test-key"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "test-project"


def test_initialize_langsmith_tracing_is_disabled_when_flag_is_false(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY_SECRET", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(tracking_module, "_LANGSMITH_INITIALIZED", False)

    tracking_module.initialize_langsmith_tracing()

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert tracking_module._LANGSMITH_INITIALIZED is False


def test_initialize_langsmith_tracing_requires_key_when_enabled(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY_SECRET", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(tracking_module, "_LANGSMITH_INITIALIZED", False)
    monkeypatch.setattr(tracking_module, "_resolve_langsmith_api_key", lambda: None)

    try:
        tracking_module.initialize_langsmith_tracing()
    except ValueError as exc:
        assert "LANGSMITH_TRACING is enabled" in str(exc)
    else:
        raise AssertionError("Expected ValueError when LANGSMITH_TRACING=true without credentials")
