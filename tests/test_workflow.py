from __future__ import annotations

from pathlib import Path

from latest_ai_development.workflow import LatestAiDevelopmentWorkflow


class _FakeGraph:
    def invoke(self, state):
        return {
            **state,
            "report": "generated report",
        }


def test_workflow_kickoff_writes_report_to_configured_output_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORT_OUTPUT_FILE", "workflow-report.md")
    monkeypatch.setattr(
        "latest_ai_development.workflow.GraphBuilder.build",
        staticmethod(lambda: _FakeGraph()),
    )
    monkeypatch.setattr(
        "latest_ai_development.workflow.get_graph_config",
        lambda: {
            "entry_point": "validate_input",
            "final_output_field": "report",
            "execution": {"save_output": True},
            "nodes": {},
            "edges": [],
            "conditional_edges": [],
        },
    )

    workflow = LatestAiDevelopmentWorkflow()
    result = workflow.kickoff({"topic": "LangGraph"})

    assert result["report"] == "generated report"
    assert workflow.settings.report_output_path.name == "workflow-report.md"
    assert workflow.settings.report_output_path.read_text(encoding="utf-8") == "generated report"


def test_workflow_kickoff_does_not_write_report_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORT_OUTPUT_FILE", "workflow-report.md")
    monkeypatch.setattr(
        "latest_ai_development.workflow.GraphBuilder.build",
        staticmethod(lambda: _FakeGraph()),
    )
    monkeypatch.setattr(
        "latest_ai_development.workflow.get_graph_config",
        lambda: {
            "entry_point": "validate_input",
            "final_output_field": "report",
            "execution": {"save_output": False},
            "nodes": {},
            "edges": [],
            "conditional_edges": [],
        },
    )

    workflow = LatestAiDevelopmentWorkflow()
    result = workflow.kickoff({"topic": "LangGraph"})

    assert result["report"] == "generated report"
    assert not workflow.settings.report_output_path.exists()


def test_workflow_kickoff_logs_warning_when_report_write_fails(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORT_OUTPUT_FILE", "workflow-report.md")
    monkeypatch.setattr(
        "latest_ai_development.workflow.GraphBuilder.build",
        staticmethod(lambda: _FakeGraph()),
    )
    monkeypatch.setattr(
        "latest_ai_development.workflow.get_graph_config",
        lambda: {
            "entry_point": "validate_input",
            "final_output_field": "report",
            "execution": {"save_output": True},
            "nodes": {},
            "edges": [],
            "conditional_edges": [],
        },
    )

    def _raise_os_error(self: Path, data: str, encoding: str = "utf-8", **_: object):
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "write_text", _raise_os_error)

    workflow = LatestAiDevelopmentWorkflow()
    result = workflow.kickoff({"topic": "LangGraph"})

    assert result["report"] == "generated report"
    assert "Report save failed" in caplog.text
