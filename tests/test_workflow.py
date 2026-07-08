from __future__ import annotations

from latest_ai_development.workflow import LatestAiDevelopmentWorkflow


class _FakeGraph:
    def invoke(self, state):
        return {
            **state,
            "report": "generated report",
        }


def test_workflow_kickoff_writes_report_to_configured_output(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("REPORT_OUTPUT_FILE", "workflow-report.md")
    monkeypatch.setattr(
        "latest_ai_development.workflow.GraphBuilder.build",
        staticmethod(lambda: _FakeGraph()),
    )

    workflow = LatestAiDevelopmentWorkflow()
    result = workflow.kickoff({"topic": "LangGraph"})

    assert result["report"] == "generated report"
    assert workflow.settings.report_output_path.name == "workflow-report.md"
    assert workflow.settings.report_output_path.read_text(encoding="utf-8") == "generated report"
