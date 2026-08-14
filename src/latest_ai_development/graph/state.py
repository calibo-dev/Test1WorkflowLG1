from typing import TypedDict

class AgentConfig(TypedDict, total=False):
    """
    Internal agent configuration carried in workflow state.
    """

    id: str
    name: str
    enabled: bool
    role: str
    tools: list[str]


class State(TypedDict):
    """
    Shared LangGraph state passed between workflow nodes.

    topic:
        Original user input for the workflow.
    agent_plan:
        Internal agent definitions used by the POC workflow.
    errors:
        Validation or orchestration errors collected during execution.
    research_result:
        Intermediate output from the research node.
    report:
        Final output produced by the reporting node.
    """

    topic: str
    agent_plan: list[AgentConfig]
    errors: list[str]
    research_result: str
    report: str