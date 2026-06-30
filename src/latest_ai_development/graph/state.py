from typing import TypedDict


class State(TypedDict):
    """
    Shared LangGraph state passed between workflow nodes.

    query:
        Original user input for the workflow.
    research_result:
        Intermediate output from the research node.
    report:
        Final output produced by the reporting node.
    """

    query: str
    research_result: str
    report: str