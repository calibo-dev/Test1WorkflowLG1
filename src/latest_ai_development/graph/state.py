from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict


class InputState(TypedDict):
    """Public input accepted by the workflow."""

    user_input: str


class OutputState(TypedDict, total=False):
    """Public output returned by the workflow."""

    final_response: str
    errors: list[str]


class State(InputState, total=False):
    """Internal workflow state; nodes return partial updates to these fields."""

    errors: Annotated[list[str], add]
    validation_status: str
    normalized_feedback: str
    sentiment: str
    confidence: str
    sentiment_reason: str
    urgency_score: int
    policy_risk_score: int
    draft_response: str
    policy_guidance: str
    human_decision: str
    final_response: str
