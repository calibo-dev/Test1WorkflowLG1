from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from langgraph.types import interrupt

from latest_ai_development.graph.state import State

RouteName = Literal["appreciation", "clarification", "service_recovery", "human_review"]


def _parse_confidence(raw_confidence: Any) -> float:
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return 0.0
    if confidence > 1:
        confidence /= 100
    return max(0.0, min(confidence, 1.0))


def _keyword_score(text: str, weighted_terms: Mapping[str, int]) -> int:
    normalized = text.lower()
    return sum(weight for term, weight in weighted_terms.items() if term in normalized)


def route_after_evaluation(state: State) -> RouteName:
    """Route using classifier confidence plus deterministic risk scores."""
    confidence = _parse_confidence(state.get("confidence"))
    sentiment = str(state.get("sentiment", "unknown")).strip().lower()
    urgency_score = int(state.get("urgency_score", 0))
    policy_risk_score = int(state.get("policy_risk_score", 0))

    if policy_risk_score >= 70 or urgency_score >= 80:
        return "human_review"
    if confidence < 0.6 or sentiment not in {"positive", "negative"}:
        return "clarification"
    if sentiment == "negative" or urgency_score >= 35:
        return "service_recovery"
    return "appreciation"


class FeedbackOrchestrationNodeBuilder:
    """Build deterministic preparation, scoring, HITL, and finalization nodes."""

    URGENCY_TERMS = {
        "urgent": 35,
        "immediately": 30,
        "refund": 25,
        "cancel": 20,
        "broken": 20,
        "never arrived": 35,
    }
    POLICY_RISK_TERMS = {
        "unsafe": 70,
        "injury": 80,
        "legal": 70,
        "fraud": 75,
        "chargeback": 50,
        "privacy": 60,
    }

    @staticmethod
    def build_prepare_feedback() -> Callable[[State], dict[str, Any]]:
        def prepare_feedback_node(state: State, /) -> dict[str, Any]:
            user_input = state.get("user_input", "")
            if not isinstance(user_input, str) or not user_input.strip():
                return {
                    "errors": ["user_input is required"],
                    "validation_status": "invalid",
                }
            normalized = " ".join(user_input.split())
            if len(normalized) < 4:
                return {
                    "errors": ["Please provide more detail about your experience."],
                    "normalized_feedback": normalized,
                    "validation_status": "invalid",
                }
            return {
                "normalized_feedback": normalized,
                "validation_status": "valid",
            }

        return prepare_feedback_node

    @staticmethod
    def build_evaluate_feedback() -> Callable[[State], dict[str, Any]]:
        def evaluate_feedback_node(state: State, /) -> dict[str, Any]:
            feedback = str(state.get("normalized_feedback", ""))
            sentiment = str(state.get("sentiment", "unknown")).lower()
            confidence = _parse_confidence(state.get("confidence"))

            urgency_score = _keyword_score(
                feedback,
                FeedbackOrchestrationNodeBuilder.URGENCY_TERMS,
            )
            policy_risk_score = _keyword_score(
                feedback,
                FeedbackOrchestrationNodeBuilder.POLICY_RISK_TERMS,
            )
            if sentiment == "negative":
                urgency_score += 15
            if confidence < 0.6:
                policy_risk_score += 20

            return {
                "urgency_score": min(urgency_score, 100),
                "policy_risk_score": min(policy_risk_score, 100),
            }

        return evaluate_feedback_node

    @staticmethod
    def build_human_review() -> Callable[[State], dict[str, Any]]:
        def human_review_node(state: State, /) -> dict[str, Any]:
            review = interrupt(
                {
                    "question": "Review this high-risk customer feedback and provide a response.",
                    "feedback": state.get("normalized_feedback", ""),
                    "sentiment": state.get("sentiment", "unknown"),
                    "urgency_score": state.get("urgency_score", 0),
                    "policy_risk_score": state.get("policy_risk_score", 0),
                }
            )
            if isinstance(review, Mapping):
                decision = str(review.get("decision", "reviewed"))
                response = str(review.get("response", ""))
            else:
                decision = "reviewed"
                response = str(review)
            return {"human_decision": decision, "final_response": response}

        return human_review_node

    @staticmethod
    def build_finalize_output() -> Callable[[State], dict[str, Any]]:
        def finalize_output_node(state: State, /) -> dict[str, Any]:
            errors = list(state.get("errors", []))
            final_response = str(state.get("final_response", "")).strip()
            if errors and not final_response:
                final_response = f"Error: {'; '.join(errors)}"
            if not final_response:
                final_response = "We could not prepare a response. Please contact customer service."
            return {"final_response": final_response}

        return finalize_output_node
