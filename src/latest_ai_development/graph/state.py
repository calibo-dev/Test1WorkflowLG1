from __future__ import annotations

from copy import deepcopy
from operator import add
from typing import Annotated, Any, NotRequired, Required, TypedDict

from langgraph.graph.message import add_messages

from latest_ai_development.config.settings import get_graph_config


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge mapping state without mutating either node update."""
    return {**left, **right}


_SCALAR_TYPES: dict[str, Any] = {
    "any": Any,
    "string": str,
    "str": str,
    "integer": int,
    "int": int,
    "float": float,
    "number": float,
    "boolean": bool,
    "bool": bool,
    "object": dict[str, Any],
    "dict": dict[str, Any],
}
_REDUCERS = {"append": add, "messages": add_messages, "merge": merge_dicts}


def _resolve_type(type_config: Any, *, field_name: str) -> Any:
    if isinstance(type_config, str):
        normalized = type_config.strip().lower()
        if normalized.startswith("list[") and normalized.endswith("]"):
            return list[_resolve_type(normalized[5:-1], field_name=field_name)]  # type: ignore[misc]
        if normalized in {"list", "array"}:
            return list[Any]
        resolved = _SCALAR_TYPES.get(normalized)
        if resolved is not None:
            return resolved
    elif isinstance(type_config, dict):
        kind = str(type_config.get("type", "")).strip().lower()
        if kind in {"list", "array"}:
            return list[  # type: ignore[misc]
                _resolve_type(type_config.get("items", "any"), field_name=field_name)
            ]
        return _resolve_type(kind, field_name=field_name)
    raise ValueError(f"State field '{field_name}' has unsupported type {type_config!r}")


def _field_annotation(field_name: str, field_config: Any) -> Any:
    if isinstance(field_config, str):
        field_config = {"type": field_config}
    if not isinstance(field_config, dict):
        raise ValueError(f"State field '{field_name}' configuration must be a mapping")

    annotation = _resolve_type(field_config, field_name=field_name)
    if field_config.get("nullable", False):
        annotation = annotation | None

    reducer_name = field_config.get("reducer")
    if reducer_name is not None:
        reducer = _REDUCERS.get(str(reducer_name).strip().lower())
        if reducer is None:
            raise ValueError(
                f"State field '{field_name}' uses unsupported reducer '{reducer_name}'"
            )
        annotation = Annotated[annotation, reducer]
    return annotation


def _state_sections() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state_config = get_graph_config().get("state")
    if not isinstance(state_config, dict):
        raise ValueError("graph_config.yaml must define a 'state' mapping")

    sections: list[dict[str, Any]] = []
    for section_name in ("input", "internal", "output"):
        section = state_config.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"graph_config.yaml state.{section_name} must be a mapping")
        sections.append(section)
    if not sections[0]:
        raise ValueError("graph_config.yaml state.input must define at least one field")
    if not sections[2]:
        raise ValueError("graph_config.yaml state.output must define at least one field")
    return sections[0], sections[1], sections[2]


def _annotations(fields: dict[str, Any], *, input_section: bool = False) -> dict[str, Any]:
    annotations: dict[str, Any] = {}
    for field_name, field_config in fields.items():
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError("State field names must be non-empty strings")
        annotation = _field_annotation(field_name, field_config)
        if input_section:
            required = not isinstance(field_config, dict) or field_config.get("required", True)
            annotation = Required[annotation] if required else NotRequired[annotation]
        annotations[field_name] = annotation
    return annotations


def _validate_duplicate_types(*sections: dict[str, Any]) -> None:
    resolved: dict[str, Any] = {}
    for section in sections:
        for field_name, field_config in section.items():
            annotation = _field_annotation(field_name, field_config)
            existing = resolved.get(field_name)
            if existing is not None and existing != annotation:
                raise ValueError(
                    f"State field '{field_name}' must use the same type and reducer "
                    "in every section"
                )
            resolved[field_name] = annotation


def _configured_defaults(*sections: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for section in sections:
        for field_name, field_config in section.items():
            if isinstance(field_config, dict) and "default" in field_config:
                defaults[field_name] = field_config["default"]
    return defaults


_INPUT_CONFIG, _INTERNAL_CONFIG, _OUTPUT_CONFIG = _state_sections()
_validate_duplicate_types(_INPUT_CONFIG, _INTERNAL_CONFIG, _OUTPUT_CONFIG)

_INPUT_ANNOTATIONS = _annotations(_INPUT_CONFIG, input_section=True)
_OUTPUT_ANNOTATIONS = _annotations(_OUTPUT_CONFIG)
_STATE_ANNOTATIONS = {
    **_annotations(_INPUT_CONFIG),
    **_annotations(_INTERNAL_CONFIG),
    **_OUTPUT_ANNOTATIONS,
}

InputState = TypedDict("InputState", _INPUT_ANNOTATIONS, total=False)  # type: ignore[misc]
OutputState = TypedDict("OutputState", _OUTPUT_ANNOTATIONS, total=False)  # type: ignore[misc]
State = TypedDict("State", _STATE_ANNOTATIONS, total=False)  # type: ignore[misc]

INPUT_FIELDS = tuple(_INPUT_CONFIG)
REQUIRED_INPUT_FIELDS = tuple(
    field_name
    for field_name, field_config in _INPUT_CONFIG.items()
    if not isinstance(field_config, dict) or field_config.get("required", True)
)
OUTPUT_FIELDS = tuple(_OUTPUT_CONFIG)
_STATE_DEFAULTS = _configured_defaults(_INPUT_CONFIG, _INTERNAL_CONFIG, _OUTPUT_CONFIG)


def state_defaults() -> dict[str, Any]:
    """Return an isolated copy of configured state defaults for one invocation."""
    return deepcopy(_STATE_DEFAULTS)
