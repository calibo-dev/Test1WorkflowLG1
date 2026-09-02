from __future__ import annotations

from importlib import import_module

from latest_ai_development.config.settings import (
    get_graph_config,
    get_models_config,
    get_settings,
    normalize_provider_name,
)
from latest_ai_development.llm.model_settings import get_model_settings


def _validate_provider_model_section(
    providers: dict[str, object],
    *,
    provider_name: str,
    section_name: str,
    section_cfg: object,
) -> None:
    if not isinstance(section_cfg, dict):
        raise ValueError(
            f"Provider '{provider_name}' {section_name} must be a mapping in models.yaml"
        )

    model_name = section_cfg.get("chat_model")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            f"Provider '{provider_name}' {section_name} must define chat_model in models.yaml"
        )

    fallback_provider = normalize_provider_name(section_cfg.get("provider") or provider_name)
    if fallback_provider not in providers:
        raise ValueError(
            f"Provider '{provider_name}' {section_name} references unknown provider '{fallback_provider}'"
        )


def _resolve_reference(reference: str):
    module_path, separator, attribute_path = reference.partition(":")
    if not separator or not module_path or not attribute_path:
        raise ValueError(
            f"Callable reference '{reference}' must use '<module>:<attribute>' format"
        )

    module = import_module(module_path)
    resolved = module
    for attribute_name in attribute_path.split("."):
        resolved = getattr(resolved, attribute_name)

    if not callable(resolved):
        raise TypeError(f"Resolved reference '{reference}' is not callable")

    return resolved


def _validate_graph_config() -> None:
    graph_cfg = get_graph_config()

    nodes = graph_cfg.get("nodes", {})
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("graph_config.yaml must define a non-empty 'nodes' mapping")

    entry_point = graph_cfg.get("entry_point")
    if not isinstance(entry_point, str) or entry_point not in nodes:
        raise ValueError("graph_config.yaml entry_point must reference a defined node")

    final_output_field = graph_cfg.get("final_output_field")
    if not isinstance(final_output_field, str) or not final_output_field.strip():
        raise ValueError("graph_config.yaml must define a non-empty final_output_field")

    for node_id, node_cfg in nodes.items():
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("graph_config.yaml node ids must be non-empty strings")
        if not isinstance(node_cfg, dict):
            raise ValueError(f"graph_config.yaml node '{node_id}' must be a mapping")

        builder_reference = node_cfg.get("builder")
        if not isinstance(builder_reference, str) or not builder_reference.strip():
            raise ValueError(f"graph_config.yaml node '{node_id}' must define a builder")

        _resolve_reference(builder_reference)

    for edge_cfg in graph_cfg.get("edges", []):
        if not isinstance(edge_cfg, dict):
            raise ValueError("graph_config.yaml edges must contain mappings")

        source = edge_cfg.get("from")
        target = edge_cfg.get("to")
        if source not in nodes:
            raise ValueError(
                f"graph_config.yaml edge source '{source}' is not a defined node"
            )
        if target != "END" and target not in nodes:
            raise ValueError(
                f"graph_config.yaml edge target '{target}' is not a defined node"
            )

    for conditional_cfg in graph_cfg.get("conditional_edges", []):
        if not isinstance(conditional_cfg, dict):
            raise ValueError("graph_config.yaml conditional_edges must contain mappings")

        source = conditional_cfg.get("from")
        if source not in nodes:
            raise ValueError(
                f"graph_config.yaml conditional edge source '{source}' is not a defined node"
            )

        router_reference = conditional_cfg.get("router")
        if not isinstance(router_reference, str) or not router_reference.strip():
            raise ValueError(
                f"graph_config.yaml conditional edge for '{source}' must define a router"
            )
        _resolve_reference(router_reference)

        path_map = conditional_cfg.get("path_map")
        if not isinstance(path_map, dict) or not path_map:
            raise ValueError(
                f"graph_config.yaml conditional edge for '{source}' must define a non-empty path_map"
            )

        for target in path_map.values():
            if target != "END" and target not in nodes:
                raise ValueError(
                    f"graph_config.yaml conditional target '{target}' is not a defined node"
                )


def validate_configuration() -> None:
    settings = get_settings()
    models_cfg = get_models_config()

    providers = models_cfg.get("providers", {})
    if not isinstance(providers, dict) or not providers:
        raise ValueError("models.yaml must define at least one provider")

    default_provider = normalize_provider_name(models_cfg.get("default_provider"))
    if default_provider and default_provider not in providers:
        raise ValueError(
            f"default_provider '{default_provider}' is not defined in models.yaml providers"
        )

    for provider_name, provider_cfg in providers.items():
        if not isinstance(provider_cfg, dict) or not provider_cfg.get("chat_model"):
            raise ValueError(
                f"Provider '{provider_name}' must define 'chat_model' in models.yaml"
            )

        fallback_cfg = provider_cfg.get("fallback")
        if fallback_cfg:
            _validate_provider_model_section(
                providers,
                provider_name=provider_name,
                section_name="fallback",
                section_cfg=fallback_cfg,
            )

    env_provider = normalize_provider_name(settings.provider)
    if env_provider not in providers:
        raise ValueError(
            f"Environment PROVIDER '{env_provider}' not defined in models.yaml providers"
        )

    provider_cfg = providers.get(env_provider, {})
    if not isinstance(provider_cfg, dict) or not provider_cfg.get("chat_model"):
        raise ValueError(
            f"Provider '{env_provider}' must define 'chat_model' in models.yaml"
        )

    cloud_provider = settings.cloud_provider.strip().upper()
    if cloud_provider not in {"", "AWS", "AZURE"}:
        raise ValueError("CLOUD_PROVIDER must be empty, AWS, or AZURE.")

    if cloud_provider == "AZURE" and not settings.azure_key_vault_url.strip():
        raise ValueError(
            "AZURE_KEY_VAULT_URL is required when CLOUD_PROVIDER is set to AZURE."
        )

    get_model_settings(env_provider)
    _validate_graph_config()
