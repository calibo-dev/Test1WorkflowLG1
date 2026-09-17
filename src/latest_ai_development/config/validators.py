from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Any

from latest_ai_development.config.settings import (
    BASE_DIR,
    get_agents_config,
    get_graph_config,
    get_mcp_config,
    get_models_config,
    get_settings,
    get_tools_config,
    normalize_provider_name,
)
from latest_ai_development.graph.state import InputState, OutputState, State
from latest_ai_development.llm.model_settings import get_model_settings
from latest_ai_development.nodes.workflow_nodes import NODE_REGISTRY, ROUTER_REGISTRY


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
            f"Provider '{provider_name}' {section_name} references unknown provider "
            f"'{fallback_provider}'"
        )


def _resolve_reference(reference: str) -> Any:
    module_path, separator, attribute_path = reference.partition(":")
    if not separator or not module_path or not attribute_path:
        raise ValueError(f"Callable reference '{reference}' must use '<module>:<attribute>' format")

    resolved: Any = import_module(module_path)
    for attribute_name in attribute_path.split("."):
        resolved = getattr(resolved, attribute_name)
    if not callable(resolved):
        raise TypeError(f"Resolved reference '{reference}' is not callable")
    return resolved


def _require_string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list of strings")
    values = [str(item).strip() for item in value]
    if any(not item for item in values):
        raise ValueError(f"{field_name} must not contain empty values")
    return values


def _validate_tools_config() -> set[str]:
    tools = get_tools_config().get("tools")
    if not isinstance(tools, Mapping):
        raise ValueError("tools.yaml must define a 'tools' mapping")

    for tool_name, tool_config in tools.items():
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tools.yaml tool names must be non-empty strings")
        if not isinstance(tool_config, Mapping):
            raise ValueError(f"Tool '{tool_name}' configuration must be a mapping")
        factory = tool_config.get("factory")
        if not isinstance(factory, str) or not factory.strip():
            raise ValueError(f"Tool '{tool_name}' must define a factory reference")
        resolved_factory = _resolve_reference(factory)

        description = tool_config.get("description")
        if description is not None and (
            not isinstance(description, str) or not description.strip()
        ):
            raise ValueError(f"Tool '{tool_name}' description must be a non-empty string")

        connector = tool_config.get("connector")
        if connector is not None and not isinstance(connector, Mapping):
            raise ValueError(f"Tool '{tool_name}' connector must be a mapping")

        factory_kwargs = tool_config.get("factory_kwargs", {})
        if not isinstance(factory_kwargs, Mapping):
            raise ValueError(f"Tool '{tool_name}' factory_kwargs must be a mapping")
        try:
            signature(resolved_factory).bind(**dict(factory_kwargs))
        except TypeError as exc:
            raise ValueError(
                f"Tool '{tool_name}' factory_kwargs do not match factory "
                f"'{factory}': {exc}"
            ) from exc

        for field_name in ("tags", "usage_guidelines", "non_usage_guidelines"):
            if field_name in tool_config:
                _require_string_list(
                    tool_config[field_name],
                    field_name=f"Tool '{tool_name}' {field_name}",
                )

    return set(tools)


def _validate_mcp_config() -> set[str]:
    servers = get_mcp_config().get("servers")
    if not isinstance(servers, Mapping):
        raise ValueError("mcp.yaml must define a 'servers' mapping")

    for server_name, server_config in servers.items():
        if not isinstance(server_name, str) or not server_name.strip():
            raise ValueError("mcp.yaml server names must be non-empty strings")
        if not isinstance(server_config, Mapping):
            raise ValueError(f"MCP server '{server_name}' configuration must be a mapping")

        transport = str(server_config.get("transport", "")).strip().lower()
        if transport == "stdio":
            command = server_config.get("command")
            if not isinstance(command, str) or not command.strip():
                raise ValueError(f"MCP server '{server_name}' must define command for stdio")
            _require_string_list(
                server_config.get("args", []),
                field_name=f"MCP server '{server_name}' args",
            )
        elif transport in {"sse", "streamable_http"}:
            url = server_config.get("url")
            if not isinstance(url, str) or not url.strip():
                raise ValueError(f"MCP server '{server_name}' must define url for {transport}")
            headers = server_config.get("headers", {})
            if not isinstance(headers, Mapping):
                raise ValueError(f"MCP server '{server_name}' headers must be a mapping")
        else:
            raise ValueError(
                f"MCP server '{server_name}' transport must be stdio, sse, or streamable_http"
            )

    return set(servers)


def _validate_agents_config() -> None:
    config = get_agents_config()
    agents = config.get("agents")
    if not isinstance(agents, Mapping):
        raise ValueError("agents.yaml must define an 'agents' mapping")

    providers = get_models_config().get("providers", {})
    tool_names = _validate_tools_config()
    mcp_server_names = _validate_mcp_config()

    state_fields = set(InputState.__annotations__) | set(State.__annotations__)
    for agent_id, agent_config in agents.items():
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agents.yaml agent ids must be non-empty strings")
        if not isinstance(agent_config, Mapping):
            raise ValueError(f"Agent '{agent_id}' configuration must be a mapping")

        provider = normalize_provider_name(agent_config.get("provider"))
        if not provider or provider not in providers:
            raise ValueError(f"Agent '{agent_id}' references unknown provider '{provider}'")
        model = agent_config.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Agent '{agent_id}' must define a model name")

        prompt_value = agent_config.get("prompt")
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            raise ValueError(f"Agent '{agent_id}' must define a non-empty prompt")
        normalized_prompt = prompt_value.strip()
        looks_like_file = "\n" not in normalized_prompt and (
            normalized_prompt.startswith("prompts/")
            or normalized_prompt.endswith((".md", ".txt"))
        )
        if looks_like_file:
            prompt_path = Path(normalized_prompt)
            if not prompt_path.is_absolute():
                prompt_path = BASE_DIR / "src" / "latest_ai_development" / prompt_path
            if not prompt_path.is_file():
                raise ValueError(
                    f"Agent '{agent_id}' prompt file does not exist: {prompt_path}"
                )

        reads_state = _require_string_list(
            agent_config.get("reads_state", []),
            field_name=f"Agent '{agent_id}' reads_state",
        )
        writes_state = _require_string_list(
            agent_config.get("writes_state", []),
            field_name=f"Agent '{agent_id}' writes_state",
        )
        if not reads_state:
            raise ValueError(f"Agent '{agent_id}' must read at least one state field")
        if not writes_state:
            raise ValueError(f"Agent '{agent_id}' must write at least one state field")

        unknown_state_fields = (set(reads_state) | set(writes_state)) - state_fields
        if unknown_state_fields:
            raise ValueError(
                f"Agent '{agent_id}' references undefined state fields: "
                f"{sorted(unknown_state_fields)}"
            )

        output_field = agent_config.get("output_field")
        output_mapping = agent_config.get("output_mapping")
        if bool(output_field) == bool(output_mapping):
            raise ValueError(
                f"Agent '{agent_id}' must define exactly one of output_field or output_mapping"
            )
        if output_field and output_field not in writes_state:
            raise ValueError(f"Agent '{agent_id}' output_field must be declared in writes_state")
        if output_mapping:
            if not isinstance(output_mapping, Mapping) or not output_mapping:
                raise ValueError(f"Agent '{agent_id}' output_mapping must be a mapping")
            undeclared_outputs = set(output_mapping.values()) - set(writes_state)
            if undeclared_outputs:
                raise ValueError(
                    f"Agent '{agent_id}' output_mapping targets undeclared state fields: "
                    f"{sorted(undeclared_outputs)}"
                )

        assigned_tools = _require_string_list(
            agent_config.get("tools", []),
            field_name=f"Agent '{agent_id}' tools",
        )
        unknown_tools = sorted(set(assigned_tools) - tool_names)
        if unknown_tools:
            raise ValueError(f"Agent '{agent_id}' references undefined tools: {unknown_tools}")

        assigned_mcp_servers = _require_string_list(
            agent_config.get("mcp_servers", []),
            field_name=f"Agent '{agent_id}' mcp_servers",
        )
        unknown_mcp_servers = sorted(set(assigned_mcp_servers) - mcp_server_names)
        if unknown_mcp_servers:
            raise ValueError(
                f"Agent '{agent_id}' references undefined MCP servers: {unknown_mcp_servers}"
            )
        if assigned_mcp_servers:
            raise ValueError(
                f"Agent '{agent_id}' assigns MCP servers, but the MCP runtime adapter "
                "is not installed"
            )


def _validate_graph_config() -> None:
    graph_config = get_graph_config()
    agents = get_agents_config()["agents"]
    nodes = graph_config.get("nodes", {})
    if not isinstance(nodes, Mapping) or not nodes:
        raise ValueError("graph_config.yaml must define a non-empty 'nodes' mapping")

    node_ids = set(nodes)
    entry_point = graph_config.get("entry_point")
    if not isinstance(entry_point, str) or entry_point not in node_ids:
        raise ValueError("graph_config.yaml entry_point must reference a defined node")

    final_output_field = graph_config.get("final_output_field")
    if not isinstance(final_output_field, str) or not final_output_field.strip():
        raise ValueError("graph_config.yaml must define a non-empty final_output_field")
    if final_output_field not in OutputState.__annotations__:
        raise ValueError("graph_config.yaml final_output_field must be declared in OutputState")

    persistence = graph_config.get("persistence", {})
    if not isinstance(persistence, Mapping):
        raise ValueError("graph_config.yaml persistence must be a mapping")
    if persistence.get("enabled", False):
        backend = str(persistence.get("backend", "memory")).strip().lower()
        if backend != "memory":
            raise ValueError(f"Unsupported persistence backend '{backend}'")

    runtime_context = graph_config.get("runtime_context", {})
    if not isinstance(runtime_context, Mapping):
        raise ValueError("graph_config.yaml runtime_context must be a mapping")
    if not isinstance(runtime_context.get("defaults", {}), Mapping):
        raise ValueError("graph_config.yaml runtime_context defaults must be a mapping")

    end_nodes = _require_string_list(
        graph_config.get("end_nodes", []),
        field_name="graph_config.yaml end_nodes",
    )
    if not end_nodes or not set(end_nodes).issubset(node_ids):
        raise ValueError("graph_config.yaml end_nodes must reference defined nodes")

    command_sources: set[str] = set()
    command_destinations_by_source: dict[str, list[str]] = {}
    deterministic_node_ids: set[str] = set()
    runtime_override_node_ids: set[str] = set()
    for node_id, node_config in nodes.items():
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("graph_config.yaml node ids must be non-empty strings")
        if not isinstance(node_config, Mapping):
            raise ValueError(f"graph_config.yaml node '{node_id}' must be a mapping")

        execution_mode = node_config.get("execution_mode")
        if execution_mode not in {None, "sync", "async", "background"}:
            raise ValueError(
                f"Node '{node_id}' execution_mode must be sync, async, background, or null"
            )

        node_type = node_config.get("type")
        runtime_override = node_config.get("runtime_override") is True
        if runtime_override:
            runtime_override_node_ids.add(str(node_id))
        if node_type == "agent":
            if execution_mode == "background":
                raise ValueError(
                    f"Agent node '{node_id}' cannot use background execution directly; "
                    "use a deterministic dispatch node with acknowledgement/job outputs"
                )
            agent_id = node_config.get("agent")
            if agent_id not in agents:
                raise ValueError(
                    f"graph_config.yaml node '{node_id}' references unknown agent '{agent_id}'"
                )
            for field_name in ("reads_state", "writes_state"):
                node_values = _require_string_list(
                    node_config.get(field_name, []),
                    field_name=f"Node '{node_id}' {field_name}",
                )
                agent_values = _require_string_list(
                    agents[agent_id].get(field_name, []),
                    field_name=f"Agent '{agent_id}' {field_name}",
                )
                if node_values != agent_values:
                    raise ValueError(
                        f"Agent node '{node_id}' {field_name} must match agent "
                        f"'{agent_id}': expected={agent_values} actual={node_values}"
                    )
        elif node_type == "deterministic":
            deterministic_node_ids.add(str(node_id))
            operation = node_config.get("operation")
            if not isinstance(operation, Mapping):
                raise ValueError(
                    f"Deterministic node '{node_id}' must define operation metadata"
                )
            operation_type = operation.get("type")
            if not isinstance(operation_type, str) or not operation_type.strip():
                raise ValueError(
                    f"Deterministic node '{node_id}' operation.type must be a non-empty string"
                )
            description = operation.get("description")
            if description is not None and (
                not isinstance(description, str) or not description.strip()
            ):
                raise ValueError(
                    f"Deterministic node '{node_id}' operation.description must be a string"
                )
            parameters = operation.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise ValueError(
                    f"Deterministic node '{node_id}' operation.parameters must be a mapping"
                )
            reads_state = _require_string_list(
                node_config.get("reads_state", []),
                field_name=f"Node '{node_id}' reads_state",
            )
            writes_state = _require_string_list(
                node_config.get("writes_state", []),
                field_name=f"Node '{node_id}' writes_state",
            )
            if not writes_state:
                raise ValueError(f"Node '{node_id}' writes_state must not be empty")
            state_fields = (
                set(InputState.__annotations__)
                | set(State.__annotations__)
                | set(OutputState.__annotations__)
            )
            unknown_state_fields = (set(reads_state) | set(writes_state)) - state_fields
            if unknown_state_fields:
                raise ValueError(
                    f"Node '{node_id}' references undefined state fields: "
                    f"{sorted(unknown_state_fields)}"
                )
        else:
            raise ValueError(
                f"graph_config.yaml node '{node_id}' type must be agent or deterministic"
            )

        if "command_destinations" in node_config:
            destinations = _require_string_list(
                node_config["command_destinations"],
                field_name=f"Node '{node_id}' command_destinations",
            )
            if not destinations:
                raise ValueError(f"Node '{node_id}' command_destinations must not be empty")
            if len(destinations) != len(set(destinations)):
                raise ValueError(f"Node '{node_id}' command_destinations contains duplicates")
            unknown_destinations = {
                target for target in destinations if target != "END" and target not in node_ids
            }
            if unknown_destinations:
                raise ValueError(
                    f"Node '{node_id}' command_destinations reference undefined nodes: "
                    f"{sorted(unknown_destinations)}"
                )
            command_sources.add(str(node_id))
            command_destinations_by_source[str(node_id)] = destinations

    registered_node_ids = set(NODE_REGISTRY)
    expected_registered_node_ids = (
        deterministic_node_ids | runtime_override_node_ids
    )
    if registered_node_ids != expected_registered_node_ids:
        raise ValueError(
            "NODE_REGISTRY keys must exactly match configured deterministic and "
            "runtime-override nodes: "
            f"expected={sorted(expected_registered_node_ids)} "
            f"actual={sorted(registered_node_ids)}"
        )
    non_callable_nodes = sorted(
        node_id for node_id, node_callable in NODE_REGISTRY.items() if not callable(node_callable)
    )
    if non_callable_nodes:
        raise TypeError(f"NODE_REGISTRY contains non-callable nodes: {non_callable_nodes}")

    adjacency: dict[str, set[str]] = defaultdict(set)
    reverse_adjacency: dict[str, set[str]] = defaultdict(set)
    direct_sources: set[str] = set()
    direct_pairs: set[tuple[str, str]] = set()
    for edge_config in graph_config.get("edges", []):
        if not isinstance(edge_config, Mapping):
            raise ValueError("graph_config.yaml edges must contain mappings")
        source = edge_config.get("from")
        target = edge_config.get("to")
        if source not in node_ids:
            raise ValueError(f"Graph edge source '{source}' is not a defined node")
        if target != "END" and target not in node_ids:
            raise ValueError(f"Graph edge target '{target}' is not a defined node")
        pair = (str(source), str(target))
        if pair in direct_pairs:
            raise ValueError(f"Duplicate graph edge: {source} -> {target}")
        direct_pairs.add(pair)
        direct_sources.add(str(source))
        adjacency[str(source)].add(str(target))
        reverse_adjacency[str(target)].add(str(source))

    for source, destinations in command_destinations_by_source.items():
        for target in destinations:
            adjacency[source].add(target)
            reverse_adjacency[target].add(source)

    conditional_sources: set[str] = set()
    for conditional_config in graph_config.get("conditional_edges", []):
        if not isinstance(conditional_config, Mapping):
            raise ValueError("graph_config.yaml conditional_edges must contain mappings")
        source = conditional_config.get("from")
        if source not in node_ids:
            raise ValueError(f"Conditional source '{source}' is not a defined node")
        if source in conditional_sources:
            raise ValueError(f"Node '{source}' has more than one conditional edge block")
        conditional_sources.add(str(source))

        router = conditional_config.get("router")
        decision_field = conditional_config.get("decision_field")
        send_destinations = conditional_config.get("send_destinations")
        if send_destinations is not None:
            if not router or decision_field or "path_map" in conditional_config:
                raise ValueError(
                    f"Dynamic Send edge '{source}' must define router and send_destinations "
                    "without decision_field or path_map"
                )
            destinations = _require_string_list(
                send_destinations,
                field_name=f"Dynamic Send edge '{source}' send_destinations",
            )
            if not destinations:
                raise ValueError(
                    f"Dynamic Send edge '{source}' send_destinations must not be empty"
                )
            unknown_destinations = set(destinations) - node_ids
            if unknown_destinations:
                raise ValueError(
                    f"Dynamic Send edge '{source}' references undefined nodes: "
                    f"{sorted(unknown_destinations)}"
                )
            router_id = str(router)
            if router_id not in ROUTER_REGISTRY or not callable(ROUTER_REGISTRY[router_id]):
                raise ValueError(
                    f"Dynamic Send edge '{source}' references unknown router '{router_id}'"
                )
            for target in destinations:
                adjacency[str(source)].add(target)
                reverse_adjacency[target].add(str(source))
            continue

        if bool(router) == bool(decision_field):
            raise ValueError(
                f"Conditional edge '{source}' must define exactly one of router or decision_field"
            )
        if router:
            router_id = str(router)
            if router_id not in ROUTER_REGISTRY or not callable(ROUTER_REGISTRY[router_id]):
                raise ValueError(
                    f"Conditional edge '{source}' references unknown router '{router_id}'"
                )
        elif decision_field not in (set(InputState.__annotations__) | set(State.__annotations__)):
            raise ValueError(
                f"Conditional edge '{source}' references undefined state field '{decision_field}'"
            )

        path_map = conditional_config.get("path_map")
        if not isinstance(path_map, Mapping) or not path_map:
            raise ValueError(f"Conditional edge '{source}' must define a non-empty path_map")
        for route, target in path_map.items():
            if not isinstance(route, str) or not route.strip():
                raise ValueError(f"Conditional edge '{source}' has an empty route label")
            if target != "END" and target not in node_ids:
                raise ValueError(
                    f"Conditional target '{target}' from '{source}' is not a defined node"
                )
            adjacency[str(source)].add(str(target))
            reverse_adjacency[str(target)].add(str(source))

        default_route = conditional_config.get("default_route")
        if default_route and default_route not in path_map:
            raise ValueError(f"Conditional edge '{source}' default_route must exist in path_map")

    configured_router_ids = {
        str(config.get("router"))
        for config in graph_config.get("conditional_edges", [])
        if isinstance(config, Mapping) and config.get("router")
    }
    if set(ROUTER_REGISTRY) != configured_router_ids:
        raise ValueError(
            "ROUTER_REGISTRY keys must exactly match configured routers: "
            f"expected={sorted(configured_router_ids)} actual={sorted(ROUTER_REGISTRY)}"
        )

    mixed_sources = (
        (direct_sources & conditional_sources)
        | (direct_sources & command_sources)
        | (conditional_sources & command_sources)
    )
    if mixed_sources:
        raise ValueError(
            "Nodes cannot mix direct and conditional outgoing edges or Command routes: "
            f"{sorted(mixed_sources)}"
        )

    reachable = _walk_graph(str(entry_point), adjacency)
    unreachable = node_ids - reachable
    if unreachable:
        raise ValueError(f"Graph contains unreachable nodes: {sorted(unreachable)}")

    can_reach_end = _walk_graph("END", reverse_adjacency)
    dead_ends = node_ids - can_reach_end
    if dead_ends:
        raise ValueError(f"Graph nodes have no path to END: {sorted(dead_ends)}")

    missing_end_edges = [node_id for node_id in end_nodes if "END" not in adjacency[node_id]]
    if missing_end_edges:
        raise ValueError(f"Configured end_nodes must route to END: {sorted(missing_end_edges)}")

    invalid_end_edges = [node_id for node_id in end_nodes if adjacency[node_id] != {"END"}]
    if invalid_end_edges:
        raise ValueError(
            f"Configured end_nodes cannot route anywhere except END: {sorted(invalid_end_edges)}"
        )


def _walk_graph(start: str, adjacency: Mapping[str, set[str]]) -> set[str]:
    visited: set[str] = set()
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        queue.extend(adjacency.get(current, set()) - visited)
    return visited


def validate_configuration() -> None:
    settings = get_settings()
    models_config = get_models_config()
    providers = models_config.get("providers", {})
    if not isinstance(providers, dict) or not providers:
        raise ValueError("models.yaml must define at least one provider")

    for provider_name, provider_config in providers.items():
        if not isinstance(provider_config, dict):
            raise ValueError(f"Provider '{provider_name}' must be a mapping in models.yaml")
        fallback_config = provider_config.get("fallback")
        if fallback_config:
            _validate_provider_model_section(
                providers,
                provider_name=provider_name,
                section_name="fallback",
                section_cfg=fallback_config,
            )

    cloud_provider = settings.cloud_provider.strip().upper()
    if cloud_provider not in {"", "AWS", "AZURE"}:
        raise ValueError("CLOUD_PROVIDER must be empty, AWS, or AZURE.")
    if cloud_provider == "AZURE" and not settings.azure_key_vault_url.strip():
        raise ValueError("AZURE_KEY_VAULT_URL is required when CLOUD_PROVIDER is set to AZURE.")

    _validate_agents_config()
    for agent_config in get_agents_config()["agents"].values():
        get_model_settings(
            str(agent_config["provider"]),
            model_name=str(agent_config["model"]),
        )
    _validate_graph_config()
