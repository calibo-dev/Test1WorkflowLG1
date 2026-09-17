# LangGraph Template Refactor Approach

## Objective

Refactor the LangGraph starter template so that configuration describes the
workflow and stable runtime code executes it. Code generation should configure a
workflow instead of repeatedly rewriting the graph runtime, API entrypoints,
model setup, tracking, and tool infrastructure.

The expected outcome is:

- An initial workflow with four agents and two tools changes approximately 6-11
  files, depending on connector and dependency requirements.
- Adding one agent to an existing workflow changes approximately 3-4 files.
- Direct and conditional routing are represented explicitly in configuration.
- Agent and tool wiring use reusable template-owned runtime components.
- Generated projects pass dependency sync, compilation, tests, lint, type checks,
  and graph-contract validation before completion.

## Current Main Baseline

The baseline reviewed for this proposal is `main` at commit `33bc0c5`.

| Check | Result |
| --- | --- |
| Working tree | Clean and aligned with `origin/main` |
| `uv sync --all-extras` | Passed |
| Unit tests | 30 passed |
| Python compilation | Passed |
| Ruff | 26 issues |
| mypy | 9 errors |

### Current template findings

| Priority | Finding | Proposed resolution |
| --- | --- | --- |
| Critical | `graph_config.yaml` references `research_node_builder.py` and `reporting_node_builder.py`, but both files identify themselves as deprecated. Replacement `primary_*` builders also remain in the template. | Choose one canonical implementation and remove the obsolete alternatives. |
| Critical | Research and reporting builders create an unused agent during build and another agent during execution. The reporting execution path also omits fallback middleware. | Move common model, fallback, tool, invocation, parsing, and logging behavior into one reusable agent-node builder. |
| High | Each conditional branch requires a custom Python router function. | Add a generic state-field router driven by `decision_field` and `path_map`. Keep custom routers as an optional escape hatch. |
| High | Tool assignment is carried in runtime `agent_plan`, separately from configured graph nodes. | Declare tools in agent/node configuration and resolve them through a stable `get_tools_for_node(node_id)` hook. |
| High | `graph/state.py` is coupled to the research/reporting sample. | Introduce a stable state envelope and extend it only when a workflow needs specialized typed fields. |
| Medium | Every agent requires a dedicated Python builder and prompt function. | Use a reusable agent builder with prompt references from configuration. |
| Medium | `GraphBuilder` contains competing node wrappers, while the richer logging wrapper is not used by graph compilation. | Keep one wrapper that consistently applies tracing, logging, error reporting, and state summaries. |
| Medium | Configuration validation resolves references but does not fully validate reachability, terminal paths, duplicate routes, or mixed routing mechanisms. | Add graph-contract validation to the template. |
| Baseline | Ruff and mypy fail, and tests attempt external LangSmith communication. | Make static checks pass and disable external tracing during unit tests. |

## Cross-Platform Principle

The proposed LangGraph design follows the same separation already recommended
for CrewAI and LangChain.

| Platform | Agent configuration | Sequence and routing | Stable runtime |
| --- | --- | --- | --- |
| CrewAI | `agents.yaml`, `tasks.yaml` | Kickoff and process configuration | `crew.py` |
| LangChain | `config.yaml` | Workflow stages and transitions | `agents.py`, `workflow.py` |
| LangGraph | `agents.yaml` | Direct and conditional edges in `graph_config.yaml` | Generic graph builder and direct callable registries |

LangGraph needs additional graph-specific configuration because it explicitly
models state, deterministic nodes, conditional routing, retries, entry points,
and terminal paths. Those differences should be represented as data rather than
new handwritten runtime plumbing for every generated workflow.

## Target Template Structure

| File | Responsibility | Codegen ownership |
| --- | --- | --- |
| `config/agents.yaml` | Agent identity, role, prompt, provider, model, tool/MCP assignments, state reads, and state writes | Generated or updated |
| `config/tools.yaml` | Python tool names and factory references | Generated or updated when tools change |
| `config/mcp.yaml` | MCP server transports and connection metadata | Generated or updated when MCP servers change |
| `config/graph_config.yaml` | Entry/end nodes, deterministic operation metadata, agent nodes, direct edges, conditional routes, and tracing labels | Generated or updated |
| `graph/state.py` | Explicit typed public input, internal state, reducers, defaults, and public output | Generated or updated |
| `nodes/agent_node_builder.py` | Generic model creation, fallback middleware, tools, invocation, parsing, and logging | Template-owned and read-only to codegen |
| `nodes/workflow_nodes.py` | Direct deterministic functions and exact `NODE_REGISTRY`/`ROUTER_REGISTRY` mappings | Generated or updated when required |
| `graph/graph_builder.py` | Compile the configured `StateGraph`, nodes, direct edges, conditional edges, `START`, and `END` | Template-owned and read-only to codegen |
| `tools/tool_registry.py` | Generic tool registration and lookup by node or agent ID | Template-owned and read-only to codegen |
| `tools/<tool>.py` | Connector-specific tool implementation | Added only for assigned tools |
| `workflow.py` | Invoke the compiled graph | Normally unchanged |
| `main.py` | Application and API entrypoint | Normally unchanged |
| `pyproject.toml`, `uv.lock` | Direct runtime dependencies and resolved lock data | Updated only when dependencies change |
| `README.md`, `.env.example` | Workflow usage and required environment variables | Updated when relevant |
| `tests/test_template.py` | Generic configuration, graph, node, tool, and state contract checks | Template-owned and read-only to codegen |

## Configuration Contract

### Agent configuration

```yaml
agents:
  sentiment_analyzer:
    role: Customer Feedback Analyst
    prompt: prompts/sentiment_analyzer.md
    provider: openai
    model: gpt-4.1-mini
    reads_state:
      - customer_message
    writes_state:
      - sentiment
      - confidence
      - issue_summary
    tools: []
    mcp_servers: []

  support_response_agent:
    role: Customer Support Response Specialist
    prompt: prompts/support_response.md
    provider: openai
    model: gpt-4.1-mini
    reads_state:
      - customer_message
      - sentiment
      - issue_summary
    writes_state:
      - response
    tools:
      - customer_policy_tool
      - customer_support_tool
    mcp_servers: []
```

Agent configuration should contain everything needed by the reusable agent-node
builder. Adding an ordinary agent should not require another copy of model and
fallback setup code.

### Graph configuration

```yaml
graph:
  entry_node: validate_input
  end_nodes:
    - format_response
  final_output_field: response
  nodes:
    - node_id: validate_input
      node_type: deterministic
      operation_type: input_validation
      reads_state: [request]
      writes_state: [validation_route]
    - node_id: sentiment_analyzer
      node_type: agent
      agent_name: sentiment_analyzer
      reads_state: [request]
      writes_state: [sentiment]
    - node_id: route_sentiment
      node_type: deterministic
      operation_type: routing
      reads_state: [sentiment]
      writes_state: [sentiment_route]
    - node_id: positive_response
      node_type: deterministic
      operation_type: response_generation
      reads_state: [request]
      writes_state: [response]
    - node_id: support_response
      node_type: agent
      agent_name: support_response_agent
      reads_state: [request, sentiment]
      writes_state: [response]
    - node_id: clarification
      node_type: agent
      agent_name: clarification_agent
      reads_state: [request]
      writes_state: [response]
    - node_id: format_response
      node_type: deterministic
      operation_type: formatting
      reads_state: [response]
      writes_state: [response]
  edges:
    - source: validate_input
      target: sentiment_analyzer
    - source: sentiment_analyzer
      target: route_sentiment
    - source: positive_response
      target: format_response
    - source: support_response
      target: format_response
    - source: clarification
      target: format_response
  conditional_edges:
    - source: route_sentiment
      decision_field: sentiment_route
      routes:
        - {route_key: positive, target: positive_response}
        - {route_key: negative, target: support_response}
        - {route_key: unclear, target: clarification}
```

The graph builder should implement `decision_field` routing generically by
reading the configured field from state and resolving it through `routes`.
Custom router callables should remain available only for routing that cannot be
expressed through a state field.

## Runtime Responsibilities

### Reusable agent-node builder

The reusable builder should:

1. Resolve the configured agent definition.
2. Resolve the selected primary and fallback model profiles.
3. Load the configured prompt.
4. Resolve assigned tools by node or agent ID.
5. Build the agent with the common middleware stack.
6. Construct input from declared state fields.
7. Invoke the agent once.
8. Parse and validate structured output when configured.
9. Write only the declared state fields.
10. Record consistent tracing and failure information.

### Deterministic nodes

Deterministic nodes should own logic that must not depend on model behavior,
including:

- Input validation and normalization.
- Confidence thresholds and retry limits.
- Routing-decision calculation from structured agent output.
- Policy-safe fallback selection.
- Final response shaping.

Agent prompts may describe the state fields an agent must produce, but prompts
must not manually decide which node executes next. Routing remains part of the
graph contract.

### Tool resolution

Tool usage should be declared in `agents.yaml` or the node configuration. The
runtime should resolve tools using one reusable hook such as
`get_tools_for_node(node_id)`. A tool is an agent dependency, not a workflow
transition, and should not require patching agent-node construction or workflow
registries.

## Expected Generated File Footprint

### Initial workflow with four agents and two tools

| Category | Expected count |
| --- | ---: |
| Core configuration | 2 |
| Typed `state.py` contract | 1 |
| Deterministic workflow logic | 1 |
| Tool implementations and connector helper | 2-3 |
| Dependency files | 0 or 2 |
| Documentation and environment example | 1-2 |
| Total | Approximately 6-11 |

### Adding one agent to an existing workflow

| Change | Expected count |
| --- | ---: |
| `agents.yaml` | 1 |
| `graph_config.yaml` | 1 |
| Typed `state.py`, when fields change | 0-1 |
| Total | Approximately 2-3 |

Connector files, the tool registry, dependency manifests, API code, workflow
bootstrap, model fallback, tracking, and broad tests should not change for a
normal agent addition.

## Template Work Before Agent-Orch Changes

Implement and verify these changes in the template first:

1. Remove the competing deprecated and replacement sample builders.
2. Introduce the reusable agent-node builder.
3. Add `agents.yaml` loading and validation.
4. Add generic `decision_field` conditional routing.
5. Add node-level tool resolution through a stable runtime hook.
6. Establish the stable state envelope.
7. Consolidate graph node wrapping and tracing.
8. Add graph-contract validation for references, reachability, terminal paths,
   duplicate routes, route destinations, and mixed routing mechanisms.
9. Keep one configuration-driven `tests/test_template.py` for agent construction,
   tool attachment, routing, validation, and the stable state envelope.
10. Make dependency sync, compilation, tests, Ruff, and mypy pass without network
    calls from unit tests.

## Agent-Orch Responsibilities After Refactor

After the template exposes stable extension points, agent-orch should:

1. Read the approved design from `task_payload.json`.
2. Map design agents to `agents.yaml`.
3. Map deterministic and agent nodes, direct edges, conditional edges, entry, and
   end nodes to `graph_config.yaml`.
4. Record deterministic operation type, description, parameters, reads, and
   writes in graph configuration and implement every deterministic callable in
   `workflow_nodes.py`.
5. Write Python tool factories to `tools.yaml`, MCP server connections to
   `mcp.yaml`, and generate only assigned tool modules and connector helpers.
6. Detect imported direct dependencies and update `pyproject.toml` and `uv.lock`
   when needed.
7. Run the template-owned test suite without rewriting it for generated agent names.
8. Treat template-owned runtime and test files as read-only unless the approved design
   requires a genuinely unsupported runtime capability.
9. Reject unexpected broad diffs instead of accepting a full template rewrite.
10. Complete only after dependency sync, compilation, tests, static checks, and
    LangGraph contract validation pass.

## Generated-Diff Guardrail

For a normal LangGraph generation run, agent-orch should expect changes only in
the configurable and workflow-specific files described above. Changes to any of
the following should require an explicit reason:

- `graph/graph_builder.py`
- `nodes/agent_node_builder.py`
- `tools/tool_registry.py`
- `workflow.py`
- `main.py`
- Model factory and fallback implementation
- Tracking and tracing infrastructure
- Shared API contracts
- `tests/test_template.py`

This guardrail prevents a generated workflow from replacing stable framework
code and provides an early signal when code generation is becoming broader than
the approved design requires.

## Acceptance Criteria

The proof of concept is successful when:

- The sample workflow supports deterministic nodes, agent nodes, direct edges,
  conditional routes, retries, tool assignments, and a final response path.
- Graph configuration is the authoritative source of workflow sequence.
- Agent configuration is the authoritative source of prompts, models, state
  access, and tools.
- Adding one ordinary agent requires no runtime-framework edits.
- A four-agent, two-tool generated workflow stays within the expected file-count
  range unless documented requirements justify additional files.
- `uv sync`, imports, compilation, tests, Ruff, mypy, and graph validation pass.
- The single template-owned test file remains unchanged and does not call external
  services.
- The generated repository contains no temporary `_work` artifacts.
