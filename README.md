# LangGraph Python Template

Configuration-driven LangGraph template with a FastAPI endpoint, reusable agent
runtime, deterministic nodes, conditional routing, model fallback, tool loading,
and optional LangSmith tracing.

## Runtime Contract

The template separates workflow configuration from stable runtime code:

- `config/agents.yaml` defines agent roles, prompts, concrete providers and model
  names, state access, outputs, and tool/MCP assignments.
- `config/tools.yaml` defines ordinary Python tool factories.
- `config/mcp.yaml` defines MCP server transports and connection metadata.
- `config/graph_config.yaml` defines the input/internal/output state contract,
  nodes, entry/end nodes, direct edges, conditional routes, tracing, and the
  final output field.
- `graph/state.py` is the stable factory that converts the configured state
  contract into LangGraph `InputState`, `State`, and `OutputState` schemas.
- `nodes/agent_node_builder.py` builds every configured agent through one runtime
  path.
- `nodes/workflow_nodes.py` contains direct deterministic functions plus the
  allowlisted node and router registries.
- `graph/graph_builder.py` compiles the configuration into a `StateGraph`.
- `tools/tool_registry.py` resolves tools assigned to configured graph nodes.
- `tests/test_template.py` verifies the reusable contract without naming sample
  agents or requiring generated test rewrites.

See `approach.md` for the refactor rationale, ownership rules, and generated-file
targets.

## Sample Workflow

```text
START
  -> prepare_feedback
       invalid -> finalize_output -> END
       valid   -> classify_sentiment -> evaluate_feedback
                    appreciation    -> appreciation_response -> finalize_output
                    clarification   -> clarification_response -> finalize_output
                    service_recovery -> service_recovery_response
                                        -> policy_guidance -> finalize_output
                    human_review    -> interrupt/resume -> finalize_output
```

The sample shows both routing styles. `prepare_feedback` uses a simple configured
`decision_field`. `evaluate_feedback` computes weighted urgency and policy-risk
scores, then a named Python router combines those scores with sentiment and model
confidence to select one of four paths. High-risk feedback pauses at
`human_review`; negative feedback runs a two-agent recovery and policy branch.

## Agent Configuration

```yaml
agents:
  sentiment_classifier:
    role: Customer feedback sentiment classifier
    prompt: prompts/sentiment_classifier.md
    provider: openai
    model: gpt-4.1-mini
    reads_state:
      - normalized_feedback
    writes_state:
      - sentiment
      - confidence
      - sentiment_reason
    output_mapping:
      sentiment: sentiment
      confidence: confidence
      reason: sentiment_reason
    tools: []
    mcp_servers: []
```

Tool-backed agents list configured tool names in `tools`. Tool definitions use a
factory reference in `config/tools.yaml`:

```yaml
tools:
  country_data:
    factory: latest_ai_development.tools.country_data:create_tool
    description: Read-only country reference lookup.
    connector:
      category: RDBMS
      type: SNOWFLAKE
    factory_kwargs:
      table_name: ak_country
      read_only: true
```

The generic registry imports the factory lazily, calls it with `factory_kwargs`,
and caches the resulting tool. Optional `description`, `connector`, `tags`,
`usage_guidelines`, and `non_usage_guidelines` document the tool contract. Keep
credentials outside YAML; factories should read them from environment variables
or the configured secrets manager.

MCP servers are declared separately in `config/mcp.yaml`:

```yaml
servers:
  catalog:
    transport: streamable_http
    url: https://mcp.example.com/mcp
    headers:
      Authorization: ${MCP_CATALOG_AUTHORIZATION}
```

Supported configuration transports are `stdio`, `sse`, and `streamable_http`.
This POC validates MCP declarations and agent references. Until an MCP adapter is
added to the reusable runtime, startup validation rejects non-empty agent
`mcp_servers` assignments instead of silently ignoring them.

For structured agent responses, replace `output_field` with an `output_mapping`.
The runtime builds a Pydantic response schema from the mapping; keys are structured
response fields and values are workflow state fields:

```yaml
writes_state:
  - sentiment
  - confidence
output_mapping:
  classification: sentiment
  score: confidence
```

## Graph Configuration

Declare graph-wide state alongside the topology. Input `required` controls the
public API requirement; internal/output defaults are copied into each new run.
Reducers are restricted to the reusable `append`, `messages`, and `merge`
implementations.

```yaml
state:
  input:
    user_input:
      type: string
      required: true
  internal:
    errors:
      type: list
      items: string
      reducer: append
      default: []
    decision:
      type: string
  output:
    final_response:
      type: string
```

Agent nodes reference an entry from `agents.yaml`:

```yaml
nodes:
  classify_sentiment:
    type: agent
    agent: sentiment_classifier
```

Every deterministic node exposes its operation contract in configuration. Its
node ID selects the direct Python callable from `NODE_REGISTRY`:

```yaml
nodes:
  prepare_feedback:
    type: deterministic
    reads_state:
      - user_input
    writes_state:
      - normalized_feedback
      - validation_status
      - errors
    operation:
      type: input_validation
      description: Validate and normalize incoming customer feedback.
      parameters:
        required: true
        min_length: 4
```

`operation.type`, `operation.description`, and `operation.parameters` are
descriptive, requirement-derived metadata. They are available to validation and
workflow UIs but are not executed by the graph runtime. This keeps one consistent
implementation path for input validation, routing calculations, interrupts,
custom algorithms, and every other deterministic node.

For example, a scoring node uses the same shape:

```yaml
nodes:
  evaluate_feedback:
    type: deterministic
    reads_state:
      - normalized_feedback
      - sentiment
      - confidence
    writes_state:
      - urgency_score
      - policy_risk_score
    operation:
      type: feedback_scoring
      description: Calculate urgency and policy-risk scores.
      parameters:
        score_range:
          minimum: 0
          maximum: 100
```

Configuration validation requires operation metadata and declared state fields
for every deterministic node, plus an exact callable under the same node ID in
`NODE_REGISTRY`. Each direct function returns a partial state update.

Use direct edges for unconditional execution:

```yaml
edges:
  - from: service_recovery_response
    to: policy_guidance
```

Use a state field and path map for ordinary conditional routing:

```yaml
conditional_edges:
  - from: prepare_feedback
    decision_field: validation_status
    path_map:
      valid: classify_sentiment
      invalid: finalize_output
```

A conditional edge may use a `ROUTER_REGISTRY` key instead of
`decision_field` when routing cannot be expressed through one state field:

```yaml
conditional_edges:
  - from: evaluate_feedback
    router: evaluate_feedback
    path_map:
      appreciation: appreciation_response
      clarification: clarification_response
      service_recovery: service_recovery_response
      human_review: human_review
```

The registered router is ordinary testable Python and may evaluate several state
fields. Keep destinations in YAML so topology remains visible to configuration
validation and workflow UIs. A node must not mix outgoing mechanisms.

Use `Command(update=..., goto=...)` when one node must atomically update state and
choose its next node. Declare every possible destination on that node so graph
visualization and configuration validation still know the topology:

```yaml
nodes:
  approval_gate:
    type: deterministic
    reads_state:
      - approval_status
    writes_state:
      - approval_status
    operation:
      type: approval_gate
      description: Route an approval decision atomically.
      parameters: {}
    command_destinations:
      - approved_work
      - rejected_response
```

The `NODE_REGISTRY["approval_gate"]` callable returns
`Command(update={...}, goto=...)`.
Do not also give that source a direct or conditional outgoing route.

Use `Send` only for dynamic fan-out, where the number of worker invocations is
known at runtime rather than design time:

```yaml
conditional_edges:
  - from: plan_work
    router: plan_work_dispatch
    send_destinations:
      - worker
```

`ROUTER_REGISTRY["plan_work_dispatch"]` points to `dispatch_work`, which returns
one `Send("worker", worker_input)` per work item.
The declared destinations are used for validation; the router's returned `Send`
objects carry the per-worker inputs.

Per-request dependencies and configuration belong in LangGraph runtime context,
not persisted workflow state. Define default values under `runtime_context` and
override them through the API request's `context` object. A node that needs these
values accepts a second `runtime` argument and reads `runtime.context["values"]`:

```yaml
runtime_context:
  defaults:
    locale: en-US
```

```python
def node(state: State, runtime: Runtime[RuntimeContext]) -> dict[str, str]:
    locale = runtime.context["values"]["locale"]
    return {"draft_response": render(state, locale=locale)}
```

Ordinary nodes remain one-argument callables. These advanced features are
independent and opt-in; subgraphs are intentionally outside the current template
contract.

## API

Request:

```http
POST /ask
Content-Type: application/json

{"user_input": "The product is beautiful and I love it", "thread_id": "demo-1", "context": {"locale": "en-US"}}
```

Response:

```json
{
  "thread_id": "demo-1",
  "status": "completed",
  "final_response": "...",
  "interrupts": []
}
```

The FastAPI path awaits `graph.ainvoke()` and never blocks the event loop with a
synchronous graph call. Persistence is enabled through the configured checkpointer;
when a node calls LangGraph `interrupt()`, send its returned `thread_id` and the
user's answer to `POST /resume` as `{"thread_id":"demo-1","value":...}`.
FastAPI lifespan validates configuration and compiles one reusable workflow at
startup, then closes graph and registered tool resources during shutdown.

Health endpoint: `GET /health`.

## Configuration

Provider generation parameters and fallbacks live in `config/models.yaml`. Each
agent selects its concrete `provider` and primary `model` in `config/agents.yaml`;
there is no environment or provider-level primary model fallback.

Common environment variables:

- `OPENAI_API_KEY`, `ANTHROPICAI_API_KEY`, `GEMINIAI_API_KEY`, or
  `BEDROCK_AI_API_KEY`.
- `CLOUD_PROVIDER`: empty for local mode, `AWS`, or `AZURE`.
- `AZURE_KEY_VAULT_URL` and `AWS_REGION`.
- `TRACING_BACKEND`: `NONE` or `LANGSMITH`.
- `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`.
- `API_HOST`, `PORT`, and `API_ROOT_PATH`.
- `OUTPUT_DIR` and `REPORT_OUTPUT_FILE`.

## Running

```bash
uv sync --all-extras
uv run latest_ai_development
```

Direct trigger:

```bash
uv run run_with_trigger '{"user_input":"The product is beautiful and I love it"}'
```

## Validation

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check src tests
uv run mypy src
uv run python -m compileall -q src tests
git diff --check
```

Startup validation rejects undefined agents or tools, invalid references,
undefined state reads/writes, unreachable nodes, paths without `END`, duplicate
edges, invalid route targets, and nodes that mix direct and conditional outgoing
routing.

## Extension Rules

For an ordinary agent addition, update only:

1. `config/agents.yaml`.
2. `config/graph_config.yaml`, including its `state` mappings.
3. A prompt file only when legacy file-based prompts are intentionally used;
   generated workflows normally keep prompts inline in `agents.yaml`.

The generated workflow should pass `tests/test_template.py` unchanged. The suite
derives its contract checks from configuration instead of hard-coding workflow
agent names.

Do not edit `graph/state.py`, `graph/graph_builder.py`, `nodes/agent_node_builder.py`,
`tools/tool_registry.py`, `workflow.py`, `main.py`, model fallback, or tracking for
ordinary generated workflows.
