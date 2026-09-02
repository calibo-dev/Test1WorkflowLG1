# LangGraph Template Guide

This document explains the current runtime and extension contract for the LangGraph template.

## What This Template Does

The template exposes a FastAPI service that accepts a `topic`, runs a LangGraph workflow, and returns a generated `report`.

Current API request:

```json
{
  "topic": "Your Topic Here"
}
```

Current API response:

```json
{
  "topic": "Your Topic Here",
  "report": "..."
}
```

## Runtime Architecture

The runtime is split into four clear layers:

1. `config/settings.py`
   Central environment and path management through `BaseSettings`.
2. `config/models.yaml`
   Provider and model defaults used by the LLM layer.
3. `config/graph_config.yaml`
   Graph topology contract: entry point, node builders, edges, conditional routes, and final output field.
4. `workflow.py`
   Graph lifecycle, initial state creation, invocation, and report persistence.
5. `main.py`
   FastAPI request/response handling and app startup.
6. `logging_utils.py`
   Shared logging setup aligned with the LangChain template pattern.

## Current Graph Flow

The compiled graph in `graph/graph_builder.py` is now loaded from `config/graph_config.yaml` and currently resolves this flow:

```text
validate_input
  -> router_agent
  -> research | reporting | finalize_output
research
  -> reporting | finalize_output
reporting
  -> finalize_output
finalize_output
  -> END
```

Node ownership:

- `validate_input`
  Deterministic orchestration node
- `router_agent`
  Agent-backed orchestration node
- `research`
  Agent-backed content generation node
- `reporting`
  Agent-backed content generation node
- `finalize_output`
  Deterministic orchestration node

## Shared State

The current shared state is defined in `graph/state.py` as a `TypedDict`.

Active fields:

- `topic`
- `agent_plan`
- `errors`
- `research_result`
- `report`

If the workflow contract changes, update state, nodes, graph wiring, and request/response models together.
For topology-only changes, update `config/graph_config.yaml` first and keep node behavior in Python.

## LLM and Provider Setup

LLM setup follows this path:

1. `config/models.yaml`
2. `llm/model_settings.py`
3. `llm/llm_factory.py`
4. node builder invocation

The node builders should not hardcode model defaults. Provider selection comes from `Settings`, and per-provider model metadata comes from `models.yaml`.

## Model Fallback

- `src/latest_ai_development/config/models.yaml` may define an optional `fallback` block under the active provider.
- Active agent-backed nodes keep the primary model in `create_agent(...)` and wire fallback models through LangChain `ModelFallbackMiddleware`.
- `src/latest_ai_development/llm/model_attempt_logging.py` adds attempt-level logging for the primary model and each fallback retry.
- Fallback is a model-call concern only; it must not be implemented as graph topology, router branching, or extra workflow nodes.

## Secrets and Runtime Variables

Deployment/runtime variables are resolved centrally in `config/settings.py` and then consumed by the app runtime.

Handled through settings:

- `API_HOST`
- `PORT`
- Default: `8089`
- `API_ROOT_PATH`
- `OUTPUT_DIR`
- Default: `output/`
- `REPORT_OUTPUT_FILE`
- `AZURE_KEY_VAULT_URL`
- `CONTEXT`
- Default: `/`

Secret handling lives in `secrets_manager.py`.

Current behavior:

- `CLOUD_PROVIDER=AWS` uses AWS Secrets Manager
- `CLOUD_PROVIDER=AZURE` uses Azure Key Vault
- empty/unset `CLOUD_PROVIDER` uses local-development mode

Local-development mode treats secret-related environment variables as direct values so the template can run locally without external secret-manager dependencies.

Typical current secret resolution:

- OpenAI:
  - direct value: `OPENAI_API_KEY`
  - secret reference: `OPENAI_API_KEY_SECRET`
- Anthropic:
  - direct value: `ANTHROPICAI_API_KEY`
  - secret reference: `ANTHROPICAI_API_KEY_SECRET`
- Gemini:
  - direct value: `GEMINIAI_API_KEY`
  - secret reference: `GEMINIAI_API_KEY_SECRET`
- Bedrock:
  - direct value: `BEDROCK_AI_API_KEY`
  - secret reference: `BEDROCK_AI_API_KEY_SECRET`

## Startup and Validation

App startup currently performs:

1. configuration loading from `config/settings.py`
2. logging configuration through `logging_utils.py`
3. provider/config validation from `config/validators.py`
   - validates that the selected provider exists in `models.yaml`
   - resolves the selected provider credentials during startup
4. workflow controller initialization through `workflow.py`
5. LangSmith tracing initialization through `tracking.py` during FastAPI lifespan startup
6. FastAPI app startup

LangSmith tracing is controlled by `LANGSMITH_TRACING`.

- If `LANGSMITH_TRACING=true`, startup requires valid LangSmith credentials.
- If `LANGSMITH_TRACING=false`, startup continues with tracing disabled.

LangSmith inputs:

- `LANGSMITH_TRACING`
- `LANGSMITH_API_KEY`
- `LANGSMITH_API_KEY_SECRET`
- `LANGSMITH_PROJECT`
- `LANGSMITH_ENDPOINT`

## Direct Invocation

Use the workflow helpers if you want to invoke the template directly in Python:

```python
from latest_ai_development.main import _build_initial_state, get_graph

graph = get_graph()
result = graph.invoke(_build_initial_state(topic="Your Topic Here"))
print(result["report"])
```

Or use the workflow facade:

```python
from latest_ai_development.workflow import LatestAiDevelopmentWorkflow

workflow = LatestAiDevelopmentWorkflow()
result = workflow.kickoff({"topic": "Your Topic Here"})
print(result["report"])
```

## File Ownership

These files are workflow-critical and should stay aligned:

- `src/latest_ai_development/config/settings.py`
- `src/latest_ai_development/config/models.yaml`
- `src/latest_ai_development/config/graph_config.yaml`
- `src/latest_ai_development/config/validators.py`
- `src/latest_ai_development/graph/state.py`
- `src/latest_ai_development/graph/graph_builder.py`
- `src/latest_ai_development/main.py`
- `src/latest_ai_development/workflow.py`
- `src/latest_ai_development/prompts/prompt_builder.py`
- `src/latest_ai_development/nodes/orchestration_node_builder.py`
- `src/latest_ai_development/nodes/research_node_builder.py`
- `src/latest_ai_development/nodes/reporting_node_builder.py`

## Output Persistence

Report-file persistence is controlled by `execution.save_output` in `src/latest_ai_development/config/graph_config.yaml`.

Current default:

- `save_output: false`

This default avoids write failures on deployments that run with `readOnlyRootFilesystem: true`.

If you enable file output:

- point `OUTPUT_DIR` to a writable path such as a mounted volume or `/tmp/output`
- do not assume `/app/output` is writable in production

If report writing fails at runtime, the workflow logs a warning and still returns the API response instead of failing the request.

## Extension Guidance

When extending the template:

- update `config/graph_config.yaml` first when changing graph topology
- add or change state fields in `graph/state.py` when the state contract changes
- update prompt helpers for any new agent-backed node
- keep node outputs aligned with shared state
- only update `graph/graph_builder.py` if the generic config-driven compiler contract itself changes
- update `main.py` request/response models if the external contract changes
- keep any optional report persistence logic inside `workflow.py`

## Validation

Recommended checks after edits:

```bash
uv run --with pytest python -m pytest
.venv/bin/python -m compileall src/latest_ai_development
```
