# LangGraph Python Template

This repository is the LangGraph workflow template for `latest-ai-development`.
It exposes a FastAPI app, centralizes runtime configuration, and executes a small graph-based research and reporting workflow.

## Runtime Contract

The template now uses a centralized configuration layer:

- `src/latest_ai_development/config/settings.py`
  Reads environment variables with `BaseSettings`, normalizes provider and API root-path values, and manages output paths.
- `src/latest_ai_development/config/models.yaml`
  Declares provider-specific model defaults.
- `src/latest_ai_development/config/graph_config.yaml`
  Declares graph topology, entry point, edges, conditional routes, and the final output field.
- `src/latest_ai_development/config/validators.py`
  Validates provider and cloud-secret configuration during startup.
- `src/latest_ai_development/workflow.py`
  Owns graph creation, initial state construction, workflow kickoff, and final report persistence.
- `src/latest_ai_development/logging_utils.py`
  Applies shared startup logging configuration without clobbering existing handlers.

## Current Workflow

The current template compiles a LangGraph workflow from `src/latest_ai_development/config/graph_config.yaml`, while node behavior remains in Python node-builder files.

Current graph-config flow:

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

Current shared state lives in `src/latest_ai_development/graph/state.py` and contains:

- `topic`
- `agent_plan`
- `errors`
- `research_result`
- `report`

The external request contract is intentionally small:

```json
{
  "topic": "Explain LangGraph workflow support"
}
```

The response contract is:

```json
{
  "topic": "Explain LangGraph workflow support",
  "report": "..."
}
```

## Project Structure

```text
src/latest_ai_development/
├── config/
│   ├── __init__.py
│   ├── graph_config.yaml
│   ├── models.yaml
│   ├── settings.py
│   └── validators.py
├── graph/
│   ├── graph_builder.py
│   └── state.py
├── logging_utils.py
├── llm/
│   ├── llm_factory.py
│   └── model_settings.py
├── nodes/
│   ├── orchestration_node_builder.py
│   ├── reporting_node_builder.py
│   └── research_node_builder.py
├── prompts/
│   └── prompt_builder.py
├── main.py
├── secrets_manager.py
├── tracking.py
└── workflow.py
```

## Configuration

Important environment variables are resolved through `Settings` in `config/settings.py`.

Common values:

- `PROVIDER`
  Supported by the current template: `OPENAI`, `ANTHROPIC`, `GEMINI`, `OLLAMA`
- `API_HOST`
- `PORT`
- `API_ROOT_PATH`
- `OUTPUT_DIR`
- `REPORT_OUTPUT_FILE`
- `LANGSMITH_API_KEY` or `LANGSMITH_API_KEY_SECRET`
- `LANGSMITH_PROJECT`
- `CLOUD_PROVIDER`
- `AWS_REGION`
- `AZURE_KEY_VAULT_URL`

Model defaults come from `src/latest_ai_development/config/models.yaml`, not from hardcoded values in the graph or node files.
Graph topology comes from `src/latest_ai_development/config/graph_config.yaml`, not from hardcoded edges inside `graph_builder.py`.

These deployment/runtime values are handled through centralized settings and then consumed by the runtime:

- `API_HOST`
  Used by uvicorn server startup.
- `PORT`
  Used by uvicorn server startup.
  Default: `8080`
- `API_ROOT_PATH`
  Normalized through `normalized_api_root_path` and applied to FastAPI `root_path`.
- `OUTPUT_DIR`
  Used to build the final report output directory.
  Default: `output/`
- `REPORT_OUTPUT_FILE`
  Used to build the final report output filename.
- `AZURE_KEY_VAULT_URL`
  Used by the Azure Key Vault secret manager when `CLOUD_PROVIDER=AZURE`.
- `CONTEXT`
  Defaults to `/` and is used as the fallback source for API root-path normalization when `API_ROOT_PATH` is not set.

## Secrets Manager

**File:** `src/latest_ai_development/secrets_manager.py`

Provides secure access to:

- OpenAI API keys
- AWS credentials
- Environment secrets
- External service tokens

AWS Secrets Manager integration is supported.

Current template behavior:

- `CLOUD_PROVIDER=AWS` uses AWS Secrets Manager
- `CLOUD_PROVIDER=AZURE` uses Azure Key Vault
- empty/unset `CLOUD_PROVIDER` is treated as local-development mode

Local-development mode:

- secret-related environment variables are treated as direct values
- this allows local runs without external secret-manager dependencies

Typical API key configuration:

- OpenAI:
  - direct value: `OPENAI_API_KEY`
  - secret reference: `API_KEY_SECRET` or `OPENAI_API_KEY_SECRET`
- Anthropic:
  - direct value: `ANTHROPICAI_API_KEY`
  - secret reference: `ANTHROPICAI_API_KEY_SECRET`
- Gemini:
  - direct value: `GEMINIAI_API_KEY`
  - secret reference: `GEMINIAI_API_KEY_SECRET`

## Observability and Tracing

- LangSmith tracing is required for this template
- the template enables LangSmith during FastAPI app startup
- if LangSmith credentials are missing, startup fails

LangSmith configuration:

- API key can come from:
  - `LANGSMITH_API_KEY`
  - `LANGSMITH_API_KEY_SECRET`
- project name can be provided with `LANGSMITH_PROJECT`

Startup order is:

1. load settings
2. configure logging
3. validate configuration
4. initialize workflow controller
5. initialize LangSmith tracing during app lifespan
6. serve FastAPI routes

## YAML-First Topology Changes

This template intentionally does not use LangChain-style `workflow.yaml` or `stages.yaml`.

For LangGraph changes:

- update `src/latest_ai_development/config/graph_config.yaml` first when the topology changes
- keep node behavior in `nodes/*_node_builder.py`
- only update Python files beyond the node builders when the state contract, prompts, or API contract changes

`graph/graph_builder.py` is now a generic compiler that reads `graph_config.yaml`, resolves Python builders and router functions, and builds the graph from that config.

## Running

Install dependencies:

```bash
uv sync
```

Run the FastAPI app:

```bash
uv run latest_ai_development
```

Run the direct test helper:

```bash
uv run test
```

Run with a trigger payload:

```bash
uv run run_with_trigger '{"topic":"Explain LangGraph workflow support"}'
```

## API

Health endpoint:

```text
GET /health
```

Workflow endpoint:

```text
POST /ask
```

Example request:

```json
{
  "topic": "Explain LangGraph workflow support"
}
```

## Testing

Run the test suite:

```bash
uv run --with pytest python -m pytest
```

## Notes

- Report persistence is handled in `workflow.py`, which writes the final report to `OUTPUT_DIR/REPORT_OUTPUT_FILE`.
- LangSmith tracing is initialized during app startup through `tracking.py`.
- Secret resolution is centralized through `secrets_manager.py` and respects the configured cloud provider.
