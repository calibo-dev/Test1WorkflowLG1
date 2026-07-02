
# LangGraph Multi-Agent Research & Reporting System

> **NOTE**
>
> This document is provided as a starter template only.
> Customers must review, update, and validate this content to ensure it meets their functional, security, compliance, and operational requirements before deployment.

---

## 1. Overview

This project is a production-oriented **LangGraph Multi-Agent AI System** built in Python.

The application orchestrates multiple AI agents in a graph-based workflow to perform:

- Research on a given topic
- Analyze collected information
- Generate structured markdown reports
- Support multiple LLM providers
- Secure secret management
- Containerized deployment using Docker, Helm, and Jenkins

The project uses **LangGraph** for orchestration and provides a modular architecture that can be extended with additional agents, tools, and workflows.

---

## 2. Architecture

The system follows a graph-based workflow with orchestration nodes and template-defined agents.

```text
                    Topic
                      |
                      v

            +----------------+
            | Graph Builder  |
            +----------------+
                      |
                      v
        +---------------------------+
        | Validate Input            |
        +---------------------------+
                      |
                      v
        +---------------------------+
        | Router Agent              |
        +---------------------------+
                      |
                      v
 +----------------+     +----------------+     +------------------+
 | Research Node  | --> | Reporting Node | --> | Finalize Output  |
 +----------------+     +----------------+     +------------------+
                      |
                      v

             Markdown Report
```

---

## 3. Tech Stack

| Technology | Version |
|------------|---------|
| Python | >=3.11 |
| LangGraph | Latest |
| LangChain | >=0.3 |
| FastAPI | Latest |
| OpenAI | Supported |
| Ollama | Supported |
| AWS Secrets Manager | Supported |
| Docker | Supported |
| Helm | Supported |
| Jenkins | Supported |

---

## 4. Project Structure

```bash
.
├── helm_chart/
│   ├── Chart.yaml
│   └── templates/
│
├── src/
│   └── latest_ai_development/
│
│       ├── graph/
│       │   ├── graph_builder.py
│       │   └── state.py
│
│       ├── llm/
│       │   ├── llm_factory.py
│       │   └── model_settings.py
│
│       ├── nodes/
│       │   ├── research_node_builder.py
│       │   └── reporting_node_builder.py
│
│       ├── prompts/
│       │   └── prompt_builder.py
│
│       ├── main.py
│       └── secrets_manager.py
│
├── AGENTS.md
├── Dockerfile
├── Jenkinsfile
├── Jenkinsfile.ci
├── Jenkinsfile.deploy
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

---

## 5. Components

### Graph Builder

**File:** `graph/graph_builder.py`

Responsible for:

- Building LangGraph workflow
- Defining node transitions
- Managing graph execution flow
- Routing state between agents
- Wiring orchestration nodes, the router agent, and conditional edges

---

### State Management

**File:** `graph/state.py`

Defines:

- Shared graph state
- Input and output schema
- Agent communication state
- Intermediate results
- Internal agent plan and validation errors

Current minimal external request:

- `topic`

Current internal state includes:

- `topic`
- `agent_plan`
- `errors`
- `research_result`
- `report`

The template uses a `State` `TypedDict` as the primary LangGraph state model.

---

### LLM Factory

**File:** `llm/llm_factory.py`

Responsible for:

- Creating LLM clients
- Selecting providers
- Managing model initialization
- Centralized model configuration

Supported providers:

- OpenAI
- Ollama
- Extendable to Anthropic, Gemini, Groq

---

### Research Node

**File:** `nodes/research_node_builder.py`

Responsible for:

- Topic understanding
- Information gathering
- Data analysis
- Structured research generation

---

### Reporting Node

**File:** `nodes/reporting_node_builder.py`

Responsible for:

- Summarizing research
- Generating markdown reports
- Formatting final output
- Creating human-readable responses
- Writing the final markdown report to `/data/report.md`

---

### Prompt Builder

**File:** `prompts/prompt_builder.py`

Responsible for:

- Prompt templates
- System prompts
- User prompt construction
- Agent-specific instructions

---

### Secrets Manager

**File:** `secrets_manager.py`

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
- Azure OpenAI:
  - direct value: `AZURE_OPENAI_API_KEY`
  - secret reference: `API_KEY_SECRET` or `AZURE_OPENAI_API_KEY_SECRET`
  - endpoint: `AZURE_OPENAI_ENDPOINT` or `AZUREAI_OPENAI_URL`
  - API version: `AZURE_OPENAI_API_VERSION` or `AZUREAI_OPENAI_API_VERSION`

### Observability and Tracing

- LangSmith tracing is required for this template
- the template enables LangSmith during FastAPI app startup
- if LangSmith credentials are missing, startup fails

LangSmith configuration:

- API key can come from:
  - `LANGSMITH_API_KEY`
  - `LANGSMITH_API_KEY_SECRET`
- project name can be provided with `LANGSMITH_PROJECT`

---

## 6. Installation

### Clone Repository

```bash
git clone <repository-url>

cd <repository-name>
```

---

### Create Virtual Environment

#### Linux / Mac

```bash
python -m venv .venv

source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

---

### Install Dependencies

Using UV:

```bash
uv sync
```

or

```bash
uv pip install -e .
```

---

## 7. Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_key

OLLAMA_BASE_URL=http://localhost:11434

AWS_REGION=us-east-1

LANGCHAIN_TRACING_V2=true

LANGCHAIN_API_KEY=your_langsmith_key

LANGCHAIN_PROJECT=langgraph-research-agent
```

---

## 8. Running the Application

Run locally:

```bash
PYTHONPATH=src uv run python -m latest_ai_development.main
```

or

```bash
python src/latest_ai_development/main.py
```

---

## 9. Docker

Build Docker image:

```bash
docker build -t langgraph-agent .
```

Run container:

```bash
docker run -p 8000:8000 langgraph-agent
```

---

## 10. Helm Deployment

Helm charts are provided inside:

```bash
helm_chart/
```

Install:

```bash
helm install langgraph-agent ./helm_chart
```

Upgrade:

```bash
helm upgrade langgraph-agent ./helm_chart
```

---

## 11. CI/CD

The project provides Jenkins pipelines:

- `Jenkinsfile`
- `Jenkinsfile.ci`
- `Jenkinsfile.deploy`

These pipelines can be used for:

- Build
- Unit Testing
- Docker Image Creation
- Kubernetes Deployment
- Helm Release Management

---

## 12. Extending the System

You can easily add:

- Additional AI agents
- Custom tools
- RAG pipelines
- Vector databases
- Human-in-the-loop approval nodes
- Multi-step workflows

Simply create a new node under:

```bash
src/latest_ai_development/nodes/
```

and register it in:

```bash
graph/graph_builder.py
```

---

## 13. License

This project is licensed under the MIT License.

---

## 14. Author

Built using **LangGraph**, **LangChain**, and modern AI engineering best practices.
