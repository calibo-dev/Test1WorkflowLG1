from __future__ import annotations

import sys
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from latest_ai_development.config.settings import get_settings
from latest_ai_development.config.validators import validate_configuration
from latest_ai_development.graph.graph_builder import build_graph


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _normalize_context(raw_context: str | None) -> str:
    if not raw_context:
        return ""

    normalized = raw_context.strip()

    if not normalized or normalized == "/":
        return ""

    return "/" + normalized.strip("/")


# --------------------------------------------------
# Initialization
# --------------------------------------------------

settings = get_settings()
validate_configuration()

graph = build_graph()

CONTEXT_PATH = _normalize_context(settings.context)


# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="LangGraph Research + Reporting Workflow",
    root_path=CONTEXT_PATH,
)


# --------------------------------------------------
# Request / Response Models
# --------------------------------------------------

class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    query: str
    report: str


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    }


# --------------------------------------------------
# Ask Endpoint
# --------------------------------------------------

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):

    try:

        result = graph.invoke(
            {
                "query": request.query,
                "research_result": "",
                "report": "",
            }
        )

        return {
            "query": request.query,
            "report": result["report"],
        }

    except Exception as e:

        return {
            "query": request.query,
            "report": f"Error: {str(e)}",
        }


# --------------------------------------------------
# CLI Test
# --------------------------------------------------

def test():

    result = graph.invoke(
        {
            "query": "Latest AI Agent Frameworks",
            "research_result": "",
            "report": "",
        }
    )

    print(result["report"])


# --------------------------------------------------
# Trigger From JSON
# --------------------------------------------------

def run_with_trigger():

    import json

    if len(sys.argv) < 2:
        raise ValueError("Provide JSON payload")

    payload = json.loads(sys.argv[1])

    result = graph.invoke(
        {
            "query": payload["query"],
            "research_result": "",
            "report": "",
        }
    )

    print(result)


# --------------------------------------------------
# Server
# --------------------------------------------------

def run():

    uvicorn.run(
        "latest_ai_development.main:app",
        host=settings.api_host,
        port=settings.port,
        reload=settings.debug,
    )


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        "latest_ai_development.main:app",
        host=settings.api_host,
        port=settings.port,
        reload=settings.debug,
    )