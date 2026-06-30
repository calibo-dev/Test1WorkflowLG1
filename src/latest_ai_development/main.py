from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from latest_ai_development.graph.graph_builder import GraphBuilder
from latest_ai_development.tracking import initialize_langsmith_tracing

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Initialization
# --------------------------------------------------

graph = None
tracing_initialized = False


def _normalize_context(raw_context: str | None) -> str:
    """Return a stable path prefix like '/testing' or ''."""
    if not raw_context:
        return ""

    normalized = raw_context.strip()
    if not normalized or normalized == "/":
        return ""

    return "/" + normalized.strip("/")


CONTEXT_PATH = _normalize_context(os.getenv("CONTEXT"))


def ensure_tracing_initialized() -> None:
    global tracing_initialized
    if not tracing_initialized:
        initialize_langsmith_tracing()
        tracing_initialized = True


def get_graph():
    global graph
    ensure_tracing_initialized()
    if graph is None:
        graph = GraphBuilder.build()
    return graph


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_tracing_initialized()
    yield


# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="Latest AI Development",
    version="0.1.0",
    description="LangGraph Research + Reporting Workflow",
    root_path=CONTEXT_PATH,
    lifespan=lifespan,
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
        result = get_graph().invoke(
            {
                "query": request.query,
                "research_result": "",
                "report": "",
            }
        )

        return AskResponse(
            query=request.query,
            report=result["report"],
        )

    except Exception as e:
        return AskResponse(
            query=request.query,
            report=f"Error: {str(e)}",
        )


# --------------------------------------------------
# CLI Test
# --------------------------------------------------

def test():

    result = get_graph().invoke(
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

    if len(sys.argv) < 2:
        raise ValueError("Provide JSON payload")

    payload = json.loads(sys.argv[1])

    result = get_graph().invoke(
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
        app,
        host="0.0.0.0",
        port=8000,
    )


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )


if CONTEXT_PATH:
    app.add_api_route(
        f"{CONTEXT_PATH}/health",
        health,
        methods=["GET"],
        include_in_schema=False,
    )
    app.add_api_route(
        f"{CONTEXT_PATH}/ask",
        ask,
        methods=["POST"],
        include_in_schema=False,
    )
