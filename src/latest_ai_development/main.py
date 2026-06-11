from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from latest_ai_development.graph.graph_builder import GraphBuilder


# --------------------------------------------------
# Initialization
# --------------------------------------------------

graph = GraphBuilder.build()


# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="Latest AI Development",
    version="0.1.0",
    description="LangGraph Research + Reporting Workflow",
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