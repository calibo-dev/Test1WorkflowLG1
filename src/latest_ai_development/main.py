from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from latest_ai_development.config.settings import get_settings
from latest_ai_development.config.validators import validate_configuration
from latest_ai_development.graph.state import State
from latest_ai_development.logging_utils import configure_logging
from latest_ai_development.tracking import initialize_langsmith_tracing
from latest_ai_development.workflow import LatestAiDevelopmentWorkflow


def _initialize_settings():
    runtime_settings = get_settings()
    configure_logging(getattr(runtime_settings, "log_level", "INFO"))
    return runtime_settings


settings = _initialize_settings()
logger = logging.getLogger(__name__)


def _validate_startup_configuration() -> None:
    try:
        validate_configuration()
    except Exception:
        logger.exception("Startup configuration validation failed")
        raise


def _initialize_workflow() -> LatestAiDevelopmentWorkflow:
    try:
        return LatestAiDevelopmentWorkflow()
    except Exception:
        logger.exception("Workflow initialization failed")
        raise


_validate_startup_configuration()
workflow = _initialize_workflow()
API_ROOT_PATH = settings.normalized_api_root_path


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_langsmith_tracing()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="LangGraph workflow for automated research and report generation",
    root_path=API_ROOT_PATH,
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    topic: str


class AskResponse(BaseModel):
    topic: str
    report: str


def get_graph():
    return workflow.get_graph()


def _build_initial_state(*, topic: str) -> State:
    return workflow.build_initial_state(topic=topic)


def _invoke_workflow(*, topic: str) -> State:
    return workflow.kickoff({"topic": topic})


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    flow_run_id = uuid4().hex[:12]
    logger.info(
        "Request started | flow_run_id=%s | topic=%s",
        flow_run_id,
        request.topic,
    )

    try:
        result = workflow.kickoff({"topic": request.topic}, flow_run_id=flow_run_id)
    except Exception:
        logger.exception(
            "Request failed | flow_run_id=%s | topic=%s",
            flow_run_id,
            request.topic,
        )
        content = {
            "status": "error",
            "response": "Something went wrong. Please contact your administrator.",
        }
        return JSONResponse(status_code=500, content=content)

    logger.info(
        "Request completed | flow_run_id=%s | topic=%s",
        flow_run_id,
        request.topic,
    )
    return AskResponse(
        topic=request.topic,
        report=str(result.get(workflow.final_output_field, "")),
    )


def test() -> None:
    result = _invoke_workflow(topic="Latest AI Agent Frameworks")
    print(result.get(workflow.final_output_field, ""))


def run_with_trigger() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Provide JSON payload")

    payload = json.loads(sys.argv[1])
    result = workflow.kickoff(payload)
    print(result)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "latest_ai_development.main:app",
        host=settings.api_host,
        port=settings.port,
        reload=settings.debug,
        log_level=str(settings.log_level).lower(),
    )


if __name__ == "__main__":
    run()


if API_ROOT_PATH:
    app.add_api_route(
        f"{API_ROOT_PATH}/health",
        health,
        methods=["GET"],
        include_in_schema=False,
    )
    app.add_api_route(
        f"{API_ROOT_PATH}/ask",
        ask,
        methods=["POST"],
        include_in_schema=False,
    )
