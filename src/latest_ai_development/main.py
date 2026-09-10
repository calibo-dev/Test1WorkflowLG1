from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

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
        _validate_startup_configuration()
        return LatestAiDevelopmentWorkflow()
    except Exception:
        logger.exception("Workflow initialization failed")
        raise


API_ROOT_PATH = settings.normalized_api_root_path


@asynccontextmanager
async def lifespan(application: FastAPI):
    initialize_langsmith_tracing()
    runtime_workflow = _initialize_workflow()
    try:
        await runtime_workflow.astart()
        application.state.workflow = runtime_workflow
        yield
    finally:
        await runtime_workflow.aclose()
        if hasattr(application.state, "workflow"):
            del application.state.workflow


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="LangGraph workflow for routed customer feedback responses",
    root_path=API_ROOT_PATH,
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    user_input: str
    thread_id: str | None = None


class AskResponse(BaseModel):
    thread_id: str
    status: str
    final_response: str
    interrupts: list[Any] = Field(default_factory=list)


class ResumeRequest(BaseModel):
    thread_id: str
    value: Any


def _get_workflow(request: Request) -> LatestAiDevelopmentWorkflow:
    workflow = getattr(request.app.state, "workflow", None)
    if not isinstance(workflow, LatestAiDevelopmentWorkflow):
        raise RuntimeError("Workflow runtime is not initialized")
    return workflow


def _build_response(
    result: State,
    *,
    workflow: LatestAiDevelopmentWorkflow,
    thread_id: str,
) -> AskResponse:
    raw_interrupts = result.get("__interrupt__", [])  # type: ignore[typeddict-item]
    interrupts = [getattr(item, "value", item) for item in raw_interrupts]
    return AskResponse(
        thread_id=thread_id,
        status="interrupted" if interrupts else "completed",
        final_response=str(result.get(workflow.final_output_field, "")),
        interrupts=interrupts,
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, http_request: Request):
    flow_run_id = uuid4().hex[:12]
    thread_id = request.thread_id or flow_run_id
    workflow = _get_workflow(http_request)
    logger.info(
        "Request started | flow_run_id=%s | thread_id=%s",
        flow_run_id,
        thread_id,
    )

    try:
        result = await workflow.akickoff(
            {"user_input": request.user_input},
            flow_run_id=flow_run_id,
            thread_id=thread_id,
        )
    except Exception as exc:
        logger.exception(
            "Request failed | flow_run_id=%s | thread_id=%s",
            flow_run_id,
            thread_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed. flow_run_id={flow_run_id}",
        ) from exc

    logger.info(
        "Request completed | flow_run_id=%s | thread_id=%s",
        flow_run_id,
        thread_id,
    )
    return _build_response(result, workflow=workflow, thread_id=thread_id)


@app.post("/resume", response_model=AskResponse)
async def resume(request: ResumeRequest, http_request: Request):
    workflow = _get_workflow(http_request)
    try:
        result = await workflow.aresume(thread_id=request.thread_id, value=request.value)
    except Exception as exc:
        logger.exception("Workflow resume failed | thread_id=%s", request.thread_id)
        raise HTTPException(status_code=500, detail="Workflow resume failed") from exc
    return _build_response(result, workflow=workflow, thread_id=request.thread_id)


async def _run_cli_workflow(inputs: dict[str, Any]) -> State:
    runtime_workflow = _initialize_workflow()
    await runtime_workflow.astart()
    try:
        return await runtime_workflow.akickoff(inputs)
    finally:
        await runtime_workflow.aclose()


def test() -> None:
    result = asyncio.run(_run_cli_workflow({"user_input": "Latest AI Agent Frameworks"}))
    print(result.get("final_response", ""))


def run_with_trigger() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Provide JSON payload")

    payload = json.loads(sys.argv[1])
    result = asyncio.run(_run_cli_workflow(payload))
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
        f"{API_ROOT_PATH}/resume",
        resume,
        methods=["POST"],
        include_in_schema=False,
    )
    app.add_api_route(
        f"{API_ROOT_PATH}/ask",
        ask,
        methods=["POST"],
        include_in_schema=False,
    )
