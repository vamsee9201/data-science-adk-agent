from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from app.agent import agent_runtime
from app.analytics import apply_proposal, modeling_request_preflight, reject_proposal, reset_dataset
from app.config import settings
from app.data import DatasetValidationError, dataset_metadata, load_sample, parse_csv, sample_catalog
from app.guardrails import UsageLimitExceeded, usage_guard
from app.sessions import SessionCapacityExceeded, SessionNotFound, registry


STATIC_DIR = Path(__file__).with_name("static")
LOGGER = logging.getLogger("data_science_agent")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    _log_handler = logging.StreamHandler()
    _log_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    LOGGER.addHandler(_log_handler)
LOGGER.propagate = False


class SampleRequest(BaseModel):
    sample_id: str
    confirm_replace: bool = False


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)


class ReplaceSessionRequest(BaseModel):
    previous_session_id: str | None = None


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(60)
        registry.cleanup_expired()


@asynccontextmanager
async def lifespan(_: FastAPI):
    usage_guard.reset()
    cleanup_task = asyncio.create_task(_cleanup_loop())
    yield
    cleanup_task.cancel()
    with suppress(asyncio.CancelledError):
        await cleanup_task
    registry.close()
    usage_guard.reset()


app = FastAPI(
    title="Data Science Agent",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.expose_api_docs else None,
    redoc_url="/redoc" if settings.expose_api_docs else None,
    openapi_url="/openapi.json" if settings.expose_api_docs else None,
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
        "connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.exception_handler(SessionNotFound)
async def session_not_found_handler(_, __):
    return JSONResponse(status_code=404, content={"detail": "Session not found or expired."})


@app.exception_handler(SessionCapacityExceeded)
async def session_capacity_handler(_, exc: SessionCapacityExceeded):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(UsageLimitExceeded)
async def usage_limit_handler(_, exc: UsageLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc)},
        headers={"Retry-After": str(exc.retry_after)},
    )


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "vertex_configured": settings.vertex_configured,
        "model": settings.google_model,
    }


@app.post("/api/sessions", status_code=201)
async def create_session() -> dict:
    session = registry.create()
    try:
        await agent_runtime.create_session(session.id)
    except Exception:
        registry.delete(session.id)
        raise HTTPException(status_code=503, detail="The agent session could not be initialized.")
    return {
        "session_id": session.id,
        "expires_in_seconds": registry.ttl_seconds,
        "usage": usage_guard.snapshot(session),
    }


@app.post("/api/sessions/replace", status_code=201)
async def replace_session(request: ReplaceSessionRequest) -> dict:
    """Create the replacement before retiring the previous session.

    This keeps reload and New session atomic: if initialization fails, the old workspace remains
    available instead of leaving the browser without a usable session.
    """
    previous_id = request.previous_session_id
    session = registry.create(replacing_session_id=previous_id)
    try:
        await agent_runtime.create_session(session.id)
    except Exception:
        registry.delete(session.id)
        raise HTTPException(status_code=503, detail="The agent session could not be initialized.")

    if previous_id and previous_id != session.id:
        try:
            previous = registry.get(previous_id, touch=False)
            previous.cancel_requested = True
            await agent_runtime.delete_session(previous_id)
            registry.delete(previous_id)
        except SessionNotFound:
            pass
    return {
        "session_id": session.id,
        "expires_in_seconds": registry.ttl_seconds,
        "usage": usage_guard.snapshot(session),
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    registry.get(session_id, touch=False)
    await agent_runtime.delete_session(session_id)
    registry.delete(session_id)
    return {"deleted": True}


@app.get("/api/samples")
async def list_samples() -> dict:
    return {"samples": sample_catalog()}


def _ensure_replace_allowed(session_id: str, confirmed: bool):
    session = registry.get(session_id)
    if session.busy:
        raise HTTPException(status_code=409, detail="Wait for the current analysis turn to finish.")
    if session.dataframe is not None and not confirmed:
        raise HTTPException(
            status_code=409,
            detail="A dataset is already active. Confirm replacement before loading another.",
        )
    return session


@app.post("/api/sessions/{session_id}/datasets/upload")
async def upload_dataset(
    session_id: str,
    file: UploadFile = File(...),
    confirm_replace: bool = Query(False),
) -> dict:
    session = _ensure_replace_allowed(session_id, confirm_replace)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Choose a file with a .csv extension.")
    content = await file.read(settings.max_upload_bytes + 1)
    try:
        dataframe = parse_csv(
            content,
            settings.max_upload_bytes,
            max_rows=settings.max_dataset_rows,
            max_columns=settings.max_dataset_columns,
        )
    except DatasetValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        await agent_runtime.reset_session(session_id)
    except Exception:
        raise HTTPException(status_code=503, detail="The agent session could not be reset.")
    upload_path = session.directory / "uploaded.csv"
    upload_path.write_bytes(content)
    session.set_dataset(file.filename, dataframe)
    return dataset_metadata(session.dataset_name, dataframe)


@app.post("/api/sessions/{session_id}/datasets/sample")
async def select_sample(session_id: str, request: SampleRequest) -> dict:
    session = _ensure_replace_allowed(session_id, request.confirm_replace)
    try:
        name, dataframe = load_sample(request.sample_id)
    except DatasetValidationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        await agent_runtime.reset_session(session_id)
    except Exception:
        raise HTTPException(status_code=503, detail="The agent session could not be reset.")
    session.set_dataset(name, dataframe)
    return dataset_metadata(name, dataframe)


@app.get("/api/sessions/{session_id}/dataset")
async def get_dataset(session_id: str) -> dict:
    session = registry.get(session_id)
    dataframe = session.require_dataframe()
    return dataset_metadata(session.dataset_name or "Dataset", dataframe)


@app.post("/api/sessions/{session_id}/chat")
async def chat(session_id: str, request: ChatRequest) -> StreamingResponse:
    session = registry.get(session_id)
    dataframe = session.require_dataframe()
    if session.busy:
        raise HTTPException(status_code=409, detail="An analysis turn is already running.")
    preflight = modeling_request_preflight(dataframe, request.message.strip())
    if preflight:
        async def blocked_stream() -> AsyncIterator[bytes]:
            message = preflight["message"]
            for event in (
                {"event": "warning", "message": message, "modeling_blocked": True},
                {"event": "message", "message": message},
                {
                    "event": "turn_complete",
                    "status": "blocked",
                    "usage": usage_guard.snapshot(session),
                },
            ):
                yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            LOGGER.info(
                "agent_turn_preflight_blocked session=%s requested=%s detected=%s",
                hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12],
                preflight["requested_task_type"],
                preflight["detected_task_type"],
            )

        return StreamingResponse(blocked_stream(), media_type="application/x-ndjson")
    usage_guard.acquire(session)
    session.busy = True
    session.cancel_requested = False

    async def stream() -> AsyncIterator[bytes]:
        started = time.monotonic()
        terminal_status = "completed"
        last_tool = "none"
        try:
            async for event in agent_runtime.stream(session_id, request.message.strip()):
                if event.get("event") == "tool_status":
                    last_tool = str(event.get("tool") or "unknown")
                if event.get("status") in {"cancelled", "failed", "timeout"}:
                    terminal_status = str(event["status"])
                yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        except asyncio.CancelledError:
            terminal_status = "cancelled"
            raise
        finally:
            usage_guard.release()
            try:
                workspace = registry.get(session_id, touch=False)
                workspace.busy = False
                workspace.cancel_requested = False
            except SessionNotFound:
                workspace = session
            session_key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
            LOGGER.info(
                "agent_turn_complete session=%s turn=%d status=%s last_tool=%s elapsed_ms=%d",
                session_key,
                workspace.agent_turns,
                terminal_status,
                last_tool,
                int((time.monotonic() - started) * 1000),
            )
        yield (
            json.dumps(
                {
                    "event": "turn_complete",
                    "status": terminal_status,
                    "usage": usage_guard.snapshot(workspace),
                },
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/sessions/{session_id}/chat/cancel")
async def cancel_chat(session_id: str) -> dict:
    session = registry.get(session_id)
    if not session.busy:
        raise HTTPException(status_code=409, detail="No analysis turn is currently running.")
    session.cancel_requested = True
    return {"cancel_requested": True}


@app.post("/api/sessions/{session_id}/transformations/{proposal_id}/apply")
async def apply_transformation(session_id: str, proposal_id: str) -> dict:
    session = registry.get(session_id)
    if session.busy:
        raise HTTPException(status_code=409, detail="Wait for the current analysis turn to finish.")
    try:
        result = apply_proposal(session, proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result["dataset"] = dataset_metadata(session.dataset_name or "Dataset", session.require_dataframe())
    return result


@app.delete("/api/sessions/{session_id}/transformations/{proposal_id}")
async def reject_transformation(session_id: str, proposal_id: str) -> dict:
    session = registry.get(session_id)
    if session.busy:
        raise HTTPException(status_code=409, detail="Wait for the current analysis turn to finish.")
    try:
        return reject_proposal(session, proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/dataset/reset")
async def restore_dataset(session_id: str) -> dict:
    session = registry.get(session_id)
    if session.busy:
        raise HTTPException(status_code=409, detail="Wait for the current analysis turn to finish.")
    try:
        reset_dataset(session)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return dataset_metadata(session.dataset_name or "Dataset", session.require_dataframe())


@app.get("/api/sessions/{session_id}/artifacts/{artifact_id}")
async def get_artifact(session_id: str, artifact_id: str) -> FileResponse:
    session = registry.get(session_id)
    path = session.artifacts.get(artifact_id)
    if path is None or not path.is_file() or path.parent != session.directory:
        raise HTTPException(status_code=404, detail="Artifact not found or expired.")
    return FileResponse(path, media_type="image/png", filename=path.name)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
