"""
Always-on gateway for the PDF inference pipeline.

- Probes Chandra OCR (/health) and main FastAPI (/health, /stats).
- Proxies traffic when upstreams are ready; otherwise queues mutating requests.
- Queued request bodies in S3 (or local); job metadata in Supabase or SQLite.
- Drains jobs when the target upstream recovers; clients poll GET /jobs/{id}.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

# ---------------------------------------------------------------------------
# Load .env.local (micro-server does not auto-load unlike start.sh)
# ---------------------------------------------------------------------------


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing vars."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


ENV_FILE = Path(os.getenv("ENV_FILE", ".env.local"))
_load_env_file(ENV_FILE)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8001").rstrip("/")
CHANDRA_URL = os.getenv("CHANDRA_URL", "http://localhost:8000").rstrip("/")
MICRO_HOST = os.getenv("MICRO_HOST", "0.0.0.0")
MICRO_PORT = int(os.getenv("MICRO_PORT", "8080"))

FASTAPI_HEALTH_PATH = os.getenv("FASTAPI_HEALTH_PATH", "/health")
FASTAPI_STATS_PATH = os.getenv("FASTAPI_STATS_PATH", "/stats")
CHANDRA_HEALTH_PATH = os.getenv("CHANDRA_HEALTH_PATH", "/health")
CHANDRA_MODELS_PATH = os.getenv("CHANDRA_MODELS_PATH", "/v1/models")

HEALTH_INTERVAL_S = float(os.getenv("HEALTH_INTERVAL_S", "10"))
HEALTH_TIMEOUT_S = float(os.getenv("HEALTH_TIMEOUT_S", "10"))
PIPELINE_REQUIRE_QWEN = os.getenv("PIPELINE_REQUIRE_QWEN", "true").lower() in {
    "1",
    "true",
    "yes",
}
TRUST_FASTAPI_CHANDRA_HEALTH = os.getenv(
    "TRUST_FASTAPI_CHANDRA_HEALTH", "true"
).lower() in {"1", "true", "yes"}
CHANDRA_DIRECT_PROBE_TIMEOUT_S = float(
    os.getenv("CHANDRA_DIRECT_PROBE_TIMEOUT_S", "3")
)

OFFLINE_MESSAGE = os.getenv(
    "OFFLINE_MESSAGE",
    "The analysis pipeline is temporarily offline. Your request has been queued "
    "and will run automatically when services are back.",
)
CHANDRA_OFFLINE_MESSAGE = os.getenv(
    "CHANDRA_OFFLINE_MESSAGE",
    "Chandra OCR is temporarily offline. Your request has been queued "
    "and will run automatically when the service is back.",
)

# Optional: first path segments proxied directly to CHANDRA_URL (not FastAPI).
# Chandra is a vLLM server (/v1/chat/completions only) — app routes like /extract
# must go to FastAPI, which calls Chandra internally. Default: none.
CHANDRA_PROXY_PATHS: frozenset[str] = frozenset(
    part.strip().strip("/")
    for part in os.getenv("CHANDRA_PROXY_PATHS", "").split(",")
    if part.strip()
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("micro-server")

from gateway_jobs_store import (
    QUEUE_DB,
    S3_BUCKET,
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    build_multipart_body,
    create_job,
    drain_jobs_once,
    get_job,
    init_db as _init_jobs_db,
    job_as_dict,
    job_public_dict,
    jobs_backend,
    list_jobs,
    storage_backend,
)


# ---------------------------------------------------------------------------
# Upstream status (updated by background poller)
# ---------------------------------------------------------------------------

_upstream_lock = asyncio.Lock()
_upstream: dict[str, Any] = {
    "fastapi_reachable": False,
    "fastapi_stats_ok": False,
    "chandra_reachable": False,
    "pipeline_ready": False,
    "last_checked_at": None,
    "fastapi_health": None,
    "chandra_health": None,
    "error": None,
}

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_first_segment(path: str) -> str:
    """Return the first segment of a URL path (no leading/trailing slashes)."""
    return path.strip("/").split("/", 1)[0] if path.strip("/") else ""


def _is_chandra_path(path: str) -> bool:
    """True when this request should be proxied to Chandra OCR, not FastAPI."""
    return _path_first_segment(path) in CHANDRA_PROXY_PATHS


def _upstream_base_url(path: str) -> str:
    return CHANDRA_URL if _is_chandra_path(path) else FASTAPI_URL


def _fastapi_reports_chandra_healthy(fastapi_health: Any) -> bool:
    if not isinstance(fastapi_health, dict):
        return False
    services = fastapi_health.get("services") or {}
    return (services.get("chandra_ocr") or {}).get("status") == "healthy"


def _effective_chandra_reachable(
    fastapi_ok: bool,
    fastapi_health: Any,
    chandra_direct_ok: bool,
) -> bool:
    if chandra_direct_ok:
        return True
    if TRUST_FASTAPI_CHANDRA_HEALTH and fastapi_ok:
        return _fastapi_reports_chandra_healthy(fastapi_health)
    return False


def _service_ready(path: str, upstream: dict[str, Any]) -> bool:
    """Check whether the target upstream for this path is ready."""
    if _is_chandra_path(path):
        return bool(upstream.get("chandra_direct_reachable"))
    return bool(upstream.get("pipeline_ready"))


def _offline_message(path: str, upstream: dict[str, Any] | None = None) -> str:
    if _is_chandra_path(path):
        if upstream and upstream.get("chandra_via_fastapi_health"):
            return (
                "Chandra OCR is healthy on the reasoning server but not directly "
                "reachable from this gateway. Open the security group for "
                f"{CHANDRA_URL} from this gateway host, or set CHANDRA_URL to a "
                "private IP reachable within your VPC."
            )
        return CHANDRA_OFFLINE_MESSAGE
    return OFFLINE_MESSAGE


def _queued_response(
    job_id: str,
    path: str,
    upstream_snapshot: dict[str, Any],
    route_target: str,
) -> JSONResponse:
    return JSONResponse(
        {
            "status": "queued",
            "job_id": job_id,
            "message": _offline_message(path, upstream_snapshot),
            "poll_url": f"/jobs/{job_id}",
            "route_target": route_target,
            "retry_after_seconds": int(HEALTH_INTERVAL_S),
            "pipeline_blockers": upstream_snapshot.get("pipeline_blockers") or [],
        },
        status_code=202,
        headers={"Retry-After": str(int(HEALTH_INTERVAL_S))},
    )


async def _forward_analyze(
    pdf_bytes: bytes,
    filename: str,
    question: str,
) -> Response | StreamingResponse | JSONResponse:
    return await _forward_upstream(
        method="POST",
        url=f"{FASTAPI_URL}/analyze",
        headers={},
        body=b"",
        upstream_label="FastAPI",
        multipart_files={
            "file": (filename, pdf_bytes, "application/pdf"),
        },
        multipart_data={"question": question},
    )


# ---------------------------------------------------------------------------
# Health probing
# ---------------------------------------------------------------------------

async def _probe_json(client: httpx.AsyncClient, url: str) -> tuple[bool, Any]:
    try:
        resp = await client.get(url)
        if resp.status_code >= 500:
            return False, {"status_code": resp.status_code, "body": resp.text[:500]}
        try:
            payload = resp.json()
        except Exception:
            payload = {"status_code": resp.status_code, "body": resp.text[:500]}
        return resp.status_code < 500, payload
    except httpx.RequestError as exc:
        return False, {"error": str(exc)}


async def _probe_chandra(client: httpx.AsyncClient) -> tuple[bool, Any]:
    """Probe Chandra; fall back to /v1/models when /health is missing or failing."""
    ok, payload = await _probe_json(client, f"{CHANDRA_URL}{CHANDRA_HEALTH_PATH}")
    if ok:
        return True, payload

    models_ok, models_payload = await _probe_json(
        client, f"{CHANDRA_URL}{CHANDRA_MODELS_PATH}"
    )
    if models_ok:
        return True, models_payload

    return False, payload


def _chandra_payload_healthy(payload: Any) -> bool:
    """Interpret a Chandra /health or /v1/models probe payload."""
    if not isinstance(payload, dict):
        return True

    if payload.get("data"):
        return True

    status = str(payload.get("status", "")).lower()
    if not status:
        return True
    return status in {"healthy", "ok", "up"}


def _pipeline_blockers(
    fastapi_ok: bool,
    fastapi_health: Any,
    chandra_direct_ok: bool,
    chandra_health: Any,
) -> list[str]:
    blockers: list[str] = []

    if not fastapi_ok:
        detail = fastapi_health if isinstance(fastapi_health, dict) else {}
        err = detail.get("error") or detail.get("status_code") or "unreachable"
        blockers.append(f"fastapi_unreachable ({err})")

    chandra_ok = _effective_chandra_reachable(
        fastapi_ok, fastapi_health, chandra_direct_ok
    )
    if not chandra_ok:
        if not chandra_direct_ok and TRUST_FASTAPI_CHANDRA_HEALTH and fastapi_ok:
            if not _fastapi_reports_chandra_healthy(fastapi_health):
                blockers.append("fastapi_reports_chandra_unhealthy")
            else:
                detail = chandra_health if isinstance(chandra_health, dict) else {}
                err = detail.get("error") or detail.get("status_code") or "timeout"
                blockers.append(f"chandra_unreachable_direct ({err})")
        elif not chandra_direct_ok:
            detail = chandra_health if isinstance(chandra_health, dict) else {}
            err = detail.get("error") or detail.get("status_code") or "unreachable"
            blockers.append(f"chandra_unreachable_direct ({err})")
        else:
            detail = chandra_health if isinstance(chandra_health, dict) else {}
            err = detail.get("error") or detail.get("status_code") or "unreachable"
            blockers.append(f"chandra_unreachable ({err})")

    if isinstance(fastapi_health, dict):
        services = fastapi_health.get("services") or {}
        chandra = (services.get("chandra_ocr") or {}).get("status")
        qwen = (services.get("qwen_reasoning") or {}).get("status")
        if chandra and chandra != "healthy":
            blockers.append(f"fastapi_reports_chandra_{chandra}")
        if PIPELINE_REQUIRE_QWEN and qwen and qwen != "healthy":
            blockers.append(f"fastapi_reports_qwen_{qwen}")

    if chandra_direct_ok and not _chandra_payload_healthy(chandra_health):
        status = (
            chandra_health.get("status")
            if isinstance(chandra_health, dict)
            else chandra_health
        )
        blockers.append(f"chandra_health_status_{status}")

    return blockers


def _pipeline_ready(
    fastapi_ok: bool,
    fastapi_health: Any,
    chandra_direct_ok: bool,
    chandra_health: Any,
) -> bool:
    return not _pipeline_blockers(
        fastapi_ok, fastapi_health, chandra_direct_ok, chandra_health
    )


async def _refresh_upstream_status() -> None:
    async with _upstream_lock:
        fastapi_ok = False
        fastapi_stats_ok = False
        chandra_direct_ok = False
        fastapi_health = None
        chandra_health = None
        error = None

        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT_S) as client:
                fastapi_ok, fastapi_health = await _probe_json(
                    client, f"{FASTAPI_URL}{FASTAPI_HEALTH_PATH}"
                )
                try:
                    stats_resp = await client.get(f"{FASTAPI_URL}{FASTAPI_STATS_PATH}")
                    fastapi_stats_ok = stats_resp.status_code in (200, 404)
                except httpx.RequestError:
                    fastapi_stats_ok = False

            # Short timeout — direct Chandra probe often blocked from gateway SG.
            async with httpx.AsyncClient(
                timeout=CHANDRA_DIRECT_PROBE_TIMEOUT_S
            ) as chandra_client:
                chandra_direct_ok, chandra_health = await _probe_chandra(chandra_client)
        except Exception as exc:
            error = str(exc)
            logger.warning("Health probe failed: %s", exc)

        chandra_ok = _effective_chandra_reachable(
            fastapi_ok, fastapi_health, chandra_direct_ok
        )
        blockers = _pipeline_blockers(
            fastapi_ok, fastapi_health, chandra_direct_ok, chandra_health
        )
        ready = not blockers

        if blockers:
            logger.warning(
                "Pipeline not ready (FASTAPI_URL=%s CHANDRA_URL=%s): %s",
                FASTAPI_URL,
                CHANDRA_URL,
                "; ".join(blockers),
            )
        elif not chandra_direct_ok and chandra_ok:
            logger.info(
                "Chandra not directly reachable from gateway; using FastAPI /health "
                "report (chandra_ocr=healthy). CHANDRA_PROXY_PATHS traffic still needs "
                "direct network access to CHANDRA_URL."
            )

        _upstream.update(
            {
                "fastapi_reachable": fastapi_ok,
                "fastapi_stats_ok": fastapi_stats_ok,
                "chandra_direct_reachable": chandra_direct_ok,
                "chandra_reachable": chandra_ok,
                "chandra_via_fastapi_health": (
                    not chandra_direct_ok
                    and chandra_ok
                    and TRUST_FASTAPI_CHANDRA_HEALTH
                ),
                "pipeline_ready": ready,
                "pipeline_blockers": blockers,
                "last_checked_at": _now_iso(),
                "fastapi_health": fastapi_health,
                "chandra_health": chandra_health,
                "error": error,
            }
        )


async def _forward_upstream(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    upstream_label: str,
    multipart_files: dict[str, tuple[str, bytes, str]] | None = None,
    multipart_data: dict[str, str] | None = None,
) -> Response | StreamingResponse | JSONResponse:
    """Proxy an HTTP request to an upstream and return the client response."""
    client = httpx.AsyncClient(timeout=None)
    try:
        if multipart_files or multipart_data:
            upstream_req = client.build_request(
                method,
                url,
                headers=headers,
                files=multipart_files or {},
                data=multipart_data or {},
            )
        else:
            upstream_req = client.build_request(method, url, headers=headers, content=body)
        upstream_resp = await client.send(upstream_req, stream=True)

        resp_headers = {
            k: v
            for k, v in upstream_resp.headers.items()
            if k.lower() not in HOP_BY_HOP
        }

        content_type = upstream_resp.headers.get("content-type", "")
        if "text/event-stream" in content_type or (
            upstream_resp.headers.get("transfer-encoding") == "chunked"
            and content_type.startswith("text/")
        ):
            async def stream_body():
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        yield chunk
                finally:
                    await upstream_resp.aclose()
                    await client.aclose()

            return StreamingResponse(
                stream_body(),
                status_code=upstream_resp.status_code,
                headers=resp_headers,
                media_type=content_type,
            )

        content = await upstream_resp.aread()
        await upstream_resp.aclose()
        await client.aclose()
        return Response(
            content=content,
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=content_type or None,
        )
    except httpx.RequestError as exc:
        await client.aclose()
        logger.error("Proxy error for %s (%s): %s", url, upstream_label, exc)
        return JSONResponse(
            {
                "error": "upstream_unreachable",
                "message": f"{upstream_label} became unreachable during proxy.",
                "detail": str(exc),
            },
            status_code=502,
        )


async def _health_poller() -> None:
    while True:
        await _refresh_upstream_status()
        async with _upstream_lock:
            upstream_snapshot = _upstream.copy()
        await drain_jobs_once(
            upstream_snapshot,
            _service_ready,
            _upstream_base_url,
        )
        await asyncio.sleep(HEALTH_INTERVAL_S)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_jobs_db()
    await _refresh_upstream_status()
    poller = asyncio.create_task(_health_poller())
    logger.info(
        "Micro server started — env_file=%s (exists=%s) FASTAPI_URL=%s "
        "CHANDRA_URL=%s port=%s chandra_paths=%s job_db=%s job_storage=%s",
        ENV_FILE,
        ENV_FILE.is_file(),
        FASTAPI_URL,
        CHANDRA_URL,
        MICRO_PORT,
        sorted(CHANDRA_PROXY_PATHS),
        jobs_backend(),
        storage_backend(),
    )
    yield
    poller.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await poller


app = FastAPI(
    title="Pipeline Gateway",
    description="Always-on proxy with offline queue for the PDF inference pipeline",
    version="1.3.0",
    lifespan=lifespan,
)


@app.get("/health")
async def gateway_health():
    async with _upstream_lock:
        status = _upstream.copy()
    return {
        "status": "healthy"
        if (status["pipeline_ready"] or status["chandra_reachable"])
        else "degraded",
        "gateway": "up",
        "pipeline_ready": status["pipeline_ready"],
        "pipeline_blockers": status.get("pipeline_blockers") or [],
        "chandra_ready": status["chandra_reachable"],
        "chandra_direct_reachable": status.get("chandra_direct_reachable", False),
        "chandra_via_fastapi_health": status.get("chandra_via_fastapi_health", False),
        "chandra_proxy_paths": sorted(CHANDRA_PROXY_PATHS),
        "pipeline_require_qwen": PIPELINE_REQUIRE_QWEN,
        "trust_fastapi_chandra_health": TRUST_FASTAPI_CHANDRA_HEALTH,
        "upstream": status,
        "queue_db": str(QUEUE_DB),
        "job_db": jobs_backend(),
        "job_storage": storage_backend(),
        "s3_bucket": S3_BUCKET or None,
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY),
    }


@app.get("/jobs")
async def list_gateway_jobs():
    rows = list_jobs()
    return {"jobs": [job_as_dict(r) for r in rows]}


@app.get("/jobs/{job_id}")
async def get_gateway_job(job_id: str):
    row = get_job(job_id)
    if row is None:
        return JSONResponse({"error": "job_not_found", "job_id": job_id}, status_code=404)
    return job_public_dict(row)


@app.post("/analyze")
async def analyze_gateway(
    file: UploadFile = File(..., description="PDF file to analyze"),
    question: str = Form(..., description="Question to ask about the document"),
    callback_url: str | None = Form(
        None, description="Optional URL to POST the result when processing completes"
    ),
    user_id: str | None = Form(None, description="Optional client user identifier"),
):
    """Proxy /analyze when online; queue when the pipeline is offline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "file_must_be_pdf"}, status_code=400)

    pdf_bytes = await file.read()
    if not pdf_bytes:
        return JSONResponse({"error": "empty_file"}, status_code=400)

    filename = file.filename or "document.pdf"

    async with _upstream_lock:
        pipeline_ready = bool(_upstream.get("pipeline_ready"))

    if pipeline_ready:
        return await _forward_analyze(pdf_bytes, filename, question)

    form_data: dict[str, str | None] = {"question": question}
    if callback_url:
        form_data["callback_url"] = callback_url
    if user_id:
        form_data["user_id"] = user_id

    try:
        body, headers = build_multipart_body(
            files={"file": (filename, pdf_bytes, "application/pdf")},
            data=form_data,
        )
        job_id = create_job(
            method="POST",
            path="analyze",
            query_string="",
            route_target="fastapi",
            headers=headers,
            body=body,
            callback_url=callback_url,
            user_id=user_id,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=413)

    async with _upstream_lock:
        upstream_snapshot = _upstream.copy()
    return _queued_response(job_id, "analyze", upstream_snapshot, "fastapi")


@app.get("/queue")
async def list_queue():
    """Legacy alias for GET /jobs."""
    rows = list_jobs()
    return {"jobs": [job_as_dict(r) for r in rows]}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def gateway_proxy(request: Request, path: str):
    if path in {"health", "queue", "jobs", "analyze", "docs", "openapi.json", "redoc"}:
        return JSONResponse({"error": "Use top-level gateway routes only."}, status_code=404)
    if path.startswith("jobs/"):
        return JSONResponse({"error": "Use GET /jobs/{job_id}."}, status_code=404)

    async with _upstream_lock:
        upstream_snapshot = _upstream.copy()

    ready = _service_ready(path, upstream_snapshot)
    route_target = "chandra" if _is_chandra_path(path) else "fastapi"

    if not ready:
        if request.method in {"POST", "PUT", "PATCH"}:
            body = await request.body()
            headers = {
                k: v
                for k, v in request.headers.items()
                if k.lower() not in HOP_BY_HOP
            }
            try:
                job_id = create_job(
                    method=request.method,
                    path=path,
                    query_string=request.url.query,
                    route_target=route_target,
                    headers=headers,
                    body=body,
                    callback_url=request.headers.get("x-callback-url"),
                    user_id=request.headers.get("x-user-id"),
                )
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=413)

            return _queued_response(job_id, path, upstream_snapshot, route_target)

        offline_error = "chandra_offline" if _is_chandra_path(path) else "pipeline_offline"
        return JSONResponse(
            {
                "error": offline_error,
                "route_target": route_target,
                "message": _offline_message(path, upstream_snapshot),
                "pipeline_blockers": upstream_snapshot.get("pipeline_blockers") or [],
                "upstream": upstream_snapshot,
            },
            status_code=503,
            headers={"Retry-After": str(int(HEALTH_INTERVAL_S))},
        )

    base_url = _upstream_base_url(path)
    url = f"{base_url}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP
    }
    body = await request.body()

    upstream_label = "Chandra OCR" if _is_chandra_path(path) else "FastAPI"
    logger.info(
        "Proxy %s /%s → %s (%s)",
        request.method,
        path,
        url,
        upstream_label,
    )
    return await _forward_upstream(
        method=request.method,
        url=url,
        headers=headers,
        body=body,
        upstream_label=upstream_label,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "micro-server:app",
        host=MICRO_HOST,
        port=MICRO_PORT,
        reload=False,
    )
