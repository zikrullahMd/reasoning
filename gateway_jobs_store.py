"""Unified offline job queue storage (Supabase or SQLite) + S3/local bodies."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

logger = logging.getLogger("micro-server")

QUEUE_DB = Path(os.getenv("QUEUE_DB", "micro_queue.db"))
QUEUE_DIR = Path(os.getenv("QUEUE_DIR", "micro_queue_files"))
MAX_QUEUE_BYTES = int(os.getenv("MAX_QUEUE_BYTES", str(50 * 1024 * 1024)))
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_PREFIX = os.getenv("S3_PREFIX", "offline-jobs").strip().strip("/")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
JOB_MAX_RETRIES = int(os.getenv("JOB_MAX_RETRIES", "3"))
JOB_DRAIN_CONCURRENCY = int(os.getenv("JOB_DRAIN_CONCURRENCY", "2"))
JOB_RESULT_LOCAL_THRESHOLD = int(
    os.getenv("JOB_RESULT_LOCAL_THRESHOLD", str(256 * 1024))
)
JOB_CALLBACK_TIMEOUT_S = float(os.getenv("JOB_CALLBACK_TIMEOUT_S", "30"))
SUPABASE_URL = (
    os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
).strip().rstrip("/")
SUPABASE_SERVICE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_SERVICE_KEY")
    or os.getenv("NEXT_PUBLIC_SUPABASE_SERVICE_KEY", "")
).strip()

GATEWAY_JOBS_TABLE = "gateway_jobs"
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

JobRow = dict[str, Any] | sqlite3.Row
_s3_client: Any = None
_supabase_client: Any = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def jobs_backend() -> str:
    return "supabase" if SUPABASE_URL and SUPABASE_SERVICE_KEY else "sqlite"


def storage_backend() -> str:
    return "s3" if S3_BUCKET else "local"


def init_db() -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(QUEUE_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gateway_jobs (
                id TEXT PRIMARY KEY,
                method TEXT NOT NULL,
                path TEXT NOT NULL,
                query_string TEXT NOT NULL DEFAULT '',
                route_target TEXT NOT NULL,
                headers_json TEXT NOT NULL DEFAULT '{}',
                content_type TEXT,
                body_storage TEXT NOT NULL,
                body_key TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                result_status_code INTEGER,
                result_content_type TEXT,
                result_storage TEXT,
                result_key TEXT,
                result_text TEXT,
                callback_url TEXT,
                user_id TEXT,
                created_at TEXT NOT NULL,
                processed_at TEXT,
                error TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()


def _get_supabase() -> Any:
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client

        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase_client


def _get_s3_client() -> Any:
    global _s3_client
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def _put_bytes(key_suffix: str, data: bytes, content_type: str) -> tuple[str, str]:
    if S3_BUCKET:
        key = f"{S3_PREFIX}/{key_suffix}"
        _get_s3_client().put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return "s3", key

    local_path = QUEUE_DIR / key_suffix
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    return "local", str(local_path)


def _get_bytes(storage: str, key: str) -> bytes:
    if storage == "s3":
        resp = _get_s3_client().get_object(Bucket=S3_BUCKET, Key=key)
        return resp["Body"].read()
    return Path(key).read_bytes()


def _job_get(row: JobRow, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return row[key]


def job_as_dict(row: JobRow) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    return dict(row)


def _filter_replay_headers(headers: dict[str, str]) -> dict[str, str]:
    skip = HOP_BY_HOP | {"host", "content-length", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in skip}


def build_multipart_body(
    *,
    files: dict[str, tuple[str, bytes, str]],
    data: dict[str, str | None],
) -> tuple[bytes, dict[str, str]]:
    form_data = {k: v for k, v in data.items() if v is not None}
    with httpx.Client() as client:
        req = client.build_request(
            "POST",
            "http://gateway.invalid/",
            files=files,
            data=form_data,
        )
        return req.content, dict(req.headers)


def create_job(
    *,
    method: str,
    path: str,
    query_string: str,
    route_target: str,
    headers: dict[str, str],
    body: bytes,
    callback_url: str | None = None,
    user_id: str | None = None,
) -> str:
    if len(body) > MAX_QUEUE_BYTES:
        raise ValueError(f"Request body exceeds {MAX_QUEUE_BYTES} bytes")

    job_id = str(uuid.uuid4())
    body_storage, body_key = _put_bytes(
        f"requests/{job_id}/body.bin",
        body,
        headers.get("content-type") or "application/octet-stream",
    )
    created_at = _now_iso()
    safe_headers = _filter_replay_headers(headers)
    row = {
        "id": job_id,
        "method": method,
        "path": path,
        "query_string": query_string,
        "route_target": route_target,
        "headers_json": json.dumps(safe_headers),
        "content_type": headers.get("content-type"),
        "body_storage": body_storage,
        "body_key": body_key,
        "status": "queued",
        "callback_url": callback_url,
        "user_id": user_id,
        "created_at": created_at,
        "retry_count": 0,
    }

    if jobs_backend() == "supabase":
        _get_supabase().table(GATEWAY_JOBS_TABLE).insert(row).execute()
    else:
        with sqlite3.connect(QUEUE_DB) as conn:
            conn.execute(
                """
                INSERT INTO gateway_jobs (
                    id, method, path, query_string, route_target, headers_json,
                    content_type, body_storage, body_key, callback_url, user_id,
                    created_at, retry_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    method,
                    path,
                    query_string,
                    route_target,
                    row["headers_json"],
                    row["content_type"],
                    body_storage,
                    body_key,
                    callback_url,
                    user_id,
                    created_at,
                    0,
                ),
            )
            conn.commit()

    logger.info(
        "Queued gateway job %s %s %s (%d bytes, job_db=%s, storage=%s)",
        job_id,
        method,
        path,
        len(body),
        jobs_backend(),
        body_storage,
    )
    return job_id


def get_job(job_id: str) -> JobRow | None:
    if jobs_backend() == "supabase":
        resp = (
            _get_supabase()
            .table(GATEWAY_JOBS_TABLE)
            .select("*")
            .eq("id", job_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None

    with sqlite3.connect(QUEUE_DB) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            f"SELECT * FROM {GATEWAY_JOBS_TABLE} WHERE id = ?",
            (job_id,),
        ).fetchone()


def list_jobs(limit: int = 100) -> list[JobRow]:
    if jobs_backend() == "supabase":
        resp = (
            _get_supabase()
            .table(GATEWAY_JOBS_TABLE)
            .select(
                "id, method, path, route_target, status, user_id, callback_url, "
                "created_at, processed_at, error, retry_count, result_status_code"
            )
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return list(resp.data or [])

    with sqlite3.connect(QUEUE_DB) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            f"""
            SELECT id, method, path, route_target, status, user_id, callback_url,
                   created_at, processed_at, error, retry_count, result_status_code
            FROM {GATEWAY_JOBS_TABLE}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def _claim_job(job_id: str) -> bool:
    if jobs_backend() == "supabase":
        resp = (
            _get_supabase()
            .table(GATEWAY_JOBS_TABLE)
            .update({"status": "processing"})
            .eq("id", job_id)
            .eq("status", "queued")
            .execute()
        )
        return bool(resp.data)

    with sqlite3.connect(QUEUE_DB) as conn:
        cur = conn.execute(
            f"""
            UPDATE {GATEWAY_JOBS_TABLE}
            SET status = 'processing'
            WHERE id = ? AND status = 'queued'
            """,
            (job_id,),
        )
        conn.commit()
        return cur.rowcount > 0


def _is_text_content_type(content_type: str | None) -> bool:
    if not content_type:
        return True
    base = content_type.lower().split(";")[0].strip()
    return base.startswith("text/") or base in {
        "application/json",
        "application/problem+json",
    }


def _store_result(
    job_id: str,
    status_code: int,
    content_type: str | None,
    result_bytes: bytes,
) -> None:
    processed_at = _now_iso()
    result_storage: str | None = None
    result_key: str | None = None
    result_text: str | None = None

    if _is_text_content_type(content_type) and len(result_bytes) <= JOB_RESULT_LOCAL_THRESHOLD:
        result_text = result_bytes.decode("utf-8", errors="replace")
    else:
        result_storage, result_key = _put_bytes(
            f"results/{job_id}/response.bin",
            result_bytes,
            content_type or "application/octet-stream",
        )

    payload = {
        "status": "completed",
        "processed_at": processed_at,
        "error": None,
        "result_status_code": status_code,
        "result_content_type": content_type,
        "result_storage": result_storage,
        "result_key": result_key,
        "result_text": result_text,
    }

    if jobs_backend() == "supabase":
        _get_supabase().table(GATEWAY_JOBS_TABLE).update(payload).eq("id", job_id).execute()
        return

    with sqlite3.connect(QUEUE_DB) as conn:
        conn.execute(
            f"""
            UPDATE {GATEWAY_JOBS_TABLE}
            SET status = 'completed',
                processed_at = ?,
                error = NULL,
                result_status_code = ?,
                result_content_type = ?,
                result_storage = ?,
                result_key = ?,
                result_text = ?
            WHERE id = ?
            """,
            (
                processed_at,
                status_code,
                content_type,
                result_storage,
                result_key,
                result_text,
                job_id,
            ),
        )
        conn.commit()


def _mark_failed(job_id: str, error: str, *, requeue: bool = False) -> None:
    if jobs_backend() == "supabase":
        if requeue:
            current = get_job(job_id)
            retry_count = int(_job_get(current, "retry_count") or 0) + 1 if current else 1
            _get_supabase().table(GATEWAY_JOBS_TABLE).update(
                {"status": "queued", "error": error, "retry_count": retry_count}
            ).eq("id", job_id).execute()
        else:
            _get_supabase().table(GATEWAY_JOBS_TABLE).update(
                {
                    "status": "failed",
                    "processed_at": _now_iso(),
                    "error": error,
                }
            ).eq("id", job_id).execute()
        return

    with sqlite3.connect(QUEUE_DB) as conn:
        if requeue:
            conn.execute(
                f"""
                UPDATE {GATEWAY_JOBS_TABLE}
                SET status = 'queued', error = ?, retry_count = retry_count + 1
                WHERE id = ?
                """,
                (error, job_id),
            )
        else:
            conn.execute(
                f"""
                UPDATE {GATEWAY_JOBS_TABLE}
                SET status = 'failed', processed_at = ?, error = ?
                WHERE id = ?
                """,
                (_now_iso(), error, job_id),
            )
        conn.commit()


def _result_bytes(row: JobRow) -> bytes | None:
    if _job_get(row, "status") != "completed":
        return None
    if _job_get(row, "result_text"):
        return str(_job_get(row, "result_text")).encode("utf-8")
    rs, rk = _job_get(row, "result_storage"), _job_get(row, "result_key")
    if rs and rk:
        return _get_bytes(rs, rk)
    return None


def job_public_dict(row: JobRow, *, include_result: bool = True) -> dict[str, Any]:
    job_id = _job_get(row, "id")
    payload: dict[str, Any] = {
        "job_id": job_id,
        "method": _job_get(row, "method"),
        "path": _job_get(row, "path"),
        "route_target": _job_get(row, "route_target"),
        "status": _job_get(row, "status"),
        "user_id": _job_get(row, "user_id"),
        "callback_url": _job_get(row, "callback_url"),
        "created_at": _job_get(row, "created_at"),
        "processed_at": _job_get(row, "processed_at"),
        "error": _job_get(row, "error"),
        "retry_count": _job_get(row, "retry_count"),
        "result_status_code": _job_get(row, "result_status_code"),
        "result_content_type": _job_get(row, "result_content_type"),
        "poll_url": f"/jobs/{job_id}",
    }
    if include_result and _job_get(row, "status") == "completed":
        raw = _result_bytes(row)
        if raw is not None:
            ct = (_job_get(row, "result_content_type") or "").lower()
            if _is_text_content_type(ct):
                text = raw.decode("utf-8", errors="replace")
                payload["result"] = text
                if "application/json" in ct:
                    try:
                        payload["result_json"] = json.loads(text)
                    except json.JSONDecodeError:
                        pass
            else:
                payload["result_base64"] = base64.b64encode(raw).decode("ascii")
    return payload


async def _deliver_callback(row: JobRow, result_bytes: bytes) -> None:
    callback_url = _job_get(row, "callback_url")
    if not callback_url:
        return

    public = job_public_dict(row, include_result=True)
    payload = {
        "job_id": public["job_id"],
        "status": "completed",
        "method": public["method"],
        "path": public["path"],
        "route_target": public["route_target"],
        "result_status_code": public["result_status_code"],
        "result_content_type": public["result_content_type"],
        "result": public.get("result"),
        "result_json": public.get("result_json"),
        "processed_at": _now_iso(),
    }
    try:
        async with httpx.AsyncClient(timeout=JOB_CALLBACK_TIMEOUT_S) as client:
            resp = await client.post(callback_url, json=payload)
        if resp.status_code >= 400:
            logger.warning(
                "Callback for job %s returned %s: %s",
                public["job_id"],
                resp.status_code,
                resp.text[:500],
            )
        else:
            logger.info("Delivered callback for job %s → %s", public["job_id"], callback_url)
    except Exception as exc:
        logger.warning("Callback for job %s failed: %s", public["job_id"], exc)


async def _process_job(
    row: JobRow,
    upstream_base_url: Callable[[str], str],
) -> None:
    job_id = _job_get(row, "id")
    if not _claim_job(job_id):
        return

    retry_count = int(_job_get(row, "retry_count") or 0)
    path = _job_get(row, "path").lstrip("/")

    try:
        body = _get_bytes(_job_get(row, "body_storage"), _job_get(row, "body_key"))
        headers = json.loads(_job_get(row, "headers_json") or "{}")
        headers = _filter_replay_headers(headers)

        base = upstream_base_url(path)
        url = f"{base}/{path}"
        query_string = _job_get(row, "query_string") or ""
        if query_string:
            url = f"{url}?{query_string}"

        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.request(
                _job_get(row, "method"),
                url,
                headers=headers,
                content=body,
            )
            result_bytes = await resp.aread()

        if resp.status_code >= 500:
            error = f"upstream {resp.status_code}: {result_bytes[:500].decode('utf-8', errors='replace')}"
            if retry_count + 1 < JOB_MAX_RETRIES:
                _mark_failed(job_id, error, requeue=True)
                logger.warning("Job %s re-queued after upstream error", job_id)
            else:
                _mark_failed(job_id, error, requeue=False)
            return

        if resp.status_code >= 400:
            _mark_failed(
                job_id,
                f"upstream {resp.status_code}: {result_bytes[:500].decode('utf-8', errors='replace')}",
                requeue=False,
            )
            return

        content_type = resp.headers.get("content-type")
        _store_result(job_id, resp.status_code, content_type, result_bytes)
        logger.info(
            "Completed gateway job %s %s %s → %s (%d bytes)",
            job_id,
            _job_get(row, "method"),
            _job_get(row, "path"),
            resp.status_code,
            len(result_bytes),
        )

        updated = get_job(job_id)
        if updated:
            await _deliver_callback(updated, result_bytes)
    except Exception as exc:
        error = str(exc)
        if retry_count + 1 < JOB_MAX_RETRIES:
            _mark_failed(job_id, error, requeue=True)
            logger.warning("Job %s re-queued after error: %s", job_id, exc)
        else:
            _mark_failed(job_id, error, requeue=False)
            logger.exception("Failed gateway job %s", job_id)


def _fetch_queued(limit: int = 20) -> list[JobRow]:
    if jobs_backend() == "supabase":
        resp = (
            _get_supabase()
            .table(GATEWAY_JOBS_TABLE)
            .select("*")
            .eq("status", "queued")
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return list(resp.data or [])

    with sqlite3.connect(QUEUE_DB) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            f"""
            SELECT * FROM {GATEWAY_JOBS_TABLE}
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


async def drain_jobs_once(
    upstream_snapshot: dict[str, Any],
    service_ready: Callable[[str, dict[str, Any]], bool],
    upstream_base_url: Callable[[str], str],
) -> None:
    rows = _fetch_queued()
    if not rows:
        return

    replayable = [
        row
        for row in rows
        if service_ready(_job_get(row, "path").lstrip("/"), upstream_snapshot)
    ]
    if not replayable:
        return

    logger.info("Processing %d queued gateway job(s)...", len(replayable))
    semaphore = asyncio.Semaphore(JOB_DRAIN_CONCURRENCY)

    async def _run(row: JobRow) -> None:
        async with semaphore:
            await _process_job(row, upstream_base_url)

    await asyncio.gather(*[_run(row) for row in replayable])
