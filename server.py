"""
PDF Inference Pipeline Server (2026 Stack)

FastAPI server that:
1. Accepts PDF uploads + questions
2. Extracts text via PyMuPDF (digital) or Surya OCR (scanned)
3. Streams LLM responses from SGLang inference server
4. Logs structured performance metrics to metrics.jsonl
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SGLANG_URL = os.getenv("SGLANG_URL", "http://localhost:30000")
MIN_TEXT_DENSITY = 50  # minimum chars/page to consider PDF as "has text"
METRICS_LOG_PATH = Path(os.getenv("METRICS_LOG", "metrics.jsonl"))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")
perf_logger = logging.getLogger("perf")

# ---------------------------------------------------------------------------
# Surya OCR lazy state + active model cache
# ---------------------------------------------------------------------------

_surya_model = None
_surya_processor = None
_active_model: str = "unknown"


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _append_metric(record: dict) -> None:
    """Append a JSON record as a single line to the JSONL metrics log."""
    try:
        with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Failed to write metrics log: %s", exc)


# ---------------------------------------------------------------------------
# Lifespan — probe SGLang for the active model at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _active_model
    logger.info("Starting up — probing SGLang for active model...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{SGLANG_URL}/v1/models")
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    _active_model = models[0].get("id", "unknown")
                    logger.info("Active model: %s", _active_model)
                else:
                    logger.warning("SGLang returned no models")
    except Exception as exc:
        logger.warning("Could not fetch model at startup: %s", exc)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PDF Inference Pipeline",
    description="Upload PDFs and ask questions - powered by SGLang",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware — HTTP-level timing + request identity headers
# ---------------------------------------------------------------------------

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        request.state.request_id = str(uuid.uuid4())[:8]
        response = await call_next(request)
        elapsed = round(time.perf_counter() - t0, 4)
        response.headers["X-Process-Time"] = str(elapsed)
        response.headers["X-Request-Id"] = request.state.request_id
        logger.info(
            "%s %s → %.3fs [req_id=%s]",
            request.method,
            request.url.path,
            elapsed,
            request.state.request_id,
        )
        return response


app.add_middleware(TimingMiddleware)


# ---------------------------------------------------------------------------
# Surya OCR
# ---------------------------------------------------------------------------

def get_surya_models():
    """Lazy-load Surya OCR models (keeps GPU free until needed)."""
    global _surya_model, _surya_processor
    if _surya_model is None:
        from surya.ocr import load_model, load_processor
        _surya_model = load_model()
        _surya_processor = load_processor()
    return _surya_model, _surya_processor


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_pymupdf(pdf_bytes: bytes) -> tuple[str, bool, int]:
    """
    Extract text from PDF using PyMuPDF.
    Returns (text, has_sufficient_text, page_count).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    total_chars = 0

    for page in doc:
        text = page.get_text()
        pages_text.append(text)
        total_chars += len(text.strip())

    page_count = len(pages_text)
    doc.close()

    full_text = "\n\n".join(pages_text)
    avg_chars_per_page = total_chars / max(page_count, 1)
    has_sufficient_text = avg_chars_per_page >= MIN_TEXT_DENSITY

    return full_text, has_sufficient_text, page_count


def extract_text_surya(pdf_bytes: bytes) -> tuple[str, int]:
    """
    Extract text from scanned PDF using Surya OCR.
    Returns (text, page_count).
    """
    from surya.ocr import run_ocr

    model, processor = get_surya_models()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    page_count = len(images)
    doc.close()

    results = run_ocr(images, model, processor)

    pages_text = []
    for page_result in results:
        page_lines = [line.text for line in page_result.text_lines]
        pages_text.append("\n".join(page_lines))

    return "\n\n".join(pages_text), page_count


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, str, int, float]:
    """
    Extract text from PDF, preferring PyMuPDF and falling back to Surya OCR.
    Returns (text, extraction_method, page_count, extraction_time_s).
    """
    t0 = time.perf_counter()
    text, has_text, page_count = extract_text_pymupdf(pdf_bytes)

    if has_text:
        return text, "pymupdf", page_count, round(time.perf_counter() - t0, 4)

    text, page_count = extract_text_surya(pdf_bytes)
    return text, "surya_ocr", page_count, round(time.perf_counter() - t0, 4)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(document_text: str, user_question: str) -> list[dict]:
    """Build OpenAI-compatible messages array for the LLM."""
    system_prompt = (
        "You are a helpful, analytical assistant. "
        "Read the following document carefully and answer the user's question "
        "based strictly on the information provided in the document. "
        "If the answer cannot be found in the document, say so clearly."
    )

    user_content = f"""## Document Content

{document_text}

---

## Question

{user_question}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# LLM streaming
# ---------------------------------------------------------------------------

async def query_llm(
    messages: list[dict],
    timing: dict,
) -> AsyncGenerator[str, None]:
    """
    Stream completion from SGLang (OpenAI-compatible API).
    Populates `timing` dict in-place with LLM performance metrics.
    """
    t_start = time.perf_counter()
    first_token_s: float | None = None
    output_chars = 0

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{SGLANG_URL}/v1/chat/completions",
            json={
                "messages": messages,
                "stream": True,
                "max_tokens": 2048,
                "temperature": 0.7,
            },
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"SGLang error: {error_text.decode()}",
                )

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_s is None:
                                first_token_s = round(time.perf_counter() - t_start, 4)
                            output_chars += len(content)
                            yield content
                    except json.JSONDecodeError:
                        continue

    total_stream_s = round(time.perf_counter() - t_start, 4)
    est_output_tokens = output_chars // 4
    tps = round(est_output_tokens / total_stream_s, 2) if total_stream_s > 0 else 0.0

    timing.update(
        {
            "time_to_first_token_s": first_token_s or 0.0,
            "total_stream_time_s": total_stream_s,
            "output_chars": output_chars,
            "estimated_output_tokens": est_output_tokens,
            "tokens_per_second": tps,
        }
    )


# ---------------------------------------------------------------------------
# Analyze endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to analyze"),
    question: str = Form(..., description="Question to ask about the document"),
):
    """
    Upload a PDF and ask a question about its contents.
    Returns a streaming response with the LLM's answer.
    All timing and quality metrics are logged to metrics.jsonl.
    """
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    t_request_start = time.perf_counter()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        document_text, extraction_method, page_count, extraction_time_s = (
            extract_text_from_pdf(pdf_bytes)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {e}")

    if not document_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from the PDF",
        )

    messages = build_prompt(document_text, question)

    # Estimate prompt size (rough: 1 token ≈ 4 chars)
    prompt_chars = sum(len(m["content"]) for m in messages)
    est_prompt_tokens = prompt_chars // 4

    llm_timing: dict = {}
    collected_output: list[str] = []

    async def timed_stream():
        async for chunk in query_llm(messages, llm_timing):
            collected_output.append(chunk)
            yield chunk

        # Stream is done — build and log the full metrics record
        full_output = "".join(collected_output)
        total_request_s = round(time.perf_counter() - t_request_start, 4)

        ttft = llm_timing.get("time_to_first_token_s", 0.0)
        est_prompt_tok = est_prompt_tokens or 1
        answer_latency_per_input_token_ms = round((ttft * 1000) / est_prompt_tok, 4)

        record = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pdf": {
                "filename": file.filename,
                "size_kb": round(len(pdf_bytes) / 1024, 2),
                "page_count": page_count,
                "extraction_method": extraction_method,
                "extraction_time_s": extraction_time_s,
                "text_chars": len(document_text),
            },
            "prompt": {
                "question": question,
                "question_words": len(question.split()),
                "total_prompt_chars": prompt_chars,
                "estimated_prompt_tokens": est_prompt_tokens,
            },
            "model": {
                "id": _active_model,
                "temperature": 0.7,
                "max_tokens": 2048,
                "sglang_url": SGLANG_URL,
            },
            "performance": {
                "extraction_time_s": extraction_time_s,
                "time_to_first_token_s": llm_timing.get("time_to_first_token_s", 0.0),
                "total_stream_time_s": llm_timing.get("total_stream_time_s", 0.0),
                "total_request_time_s": total_request_s,
                "estimated_output_tokens": llm_timing.get("estimated_output_tokens", 0),
                "tokens_per_second": llm_timing.get("tokens_per_second", 0.0),
                "output_chars": llm_timing.get("output_chars", 0),
            },
            "quality_signals": {
                "response_empty": len(full_output.strip()) == 0,
                "said_not_found": "cannot be found" in full_output.lower()
                or "not found in the document" in full_output.lower(),
                "answer_latency_per_input_token_ms": answer_latency_per_input_token_ms,
            },
        }

        _append_metric(record)
        perf_logger.info(
            "[req=%s] ttft=%.3fs tps=%.1f tokens=%d total=%.3fs model=%s method=%s",
            request_id,
            record["performance"]["time_to_first_token_s"],
            record["performance"]["tokens_per_second"],
            record["performance"]["estimated_output_tokens"],
            total_request_s,
            _active_model,
            extraction_method,
        )

    return StreamingResponse(timed_stream(), media_type="text/plain")


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------

@app.get("/stats")
async def get_stats():
    """
    Aggregate performance stats from the metrics.jsonl log.
    Returns per-model summaries, slowest/fastest prompts, and method breakdown.
    """
    if not METRICS_LOG_PATH.exists():
        return JSONResponse({"error": "No metrics recorded yet."}, status_code=404)

    records: list[dict] = []
    with METRICS_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return JSONResponse({"error": "Metrics file is empty."}, status_code=404)

    total = len(records)

    ttfts = [r["performance"]["time_to_first_token_s"] for r in records]
    tpss = [r["performance"]["tokens_per_second"] for r in records if r["performance"]["tokens_per_second"] > 0]
    total_times = [r["performance"]["total_request_time_s"] for r in records]

    method_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    for r in records:
        m = r["pdf"]["extraction_method"]
        method_counts[m] = method_counts.get(m, 0) + 1
        mid = r["model"]["id"]
        model_counts[mid] = model_counts.get(mid, 0) + 1

    said_not_found = sum(1 for r in records if r["quality_signals"]["said_not_found"])
    empty_responses = sum(1 for r in records if r["quality_signals"]["response_empty"])

    # Slowest and fastest by TTFT
    sorted_by_ttft = sorted(records, key=lambda r: r["performance"]["time_to_first_token_s"])
    fastest = sorted_by_ttft[0]
    slowest = sorted_by_ttft[-1]

    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "total_requests": total,
        "averages": {
            "time_to_first_token_s": _avg(ttfts),
            "tokens_per_second": _avg(tpss),
            "total_request_time_s": _avg(total_times),
        },
        "extraction_methods": method_counts,
        "models_used": model_counts,
        "quality": {
            "said_not_found_count": said_not_found,
            "empty_response_count": empty_responses,
            "said_not_found_pct": round(said_not_found / total * 100, 1),
        },
        "fastest_prompt": {
            "request_id": fastest["request_id"],
            "question": fastest["prompt"]["question"],
            "time_to_first_token_s": fastest["performance"]["time_to_first_token_s"],
            "tokens_per_second": fastest["performance"]["tokens_per_second"],
            "model": fastest["model"]["id"],
            "extraction_method": fastest["pdf"]["extraction_method"],
        },
        "slowest_prompt": {
            "request_id": slowest["request_id"],
            "question": slowest["prompt"]["question"],
            "time_to_first_token_s": slowest["performance"]["time_to_first_token_s"],
            "tokens_per_second": slowest["performance"]["tokens_per_second"],
            "model": slowest["model"]["id"],
            "extraction_method": slowest["pdf"]["extraction_method"],
        },
    }


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Check if the server and SGLang backend are healthy."""
    sglang_status = "unknown"
    sglang_models: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{SGLANG_URL}/v1/models")
            if response.status_code == 200:
                sglang_status = "healthy"
                data = response.json()
                sglang_models = [m.get("id") for m in data.get("data", [])]
                # Keep the cached model name up to date
                global _active_model
                if sglang_models:
                    _active_model = sglang_models[0]
            else:
                sglang_status = f"error: {response.status_code}"
    except httpx.ConnectError:
        sglang_status = "unreachable"
    except Exception as e:
        sglang_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "active_model": _active_model,
        "sglang": {
            "url": SGLANG_URL,
            "status": sglang_status,
            "models": sglang_models,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
