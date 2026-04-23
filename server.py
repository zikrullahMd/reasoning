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

# --- PDF classification thresholds (multi-signal) ---
MIN_TEXT_DENSITY = 50           # avg chars/page: below → no meaningful text
MIN_DIGITAL_FONT_RATIO = 0.5   # fraction of pages that must carry embedded fonts
MAX_SCANNED_IMAGE_RATIO = 0.6  # avg image-area/page-area: above → full-page scan image
MIN_TEXT_BLOCK_DENSITY = 2.0   # avg text blocks/page: below → no structural text
CLASSIFICATION_VOTES_NEEDED = 3 # signals (out of 4) required to call a PDF "digital"

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

def _classify_pdf_nature(doc: fitz.Document) -> dict:
    """
    Multi-signal heuristic that decides whether a PDF is digital or scanned.

    Four independent signals are each voted as digital (1) or scanned (0).
    A final majority vote (CLASSIFICATION_VOTES_NEEDED of 4) gives the verdict.

    Signals
    -------
    1. char_density   – avg extracted chars/page  (low → no selectable text)
    2. font_ratio     – fraction of pages with embedded fonts (absent → raster scan)
    3. image_ratio    – avg image area / page area (high → full-page scan background)
    4. text_blocks    – avg PyMuPDF text-block count/page (near-zero → no structure)

    Returns a dict with 'is_digital', 'digital_votes', 'confidence', and 'signals'.
    """
    page_count = len(doc)
    if page_count == 0:
        return {"is_digital": False, "digital_votes": 0, "confidence": "low", "signals": {}}

    total_chars = 0
    pages_with_fonts = 0
    total_image_ratio = 0.0
    total_text_blocks = 0

    for page in doc:
        # Signal 1 — character count from selectable text layer
        text = page.get_text()
        total_chars += len(text.strip())

        # Signal 2 — embedded font presence
        if page.get_fonts():
            pages_with_fonts += 1

        # Signal 3 — image area coverage
        page_area = page.rect.width * page.rect.height
        if page_area > 0:
            img_area = 0.0
            for info in page.get_image_info():
                bbox = info.get("bbox", (0, 0, 0, 0))
                img_area += abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            total_image_ratio += min(img_area / page_area, 1.0)

        # Signal 4 — text block count (block_type 0 = text, 1 = image)
        blocks = page.get_text("blocks")
        total_text_blocks += sum(1 for b in blocks if len(b) > 6 and b[6] == 0)

    avg_chars = total_chars / page_count
    font_page_ratio = pages_with_fonts / page_count
    avg_image_ratio = total_image_ratio / page_count
    avg_text_blocks = total_text_blocks / page_count

    sig_chars = avg_chars >= MIN_TEXT_DENSITY
    sig_fonts = font_page_ratio >= MIN_DIGITAL_FONT_RATIO
    sig_images = avg_image_ratio < MAX_SCANNED_IMAGE_RATIO
    sig_blocks = avg_text_blocks >= MIN_TEXT_BLOCK_DENSITY

    digital_votes = sum([sig_chars, sig_fonts, sig_images, sig_blocks])
    is_digital = digital_votes >= CLASSIFICATION_VOTES_NEEDED

    # 4/4 or 0/4 → high confidence; 3/4 or 1/4 → medium; 2/4 → ambiguous
    if digital_votes in (0, 4):
        confidence = "high"
    elif digital_votes in (1, 3):
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "is_digital": is_digital,
        "digital_votes": digital_votes,
        "confidence": confidence,
        "signals": {
            "avg_chars_per_page": round(avg_chars, 1),
            "char_density_ok": sig_chars,
            "font_page_ratio": round(font_page_ratio, 3),
            "fonts_ok": sig_fonts,
            "avg_image_area_ratio": round(avg_image_ratio, 3),
            "image_ratio_ok": sig_images,
            "avg_text_blocks_per_page": round(avg_text_blocks, 1),
            "text_blocks_ok": sig_blocks,
        },
    }


def _extract_pages_pymupdf(doc: fitz.Document) -> str:
    """Return concatenated page text from an already-open PyMuPDF document."""
    return "\n\n".join(page.get_text() for page in doc)


def _classify_page(page: fitz.Page) -> tuple[bool, str, dict]:
    """
    Classify a single PDF page as digital or scanned using the same 4-signal
    vote as the document-level classifier.

    Uses page.get_text("blocks") in one call to derive both the plain text
    content and the text-block count, avoiding a second page read.

    Returns
    -------
    is_digital : bool
    text       : str   – extracted text (only meaningful when is_digital=True)
    signals    : dict  – per-signal values and vote breakdown
    """
    raw_blocks = page.get_text("blocks")
    text_blocks = [b for b in raw_blocks if len(b) > 6 and b[6] == 0]
    plain_text = "\n".join(b[4] for b in text_blocks)
    char_count = len(plain_text.strip())
    text_block_count = len(text_blocks)

    has_fonts = bool(page.get_fonts())

    page_area = page.rect.width * page.rect.height
    img_ratio = 0.0
    if page_area > 0:
        img_area = 0.0
        for info in page.get_image_info():
            bbox = info.get("bbox", (0, 0, 0, 0))
            img_area += abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        img_ratio = min(img_area / page_area, 1.0)

    sig_chars  = char_count       >= MIN_TEXT_DENSITY
    sig_fonts  = has_fonts
    sig_images = img_ratio        <  MAX_SCANNED_IMAGE_RATIO
    sig_blocks = text_block_count >= MIN_TEXT_BLOCK_DENSITY

    votes = sum([sig_chars, sig_fonts, sig_images, sig_blocks])
    is_digital = votes >= CLASSIFICATION_VOTES_NEEDED

    return is_digital, plain_text, {
        "char_count":       char_count,
        "char_density_ok":  sig_chars,
        "has_fonts":        has_fonts,
        "fonts_ok":         sig_fonts,
        "image_area_ratio": round(img_ratio, 3),
        "image_ratio_ok":   sig_images,
        "text_block_count": text_block_count,
        "text_blocks_ok":   sig_blocks,
        "digital_votes":    votes,
    }


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


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, str, int, float, dict]:
    """
    Extract text from a PDF using a per-page digital/scanned decision.

    Each page is independently classified with the 4-signal vote.  Digital
    pages are read via PyMuPDF; scanned pages are rendered to images and
    collected for a single batched Surya OCR call.  Mixed documents (some
    digital, some scanned) are handled correctly without skipping any page.

    Returns
    -------
    text              : str   – full document text, pages joined by double newline
    extraction_method : str   – 'pymupdf' | 'surya_ocr' | 'mixed'
    page_count        : int
    extraction_time_s : float
    classification    : dict  – per-page modes, signals, and document-level summary
    """
    t0 = time.perf_counter()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)

    page_texts: list[str]             = [""] * page_count
    page_modes: list[str]             = []
    page_signals: list[dict]          = []
    scanned_indices: list[int]        = []
    scanned_images: list[Image.Image] = []

    for i, page in enumerate(doc):
        is_digital, text, signals = _classify_page(page)
        page_signals.append({"page": i + 1, **signals})

        if is_digital:
            page_texts[i] = text
            page_modes.append("pymupdf")
        else:
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            scanned_indices.append(i)
            scanned_images.append(img)
            page_modes.append("surya_ocr")
            logger.debug(
                "Page %d/%d → OCR (votes=%d/4)", i + 1, page_count, signals["digital_votes"]
            )

    doc.close()

    # All scanned pages go through a single model call for efficiency
    if scanned_images:
        from surya.ocr import run_ocr
        model, processor = get_surya_models()
        logger.info(
            "Running Surya OCR on %d/%d scanned page(s)", len(scanned_images), page_count
        )
        ocr_results = run_ocr(scanned_images, model, processor)
        for idx, result in zip(scanned_indices, ocr_results):
            page_texts[idx] = "\n".join(line.text for line in result.text_lines)

    n_scanned = len(scanned_indices)
    n_digital = page_count - n_scanned

    if n_scanned == 0:
        method = "pymupdf"
    elif n_digital == 0:
        method = "surya_ocr"
    else:
        method = "mixed"

    logger.info(
        "Extraction complete: method=%s digital_pages=%d scanned_pages=%d",
        method, n_digital, n_scanned,
    )

    classification = {
        "is_mixed":           0 < n_scanned < page_count,
        "digital_page_count": n_digital,
        "scanned_page_count": n_scanned,
        "page_modes":         page_modes,
        "page_signals":       page_signals,
    }

    return "\n\n".join(page_texts), method, page_count, round(time.perf_counter() - t0, 4), classification


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
        document_text, extraction_method, page_count, extraction_time_s, pdf_classification = (
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
                "classification": pdf_classification,
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
