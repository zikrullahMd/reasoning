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
import re
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

# BM25 retrieval — optional dependency; falls back to positional selection if absent
try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25Okapi = None   # type: ignore[assignment,misc]
    _BM25_AVAILABLE = False

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

# --- OCR quality settings ---
OCR_RENDER_DPI           = 200   # base render resolution (higher → better OCR, more RAM)
OCR_CONFIDENCE_THRESHOLD = 0.70  # avg line-confidence below this → attempt second pass
OCR_SECOND_PASS_DPI      = 300   # re-render resolution for low-confidence pages
OCR_MIN_LINES_FOR_QUALITY = 3    # minimum lines needed for a meaningful confidence signal

# --- Context / token budget ---
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))
CHARS_PER_TOKEN    = 4           # rough approximation (1 token ≈ 4 chars)
MAX_CONTEXT_CHARS  = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN  # 24 000 chars
MAX_CHUNK_CHARS    = 1500        # max chars per individual chunk (~375 tokens)

# --- BM25 retrieval ---
# Retrieve this many top-scored chunks; the budget cap then selects how many fit.
# Setting it higher than needed is fine — budget is the hard limit.
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "30"))

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


def _score_ocr_result(page_result) -> tuple[float, int]:
    """
    Compute average confidence and line count from a Surya OCR page result.

    Surya's TextLine objects carry a `confidence` float in [0, 1].
    We use getattr with a safe default so the function stays compatible with
    future Surya versions that might rename the field.

    Returns (avg_confidence, line_count).
    avg_confidence is 0.0 when there are no lines (blank / failed page).
    """
    lines = getattr(page_result, "text_lines", [])
    if not lines:
        return 0.0, 0
    confidences = [getattr(line, "confidence", 1.0) for line in lines]
    return sum(confidences) / len(confidences), len(lines)


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
        pix = page.get_pixmap(dpi=OCR_RENDER_DPI)
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
            pix = page.get_pixmap(dpi=OCR_RENDER_DPI)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            scanned_indices.append(i)
            scanned_images.append(img)
            page_modes.append("surya_ocr")
            logger.debug(
                "Page %d/%d → OCR (votes=%d/4)", i + 1, page_count, signals["digital_votes"]
            )

    doc.close()

    # Track which doc-page indices needed a second pass (used for classification summary)
    low_conf_page_indices: list[int] = []

    # All scanned pages go through a single batched model call
    if scanned_images:
        from surya.ocr import run_ocr
        model, processor = get_surya_models()
        logger.info(
            "OCR pass 1 at %d DPI on %d/%d scanned page(s)",
            OCR_RENDER_DPI, len(scanned_images), page_count,
        )
        first_pass_results = run_ocr(scanned_images, model, processor)

        # Assess quality; store first-pass text and confidence flags
        second_pass_doc_indices: list[int]   = []  # doc-page indices needing retry
        second_pass_images:      list[Image.Image] = []

        for local_i, (doc_idx, result) in enumerate(zip(scanned_indices, first_pass_results)):
            avg_conf, line_count = _score_ocr_result(result)
            needs_second_pass = (
                line_count >= OCR_MIN_LINES_FOR_QUALITY
                and avg_conf < OCR_CONFIDENCE_THRESHOLD
            )
            page_texts[doc_idx] = "\n".join(
                line.text for line in getattr(result, "text_lines", [])
            )
            page_signals[doc_idx].update({
                "ocr_avg_confidence":    round(avg_conf, 4),
                "ocr_line_count":        line_count,
                "ocr_low_confidence":    needs_second_pass,
                "ocr_second_pass_used":  False,
            })
            if needs_second_pass:
                second_pass_doc_indices.append(doc_idx)

        # Second pass: re-render only the low-confidence pages at higher DPI
        if second_pass_doc_indices:
            logger.info(
                "OCR pass 2 at %d DPI for %d low-confidence page(s): %s",
                OCR_SECOND_PASS_DPI,
                len(second_pass_doc_indices),
                [p + 1 for p in second_pass_doc_indices],
            )
            doc2 = fitz.open(stream=pdf_bytes, filetype="pdf")
            for doc_idx in second_pass_doc_indices:
                pix = doc2[doc_idx].get_pixmap(dpi=OCR_SECOND_PASS_DPI)
                second_pass_images.append(
                    Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                )
            doc2.close()

            second_pass_results = run_ocr(second_pass_images, model, processor)

            for doc_idx, result in zip(second_pass_doc_indices, second_pass_results):
                avg_conf2, line_count2 = _score_ocr_result(result)
                prev_conf = page_signals[doc_idx]["ocr_avg_confidence"]
                if avg_conf2 > prev_conf:
                    page_texts[doc_idx] = "\n".join(
                        line.text for line in getattr(result, "text_lines", [])
                    )
                    page_signals[doc_idx].update({
                        "ocr_avg_confidence":   round(avg_conf2, 4),
                        "ocr_line_count":       line_count2,
                        "ocr_second_pass_used": True,
                    })
                    logger.debug(
                        "Page %d: pass 2 improved confidence %.3f → %.3f",
                        doc_idx + 1, prev_conf, avg_conf2,
                    )
                else:
                    logger.debug(
                        "Page %d: pass 2 no improvement (%.3f vs %.3f), keeping pass 1",
                        doc_idx + 1, avg_conf2, prev_conf,
                    )

            low_conf_page_indices = second_pass_doc_indices

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

    # Build OCR quality summary (only meaningful when scanned pages exist)
    ocr_quality: dict = {}
    if n_scanned > 0:
        ocr_sigs = [ps for ps in page_signals if "ocr_avg_confidence" in ps]
        overall_conf = (
            round(sum(ps["ocr_avg_confidence"] for ps in ocr_sigs) / len(ocr_sigs), 4)
            if ocr_sigs else None
        )
        ocr_quality = {
            "render_dpi":              OCR_RENDER_DPI,
            "confidence_threshold":    OCR_CONFIDENCE_THRESHOLD,
            "overall_avg_confidence":  overall_conf,
            "low_confidence_pages":    [p + 1 for p in low_conf_page_indices],
            "second_pass_page_count":  sum(
                1 for ps in page_signals if ps.get("ocr_second_pass_used")
            ),
        }

    classification = {
        "is_mixed":           0 < n_scanned < page_count,
        "digital_page_count": n_digital,
        "scanned_page_count": n_scanned,
        "page_modes":         page_modes,
        "page_signals":       page_signals,
        "ocr_quality":        ocr_quality,
        # page_texts is popped in the endpoint before metrics logging to avoid
        # storing large text blobs in the JSONL record
        "_page_texts":        page_texts,
    }

    return "\n\n".join(page_texts), method, page_count, round(time.perf_counter() - t0, 4), classification


# ---------------------------------------------------------------------------
# Document chunking + context budget
# ---------------------------------------------------------------------------

def _split_page_into_chunks(page_text: str, page_num: int, max_chars: int) -> list[dict]:
    """
    Split one page's text into sub-chunks of at most max_chars.
    Breaks prefer double-newlines (paragraphs) then single newlines.
    Each chunk carries page number and part index for labelling in the prompt.
    """
    text = page_text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [{"page": page_num, "part": 1, "text": text, "char_count": len(text)}]

    parts: list[dict] = []
    remaining = text
    part = 1
    while remaining:
        if len(remaining) <= max_chars:
            parts.append({"page": page_num, "part": part, "text": remaining, "char_count": len(remaining)})
            break
        cut = max_chars
        # Try paragraph break first, then line break
        b = remaining.rfind("\n\n", 0, cut)
        if b > max_chars // 3:
            cut = b
        else:
            b = remaining.rfind("\n", 0, cut)
            if b > max_chars // 3:
                cut = b
        chunk_text = remaining[:cut].rstrip()
        if chunk_text:
            parts.append({"page": page_num, "part": part, "text": chunk_text, "char_count": len(chunk_text)})
        remaining = remaining[cut:].lstrip()
        part += 1
    return parts


def chunk_document(page_texts: list[str]) -> list[dict]:
    """
    Convert a list of per-page texts into a flat list of chunks.
    Long pages are split into multiple parts via _split_page_into_chunks.
    Each chunk: {"page": int, "part": int, "text": str, "char_count": int}.
    """
    chunks: list[dict] = []
    for i, page_text in enumerate(page_texts):
        chunks.extend(_split_page_into_chunks(page_text, page_num=i + 1, max_chars=MAX_CHUNK_CHARS))
    return chunks


def select_chunks_within_budget(
    chunks: list[dict],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[list[dict], dict]:
    """
    Greedily select chunks from the start until the character budget is exhausted.
    Returns (selected_chunks, budget_info).

    This is a positional fallback.  Item 5 replaces the selection logic with
    relevance-ranked retrieval while keeping this function's signature intact.
    """
    selected: list[dict] = []
    total_chars = 0
    for chunk in chunks:
        if total_chars + chunk["char_count"] > max_chars:
            break
        selected.append(chunk)
        total_chars += chunk["char_count"]

    total_pages = chunks[-1]["page"] if chunks else 0
    last_selected_page = selected[-1]["page"] if selected else 0
    was_truncated = len(selected) < len(chunks)

    return selected, {
        "total_chunks":              len(chunks),
        "selected_chunks":           len(selected),
        "was_truncated":             was_truncated,
        "context_chars":             total_chars,
        "estimated_context_tokens":  total_chars // CHARS_PER_TOKEN,
        "max_context_tokens":        MAX_CONTEXT_TOKENS,
        "pages_in_context":          f"1–{last_selected_page}" if was_truncated else f"1–{total_pages}",
        "pages_omitted":             total_pages - last_selected_page if was_truncated else 0,
    }


def _tokenize_for_bm25(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, split on whitespace.
    Used for both the corpus (chunks) and the query (question) so they share
    the same vocabulary and match on stems like 'total' == 'total:'.
    """
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def retrieve_relevant_chunks(
    question: str,
    chunks: list[dict],
    top_k: int = BM25_TOP_K,
) -> tuple[list[dict], dict]:
    """
    Rank all chunks by BM25 relevance to the question, return the top_k.

    BM25 (Okapi BM25) is a bag-of-words ranking function that weighs term
    frequency against inverse document frequency.  It finds chunks that share
    keywords with the question, naturally surfacing the pages most likely to
    contain the answer — regardless of where they sit in the document.

    Falls back to positional order if rank_bm25 is not installed.

    Returns (ranked_chunks, retrieval_info).
    Chunks are in RELEVANCE order (highest score first).
    The caller must re-sort by page number before formatting the prompt.
    """
    if not chunks:
        return [], {"method": "none", "reason": "no chunks", "retrieved": 0}

    if not _BM25_AVAILABLE:
        logger.warning(
            "rank_bm25 not installed — falling back to positional chunk selection. "
            "Install with: pip install rank-bm25"
        )
        fallback = chunks[:top_k]
        return fallback, {
            "method":       "positional_fallback",
            "reason":       "rank_bm25 not installed",
            "top_k":        top_k,
            "total_chunks": len(chunks),
            "retrieved":    len(fallback),
        }

    tokenized_corpus = [_tokenize_for_bm25(c["text"]) for c in chunks]
    bm25 = _BM25Okapi(tokenized_corpus)

    query_tokens = _tokenize_for_bm25(question)
    scores = bm25.get_scores(query_tokens)

    # Sort by score descending, take top_k
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    ranked = [{"bm25_score": round(float(scores[i]), 4), **chunks[i]} for i in ranked_indices]

    return ranked, {
        "method":      "bm25",
        "top_k":       top_k,
        "total_chunks": len(chunks),
        "retrieved":   len(ranked),
        "top_score":   round(float(scores[ranked_indices[0]]), 4) if ranked_indices else 0.0,
        "min_score":   round(float(scores[ranked_indices[-1]]), 4) if ranked_indices else 0.0,
    }


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Render selected chunks as a labelled string ready for the prompt.
    Single-part pages get a plain [Page N] header; sub-chunked pages get [Page N, Part M].
    """
    parts: list[str] = []
    for chunk in chunks:
        page, part = chunk["page"], chunk.get("part", 1)
        label = f"[Page {page}]" if part == 1 else f"[Page {page}, Part {part}]"
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n".join(parts)


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

    # --- Chunking → BM25 retrieval → context budget ---
    # Pop page_texts before metrics logging (avoid storing large blobs in JSONL)
    page_texts_list: list[str] = pdf_classification.pop("_page_texts", [document_text])

    chunks = chunk_document(page_texts_list)

    # Rank all chunks by relevance to the question
    ranked_chunks, retrieval_info = retrieve_relevant_chunks(question, chunks)

    # Apply token budget — now fills with the most relevant chunks first
    selected_chunks, budget_info = select_chunks_within_budget(ranked_chunks)

    # Re-sort selected chunks into document (page) order so the context reads
    # coherently: the model sees page 3 before page 24, not relevance order
    selected_chunks_ordered = sorted(
        selected_chunks, key=lambda c: (c["page"], c.get("part", 1))
    )

    if budget_info["was_truncated"]:
        logger.info(
            "[req=%s] BM25 retrieved %d/%d chunks; %d fit budget (%s of %d pages). "
            "Low-relevance chunks omitted.",
            request_id,
            retrieval_info["retrieved"],
            retrieval_info["total_chunks"],
            budget_info["selected_chunks"],
            budget_info["pages_in_context"],
            page_count,
        )
    else:
        logger.info(
            "[req=%s] BM25 retrieved %d relevant chunks — all fit budget.",
            request_id,
            retrieval_info["retrieved"],
        )

    context_text = format_chunks_for_prompt(selected_chunks_ordered)
    messages = build_prompt(context_text, question)

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
                "retrieval": retrieval_info,
                "context_budget": budget_info,
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
