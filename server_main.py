"""
PDF Inference Pipeline Server (2026 Stack)

FastAPI server that:
1. Accepts PDF uploads + questions
2. Extracts text via PyMuPDF (digital) or Surya OCR (scanned)
3. Streams LLM responses from SGLang inference server
"""

import io
import os
from typing import AsyncGenerator

import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

# Surya OCR imports (lazy-loaded to avoid GPU initialization on import)
_surya_model = None
_surya_processor = None

SGLANG_URL = os.getenv("SGLANG_URL", "http://localhost:30000")
MIN_TEXT_DENSITY = 50  # Minimum characters per page to consider as "has text"

app = FastAPI(
    title="PDF Inference Pipeline",
    description="Upload PDFs and ask questions - powered by SGLang",
    version="1.0.0",
)


def get_surya_models():
    """Lazy-load Surya OCR models (keeps GPU free until needed)."""
    global _surya_model, _surya_processor
    if _surya_model is None:
        from surya.ocr import load_model, load_processor
        _surya_model = load_model()
        _surya_processor = load_processor()
    return _surya_model, _surya_processor


def extract_text_pymupdf(pdf_bytes: bytes) -> tuple[str, bool]:
    """
    Extract text from PDF using PyMuPDF.
    Returns (text, has_sufficient_text) tuple.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    total_chars = 0
    
    for page in doc:
        text = page.get_text()
        pages_text.append(text)
        total_chars += len(text.strip())
    
    doc.close()
    
    full_text = "\n\n".join(pages_text)
    avg_chars_per_page = total_chars / max(len(pages_text), 1)
    has_sufficient_text = avg_chars_per_page >= MIN_TEXT_DENSITY
    
    return full_text, has_sufficient_text


def extract_text_surya(pdf_bytes: bytes) -> str:
    """
    Extract text from scanned PDF using Surya OCR.
    Converts PDF pages to images and runs OCR.
    """
    from surya.ocr import run_ocr
    
    model, processor = get_surya_models()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    doc.close()
    
    results = run_ocr(images, model, processor)
    
    pages_text = []
    for page_result in results:
        page_lines = [line.text for line in page_result.text_lines]
        pages_text.append("\n".join(page_lines))
    
    return "\n\n".join(pages_text)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF, using PyMuPDF for digital PDFs
    and falling back to Surya OCR for scanned documents.
    """
    text, has_text = extract_text_pymupdf(pdf_bytes)
    
    if has_text:
        return text
    
    # Fallback to Surya OCR for scanned PDFs
    return extract_text_surya(pdf_bytes)


def build_prompt(document_text: str, user_question: str) -> list[dict]:
    """
    Build OpenAI-compatible messages array for the LLM.
    """
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


async def query_llm(messages: list[dict]) -> AsyncGenerator[str, None]:
    """
    Stream completion from SGLang server (OpenAI-compatible API).
    """
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
                    
                    import json
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


@app.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(..., description="PDF file to analyze"),
    question: str = Form(..., description="Question to ask about the document"),
):
    """
    Upload a PDF and ask a question about its contents.
    Returns a streaming response with the LLM's answer.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    pdf_bytes = await file.read()
    
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    try:
        document_text = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {e}")
    
    if not document_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from the PDF",
        )
    
    messages = build_prompt(document_text, question)
    
    return StreamingResponse(
        query_llm(messages),
        media_type="text/plain",
    )


@app.get("/health")
async def health_check():
    """
    Check if the server and SGLang backend are healthy.
    """
    sglang_status = "unknown"
    sglang_models = []
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{SGLANG_URL}/v1/models")
            if response.status_code == 200:
                sglang_status = "healthy"
                data = response.json()
                sglang_models = [m.get("id") for m in data.get("data", [])]
            else:
                sglang_status = f"error: {response.status_code}"
    except httpx.ConnectError:
        sglang_status = "unreachable"
    except Exception as e:
        sglang_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "sglang": {
            "url": SGLANG_URL,
            "status": sglang_status,
            "models": sglang_models,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
