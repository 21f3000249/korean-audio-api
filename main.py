import os
import json
import re
import base64
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("korean-audio-api")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-5"

app = FastAPI()

REQUIRED_KEYS = [
    "rows", "columns", "mean", "std", "variance", "min", "max",
    "median", "mode", "range", "allowed_values", "value_range", "correlation",
]

SYSTEM_PROMPT = """You listen to a transcript (in Korean, transcribed from audio) that
describes the specification of a dataset out loud - e.g. how many rows it has,
what columns/fields it contains, and statistics for each column (mean, std,
variance, min, max, median, mode, range, allowed categorical values, value
ranges, and correlations between columns).

Extract EXACTLY this JSON structure from what is stated in the transcript:

{
  "rows": <integer, total row count mentioned>,
  "columns": [<list of column name strings, in the order mentioned>],
  "mean": {<column name>: <number>, ...},
  "std": {<column name>: <number>, ...},
  "variance": {<column name>: <number>, ...},
  "min": {<column name>: <number>, ...},
  "max": {<column name>: <number>, ...},
  "median": {<column name>: <number>, ...},
  "mode": {<column name>: <number or string>, ...},
  "range": {<column name>: <number>, ...},
  "allowed_values": {<column name>: [<allowed category strings>], ...},
  "value_range": {<column name>: [<min>, <max>], ...},
  "correlation": [{"columns": [<col1>, <col2>], "value": <number>}, ...]
}

Rules:
- Only include a column under a statistic key if that statistic was actually
  stated for that column in the transcript. If a statistic type (e.g. "mode")
  is never mentioned for any column, return it as an empty object {}.
- Numbers must be actual JSON numbers, not strings.
- Respond with ONLY the raw JSON object. No markdown fences, no commentary.
"""


async def _transcribe_korean(audio_bytes: bytes) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    files = {"file": ("audio.wav", audio_bytes, "application/octet-stream")}
    data = {"model": "whisper-large-v3", "language": "ko", "response_format": "text"}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(GROQ_URL, headers=headers, files=files, data=data)
        if resp.status_code >= 400:
            logger.error("GROQ ERROR %s: %s", resp.status_code, resp.text)
            raise HTTPException(
                status_code=502, detail=f"Groq transcription error {resp.status_code}: {resp.text}"
            )
        return resp.text.strip()


def _extract_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


async def _extract_spec(transcript: str) -> dict[str, Any]:
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured.")

    payload = {
        "model": MODEL,
        "max_tokens": 1500,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Transcript:\n{transcript}"}],
        "output_config": {"effort": "low"},
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(ANTHROPIC_URL, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error("ANTHROPIC ERROR %s: %s", resp.status_code, resp.text)
            raise HTTPException(
                status_code=502, detail=f"Anthropic API error {resp.status_code}: {resp.text}"
            )
        data = resp.json()

    text = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
    parsed = _extract_json(text)

    # Ensure all required keys exist even if the model omitted an empty one.
    for key in REQUIRED_KEYS:
        if key not in parsed:
            parsed[key] = [] if key in ("columns", "correlation") else ({} if key != "rows" else 0)

    return parsed


async def _handle(request: Request) -> dict:
    body = await request.json()
    audio_b64 = body.get("audio_base64") or body.get("audio") or ""
    if not audio_b64:
        raise HTTPException(status_code=422, detail="Missing 'audio_base64'.")

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        logger.exception("Base64 decode failed")
        raise HTTPException(status_code=422, detail=f"Invalid base64 audio: {e}")

    try:
        transcript = await _transcribe_korean(audio_bytes)
        logger.info("TRANSCRIPT: %s", transcript[:500])
        spec = await _extract_spec(transcript)
        return spec
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in _handle")
        raise HTTPException(status_code=502, detail=f"Unexpected error: {e}")


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/")
async def solve_root(request: Request):
    return await _handle(request)


@app.post("/solve")
async def solve(request: Request):
    return await _handle(request)
