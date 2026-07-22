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

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
EXTRACT_MODEL = "llama-3.3-70b-versatile"

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
- CRITICAL: "columns" must list the name of EVERY field mentioned anywhere in
  the transcript, in the order first mentioned - including fields that are
  only described via allowed values, a value range, or a single statistic,
  even if the word "column" is never said. If the transcript says something
  like "카테고리는 A, B, C 중 하나입니다" (category is one of A, B, C), that
  means there IS a column named "카테고리", and it must appear in "columns"
  AND in "allowed_values" as {"카테고리": ["A", "B", "C"]}. Do not leave
  "columns" empty if any field-level fact is stated anywhere in the transcript.
- Only include a column under a statistic key (mean/std/etc.) if that
  specific statistic was actually stated for that column. If a statistic type
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


_CATEGORICAL_PATTERN = re.compile(
    r"([가-힣A-Za-z0-9_]+)(?:는|은|이|가)\s*([^.]+?)\s*중\s*(?:하나|하나입니다|하나이다)"
)


def _regex_categorical_fields(transcript: str) -> dict[str, list[str]]:
    """Deterministically catch '<field>는 A, B, C 중 하나입니다' style statements
    that name a categorical column and its allowed values, so we don't depend
    entirely on the LLM noticing this pattern."""
    found: dict[str, list[str]] = {}
    for match in _CATEGORICAL_PATTERN.finditer(transcript):
        field_name = match.group(1).strip()
        values_blob = match.group(2)
        # Split "A, B, C" / "A,B,C" / "A, B and C" style lists.
        values = [v.strip() for v in re.split(r"[,，、]|\s+(?:and|및|또는)\s+", values_blob) if v.strip()]
        if field_name and values:
            found[field_name] = values
    return found


def _normalize_field_name(name: str) -> str:
    """Whisper often transcribes '점수1' as '점수 1' (space before a trailing
    number). Collapse that so column names match the expected exact form."""
    if not isinstance(name, str):
        return name
    return re.sub(r"(?<=[가-힣A-Za-z])\s+(?=\d)", "", name).strip()


def _normalize_spec_names(spec: dict[str, Any]) -> dict[str, Any]:
    if isinstance(spec.get("columns"), list):
        spec["columns"] = [_normalize_field_name(c) for c in spec["columns"]]

    per_column_keys = [
        "mean", "std", "variance", "min", "max", "median", "mode",
        "range", "allowed_values", "value_range",
    ]
    for key in per_column_keys:
        if isinstance(spec.get(key), dict):
            spec[key] = {
                _normalize_field_name(col): val for col, val in spec[key].items()
            }

    if isinstance(spec.get("correlation"), list):
        for entry in spec["correlation"]:
            if isinstance(entry, dict) and isinstance(entry.get("columns"), list):
                entry["columns"] = [_normalize_field_name(c) for c in entry["columns"]]

    return spec


def _merge_regex_fallback(spec: dict[str, Any], transcript: str) -> dict[str, Any]:
    categorical = _regex_categorical_fields(transcript)
    for field_name, values in categorical.items():
        if field_name not in spec.get("columns", []):
            spec.setdefault("columns", []).append(field_name)
        spec.setdefault("allowed_values", {})
        if field_name not in spec["allowed_values"]:
            spec["allowed_values"][field_name] = values
    return spec


async def _extract_spec(transcript: str) -> dict[str, Any]:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    payload = {
        "model": EXTRACT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(GROQ_CHAT_URL, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error("GROQ CHAT ERROR %s: %s", resp.status_code, resp.text)
            raise HTTPException(
                status_code=502, detail=f"Groq chat error {resp.status_code}: {resp.text}"
            )
        data = resp.json()

    text = data["choices"][0]["message"]["content"]
    parsed = _extract_json(text)

    # Ensure all required keys exist even if the model omitted an empty one.
    for key in REQUIRED_KEYS:
        if key not in parsed:
            parsed[key] = [] if key in ("columns", "correlation") else ({} if key != "rows" else 0)

    parsed = _merge_regex_fallback(parsed, transcript)
    parsed = _normalize_spec_names(parsed)
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
