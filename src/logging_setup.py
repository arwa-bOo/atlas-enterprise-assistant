"""Structured JSON logging. Owner: Person 3."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src import settings

# Prefer the shared value from settings.py (contract: no hard-coded values).
RUN_LOG_PATH = getattr(settings, "RUN_LOG_PATH", "outputs/run_log.jsonl")


def new_run_id() -> str:
    """Return a fresh UUID string for one question."""
    return str(uuid.uuid4())


def _one_chunk(chunk) -> dict:
    """Loggable summary of one chunk: filename/IDs only, never the full text."""
    if isinstance(chunk, dict):
        get = chunk.get
    else:

        def get(key, default=None):
            return getattr(chunk, key, default)

    return {
        "source_filename": get("source_filename"),
        "chunk_id": get("chunk_id"),
        "domain": get("domain"),
        "score": get("score"),
    }


def chunk_metadata(chunks) -> list[dict]:
    """Return a list of loggable dicts for the retrieved chunks."""
    return [_one_chunk(c) for c in (chunks or [])]


def log_event(run_id: str, event: str, **data) -> None:
    """Append one JSON line to RUN_LOG_PATH. Log filenames/chunk IDs, not full text."""
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "event": event,
        **data,
    }
    path = Path(RUN_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
