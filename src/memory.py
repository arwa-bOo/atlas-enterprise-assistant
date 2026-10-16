"""Session memory keyed by session_id."""

import json
from threading import RLock

from src.settings import OUTPUTS_DIR

MEMORY_PATH = OUTPUTS_DIR / "session_memory.json"
_LOCK = RLock()


def _read_memory() -> dict:
    """Read saved sessions from disk."""
    if not MEMORY_PATH.exists():
        return {}

    try:
        with MEMORY_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        # Do not silently overwrite a corrupted memory file.
        raise RuntimeError(f"Could not safely read session memory: {MEMORY_PATH}")


def _write_memory(data: dict) -> None:
    """Persist sessions safely using a temporary file."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = MEMORY_PATH.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temp_path.replace(MEMORY_PATH)


def load_history(session_id: str) -> list[dict]:
    """Return previous user/assistant turns for this session."""
    if not session_id.strip():
        raise ValueError("session_id cannot be empty")

    with _LOCK:
        data = _read_memory()
        history = data.get(session_id, [])

        if not isinstance(history, list):
            return []

        return [
            item
            for item in history
            if isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
        ]


def save_turn(session_id: str, question: str, answer: str) -> None:
    """Append one question/answer pair to the session."""
    if not session_id.strip():
        raise ValueError("session_id cannot be empty")

    if not isinstance(question, str) or not isinstance(answer, str):
        raise TypeError("question and answer must be strings")

    with _LOCK:
        data = _read_memory()
        history = data.setdefault(session_id, [])

        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )

        _write_memory(data)


def clear_session(session_id: str) -> None:
    """Delete the conversation history for one session."""
    if not session_id.strip():
        raise ValueError("session_id cannot be empty")

    with _LOCK:
        data = _read_memory()
        data.pop(session_id, None)
        _write_memory(data)
