"""
Run event log written by the pytest plugin (``--fe-events DIR``).

Each process appends JSON lines to its own file so writes never interleave:
the main/controller process writes ``events.jsonl``; each pytest-xdist
worker writes ``events-<worker id>.jsonl``. Readers (the fasteval UI) merge
all files in the directory. When the run ends, ``result.json`` is written.

Event types (``type`` field):
    run_start   pytest started             (main/controller)
    collected   selected evals + metadata  (main process, or xdist worker gw0)
    test_start  a test began               (main/controller)
    eval        one fe.score() result      (process that ran the test)
    test_end    a test finished            (main/controller)
    run_end     run finished + summary     (main/controller)
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

EVENTS_FILE = "events.jsonl"
RESULT_FILE = "result.json"
MAX_STRING = 20_000


def bounded(value: Any, limit: int = MAX_STRING) -> Any:
    """Recursively truncate long strings so event lines stay manageable."""
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + f"\n… [{len(value) - limit} chars truncated]"
    if isinstance(value, dict):
        return {k: bounded(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [bounded(v, limit) for v in value]
    return value


class EventWriter:
    """Append-only JSONL writer, safe to call from multiple threads."""

    def __init__(self, directory: Path, worker_id: Optional[str] = None) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        name = EVENTS_FILE if worker_id is None else f"events-{worker_id}.jsonl"
        self._file = (directory / name).open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self.worker_id = worker_id

    def emit(self, type: str, **fields: Any) -> None:
        event: Dict[str, Any] = {"type": type, "ts": time.time(), **fields}
        if self.worker_id:
            event["worker"] = self.worker_id
        line = json.dumps(bounded(event), default=str)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def write_result(self, payload: Dict[str, Any]) -> None:
        payload = {"finished_at": now_iso(), **payload}
        tmp = self.directory / (RESULT_FILE + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.directory / RESULT_FILE)

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
