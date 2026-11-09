"""Append-only response cache so long runs survive sleep, crashes and restarts.

Every completed call is written (and fsynced) to a JSONL file as soon as it
returns. Re-running the same command replays cached results and only calls the
model for work that never finished. Repeats are kept distinct: the k-th call
with identical inputs maps to the k-th cached result, so three repeats of one
conversation stay three independent samples.
"""

import hashlib
import json
import os
import threading
from collections import defaultdict
from pathlib import Path


class ResponseCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stored: dict[str, list[dict]] = defaultdict(list)
        self._served: dict[str, int] = defaultdict(int)
        if path.exists():
            text = path.read_text()
            if text and not text.endswith("\n"):
                with path.open("a") as fh:  # isolate a torn last line so the next record starts clean
                    fh.write("\n")
            for line in text.splitlines():
                if line.strip():
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn last line from an interrupted write is ignored
                    self._stored[rec["key"]].append(rec["value"])

    @staticmethod
    def key(payload) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get_or_call(self, payload, fn):
        k = self.key(payload)
        with self._lock:
            i = self._served[k]
            self._served[k] += 1
            if i < len(self._stored[k]):
                return self._stored[k][i]
        value = fn()
        with self._lock:
            self._stored[k].append(value)
            with self.path.open("a") as fh:
                fh.write(json.dumps({"key": k, "value": value}, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return value

    @property
    def hits(self) -> int:
        return sum(min(self._served[k], len(v)) for k, v in self._stored.items())
