"""Append-only JSONL log of every graph edit and why it happened (shown in the demo)."""
from __future__ import annotations

import json
import os
import threading
import time


class EditLog:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        self.f = open(path, "a", encoding="utf-8")
        self.counts: dict[str, int] = {}

    def write(self, kind: str, **fields):
        rec = {"t": round(time.time(), 3), "kind": kind, **fields}
        with self.lock:
            self.f.write(json.dumps(rec, default=str) + "\n")
            self.f.flush()
            self.counts[kind] = self.counts.get(kind, 0) + 1

    def close(self):
        self.f.close()
