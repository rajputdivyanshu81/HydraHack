"""Claude calls (Haiku 5.5) with structured JSON output, token accounting and a SQLite cache.

The cache makes every eval rerun free and reproducible: same prompt -> same stored response.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any

import anthropic

MODEL = os.getenv("DESIREPATH_MODEL", "claude-haiku-5-5")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CACHE_PATH = os.path.join(ROOT, "results", "cache", "llm.sqlite")

# USD per 1M tokens for claude-haiku-5-5 (prompts under 100K tokens).
PRICE_IN, PRICE_OUT = 0.10, 0.50


class CacheMiss(RuntimeError):
    """Raised when there is no API key and the request is not in the recorded cache."""


class Usage:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.calls = self.cached = self.input_tokens = self.output_tokens = 0

    def add(self, inp: int, out: int, cached: bool):
        with self.lock:
            self.calls += 1
            if cached:
                self.cached += 1
            self.input_tokens += inp
            self.output_tokens += out

    @property
    def usd(self) -> float:
        return self.input_tokens / 1e6 * PRICE_IN + self.output_tokens / 1e6 * PRICE_OUT

    def snapshot(self) -> dict:
        return {"calls": self.calls, "cached": self.cached, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usd": round(self.usd, 4)}


USAGE = Usage()
_client: anthropic.Anthropic | None = None
_db_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    con = sqlite3.connect(CACHE_PATH, check_same_thread=False, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT, inp INT, out INT)")
    return con


_con = None


def _get_con():
    global _con
    if _con is None:
        _con = _db()
    return _con


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=6)
    return _client


def call_json(system: str, prompt: str, schema: dict, *, max_tokens: int = 2000,
              effort: str = "low", tag: str = "") -> dict:
    """One Claude call constrained to `schema`. Cached by (model, system, prompt, schema)."""
    key = hashlib.sha256(json.dumps([MODEL, system, prompt, schema, effort], sort_keys=True).encode()).hexdigest()
    con = _get_con()
    with _db_lock:
        row = con.execute("SELECT v, inp, out FROM cache WHERE k = ?", (key,)).fetchone()
    if row:
        USAGE.add(row[1], row[2], cached=True)
        return json.loads(row[0])

    if not os.getenv("ANTHROPIC_API_KEY"):
        # Hosted demo: no key, only the recorded cache. A miss means this exact request was never recorded.
        raise CacheMiss(f"not in the recorded cache ({tag or 'call'}); this demo has no API key")
    resp = None
    for attempt in range(8):
        try:
            resp = client().messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            )
            break
        except anthropic.APIStatusError as e:
            # The SDK already retries 408/409/429/5xx. Some gateways also return *intermittent* 401s
            # under load (seen in practice), so retry those too, with backoff, before giving up.
            if e.status_code in (400, 403, 404, 413, 422) or attempt == 7:
                raise
            time.sleep(min(30, 2 ** attempt))
        except anthropic.APIConnectionError:
            if attempt == 7:
                raise
            time.sleep(min(30, 2 ** attempt))
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal on {tag}: {resp.stop_details}")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"max_tokens hit on {tag}")
    data = json.loads(text)
    inp, out = resp.usage.input_tokens, resp.usage.output_tokens
    USAGE.add(inp, out, cached=False)
    with _db_lock:
        con.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)", (key, json.dumps(data), inp, out))
        con.commit()
    return data


def obj(props: dict[str, Any], required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or list(props),
            "additionalProperties": False}
