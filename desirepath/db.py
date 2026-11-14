"""Thin client for the HydraDB open-source engine's HTTP query API.

POST /v1/graphs/{graph}/query with {cell_id, query, parameters, bookmark?, consistency?}.
Result cells are typed: {"type": "...", "value": ...}; nulls omit "value".
"""
from __future__ import annotations

import os
import time
from typing import Any, Iterable

import httpx


class HydraError(RuntimeError):
    pass


def _decode(cell: Any) -> Any:
    """Turn a typed HydraDB cell (or nested property value) into plain Python."""
    if isinstance(cell, dict):
        if "type" in cell and set(cell) <= {"type", "value"}:
            return _decode(cell.get("value"))
        # property values arrive tagged, e.g. {"String": "alice"} / {"Float": 1.0}
        if len(cell) == 1:
            (k, v), = cell.items()
            if k in {"String", "Int", "Integer", "Float", "Bool", "Boolean"}:
                return v
        return {k: _decode(v) for k, v in cell.items()}
    if isinstance(cell, list):
        return [_decode(c) for c in cell]
    return cell


class Hydra:
    def __init__(self, url: str | None = None, token: str | None = None,
                 cell: str | None = None, namespace: str | None = None,
                 graph: str = "default", timeout: float = 60.0):
        self.url = (url or os.getenv("HYDRA_URL", "http://127.0.0.1:8443")).rstrip("/")
        token = token or os.getenv("HYDRA_TOKEN") or _read_token()
        self.cell = cell or os.getenv("HYDRA_CELL", "cell-0")
        self.graph = graph
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        ns = namespace or os.getenv("HYDRA_NAMESPACE", "default")
        if ns:
            headers["X-Graph-Namespace"] = ns
        self.http = httpx.Client(base_url=self.url, headers=headers, timeout=timeout)
        self.bookmark: str | None = None  # last write bookmark, for read-your-writes
        self.stats = {"queries": 0, "writes": 0, "ms": 0.0}

    # ------------------------------------------------------------------ core
    def run(self, query: str, params: dict | None = None, *, write: bool = False,
            page_size: int | None = None, retries: int = 3) -> list[dict]:
        body: dict[str, Any] = {"cell_id": self.cell, "query": query}
        if params:
            body["parameters"] = params
        if self.bookmark and not write:
            body["bookmark"] = self.bookmark
        if page_size:
            body["page_size"] = page_size
        t0 = time.perf_counter()
        for attempt in range(retries):
            try:
                r = self.http.post(f"/v1/graphs/{self.graph}/query", json=body)
                break
            except httpx.TransportError:
                if attempt == retries - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
        self.stats["ms"] += (time.perf_counter() - t0) * 1000
        self.stats["queries"] += 1
        try:
            data = r.json()
        except ValueError:
            raise HydraError(f"HTTP {r.status_code}: {r.text[:300]}")
        if not isinstance(data, dict):
            raise HydraError(f"HTTP {r.status_code}: {str(data)[:300]}")
        if "error" in data:
            raise HydraError(f"{data['error'].get('code')}: {data['error'].get('message')}\n  query: {query[:300]}")
        if "rows" not in data:
            raise HydraError(f"HTTP {r.status_code}: unexpected body {str(data)[:300]}")
        if write:
            self.stats["writes"] += 1
            if data.get("bookmark"):
                self.bookmark = data["bookmark"]
        rows = data["rows"]
        cols = data.get("columns", [])
        out = [dict(zip(cols, (_decode(c) for c in row))) for row in rows]
        # follow pagination
        cursor = data.get("next_cursor")
        while cursor:
            nb = dict(body, cursor=cursor, query_id=data["query_id"], page_size=page_size or 1024)
            data = self.http.post(f"/v1/graphs/{self.graph}/query", json=nb).json()
            if "error" in data:
                raise HydraError(str(data["error"]))
            out += [dict(zip(cols, (_decode(c) for c in row))) for row in data["rows"]]
            cursor = data.get("next_cursor")
        return out

    def write(self, query: str, params: dict | None = None) -> list[dict]:
        return self.run(query, params, write=True)

    def batch(self, query: str, rows: Iterable[dict], size: int = 1000) -> int:
        """Run an `UNWIND $rows AS row ...` write in chunks (server cap is 1024)."""
        buf, n = [], 0
        for r in rows:
            buf.append(r)
            if len(buf) >= size:
                self.write(query, {"rows": buf}); n += len(buf); buf = []
        if buf:
            self.write(query, {"rows": buf}); n += len(buf)
        return n

    # ------------------------------------------------------------ health
    def ready(self, admin_url: str | None = None) -> bool:
        default_admin = "http://127.0.0.1:19090" if self.url.endswith(":18443") else "http://127.0.0.1:9090"
        admin = admin_url or os.getenv("HYDRA_ADMIN_URL", default_admin)
        try:
            return httpx.get(f"{admin}/readyz", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    def wait_ready(self, seconds: int = 90) -> None:
        for _ in range(seconds):
            if self.ready():
                return
            time.sleep(1)
        raise HydraError("HydraDB did not become ready")


def _read_token() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "config", "auth-token")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()
