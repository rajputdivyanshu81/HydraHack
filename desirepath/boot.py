"""Boot the live system for the demo: wait for HydraDB, load the damaged graph if empty, build the System.

Mode: with ANTHROPIC_API_KEY -> live Claude; without a key but with the recorded cache -> replays the real
Claude planner/extractor from cache (the hosted demo); otherwise the offline rule-based planner/extractor."""
from __future__ import annotations

import json
import os

from . import llm
from .data import OUT
from .db import Hydra
from .graph import Graph, load_prepared
from .heal import Corpus
from .system import Config, System


def cache_rows() -> int:
    if not os.path.exists(llm.CACHE_PATH):
        return 0
    try:
        return llm._get_con().execute("SELECT count(*) FROM cache").fetchone()[0]
    except Exception:  # noqa: BLE001
        return 0


def boot(log_path: str, progress=print):
    db = Hydra()
    progress("waiting for HydraDB ...")
    db.wait_ready(180)
    ents, kb = load_prepared()
    g = Graph(db, ents)
    have = db.run("MATCH (n:Entity) RETURN count(*) AS n")[0]["n"]
    if have == 0:
        progress("loading the damaged knowledge graph into HydraDB (about a minute) ...")
        g.load(kb["kept"], log=progress)
    else:
        g.mirror(kb["kept"])
    if os.getenv("ANTHROPIC_API_KEY"):
        mode = "live Claude Haiku 5.5"
        cfg = Config(heal="demand", plans="learn", planner="llm", extractor="llm")
    elif cache_rows() > 1000:
        mode = "replaying recorded Claude Haiku 5.5 responses (no API key)"
        cfg = Config(heal="demand", plans="learn", planner="llm", extractor="llm")
    else:
        mode = "offline (rule planner, regex extractor)"
        cfg = Config(heal="demand", plans="learn", planner="rules", extractor="rules")
    sysm = System(db, g, Corpus(), cfg, log_path)
    wl = json.load(open(os.path.join(OUT, "workload.json")))
    return db, g, sysm, wl, mode
