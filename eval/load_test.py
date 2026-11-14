"""Load test: concurrent traversal readers while a single writer applies healing/reweighting edits.

    python -m eval.load_test --readers 32 --seconds 60

Readers replay held-out plans (2-3 hop traversals) over HTTP; the writer continuously writes text edges
and SETs USES-edge costs, as the learning loop does. Reports p50/p95/p99 latency, QPS, errors.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from desirepath.answer import chain  # noqa: E402
from desirepath.data import OUT  # noqa: E402
from desirepath.db import Hydra  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", type=int, default=32)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--rows", default=os.path.join(ROOT, "results", "desirepath.heldout_r6.json"))
    args = ap.parse_args()

    ents = json.load(open(os.path.join(OUT, "entities.json")))["ent_id"]
    wl = json.load(open(os.path.join(OUT, "workload.json")))
    qs = {q["qid"]: q for q in wl["heldout"] + wl["stream"]}
    work = []
    rows_path = args.rows
    if not os.path.exists(rows_path):   # fall back to any held-out result with plans
        cands = sorted(f for f in os.listdir(os.path.join(ROOT, "results")) if ".heldout_r" in f)
        rows_path = os.path.join(ROOT, "results", cands[-1])
    for r in json.load(open(rows_path)):
        if r.get("plan"):
            work.append((ents[qs[r["qid"]]["topic"]], r["plan"].split(">")))
    print(f"{len(work)} traversal queries from {os.path.basename(rows_path)}")

    lat: list[float] = []
    errors = [0]
    writes = [0]
    stop = time.time() + args.seconds
    lock = threading.Lock()

    def reader(seed):
        db = Hydra(timeout=30)
        rng = random.Random(seed)
        while time.time() < stop:
            topic, plan = rng.choice(work)
            t0 = time.perf_counter()
            try:
                db.run(chain(plan, len(plan), str(len(plan))), {"t": topic})
                with lock:
                    lat.append((time.perf_counter() - t0) * 1000)
            except Exception:  # noqa: BLE001
                with lock:
                    errors[0] += 1

    def writer():
        db = Hydra(timeout=30)
        rng = random.Random(99)
        ids = list(ents.values())
        eid = 90_000_000
        while time.time() < stop:
            rows = [{"eid": eid + i, "s": rng.choice(ids), "d": rng.choice(ids), "c": 1.0, "src": "loadtest"}
                    for i in range(50)]
            eid += 50
            try:
                db.batch("UNWIND $rows AS row MATCH (a:Entity {id: row.s}), (b:Entity {id: row.d}) "
                         "CREATE (a)-[r:LOADTEST {id: row.eid, cost: row.c, src: row.src}]->(b)", rows)
                with lock:
                    writes[0] += 1
            except Exception:  # noqa: BLE001
                with lock:
                    errors[0] += 1

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(args.readers)]
    threads.append(threading.Thread(target=writer))
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dur = time.time() - t0
    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else 0  # noqa: E731
    out = {"readers": args.readers, "seconds": round(dur, 1), "reads": len(lat), "qps": round(len(lat) / dur, 1),
           "p50_ms": round(q(0.5), 1), "p95_ms": round(q(0.95), 1), "p99_ms": round(q(0.99), 1),
           "mean_ms": round(statistics.mean(lat), 1) if lat else 0, "write_batches": writes[0],
           "edges_written": writes[0] * 50, "errors": errors[0]}
    Hydra().write("MATCH (a:Entity)-[r:LOADTEST]->(b:Entity) DELETE r")
    out["compression"] = compression_bench()
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", f"load_test_{args.readers}.json"), "w"), indent=1)


def compression_bench(reps: int = 5) -> dict:
    """Latency of SHORTCUT reads vs the full multi-hop traversal they replace (same topic, plan)."""
    db = Hydra()
    keys = db.run("MATCH (a:Entity)-[s:SHORTCUT]->(b:Entity) RETURN a.id AS t, s.plan AS p, count(*) AS n")
    if not keys:
        return {"shortcuts": 0}
    full, short = [], []
    for k in keys:
        plan = k["p"].split(">")
        for _ in range(reps):
            t0 = time.perf_counter()
            db.run(chain(plan, len(plan), str(len(plan))), {"t": k["t"]})
            full.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            db.run("MATCH (a:Entity {id: $t})-[s:SHORTCUT]->(x:Entity) WHERE s.plan = $p RETURN x.id AS x, s.n AS c",
                   {"t": k["t"], "p": k["p"]})
            short.append((time.perf_counter() - t0) * 1000)
    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    return {"shortcuts": len(keys), "hops_mean": round(sum(len(k["p"].split(">")) for k in keys) / len(keys), 2),
            "full_traversal_p50_ms": round(med(full), 2), "shortcut_p50_ms": round(med(short), 2),
            "speedup": round(med(full) / max(1e-6, med(short)), 2)}


if __name__ == "__main__":
    main()
