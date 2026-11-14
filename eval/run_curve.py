"""Learning-curve experiment.

    python -m eval.run_curve --variant desirepath --rounds 6
    python -m eval.run_curve --variant static --offline          # rules planner/extractor, no API key

Protocol: round 0 scores the frozen held-out set on the damaged graph. Each round then processes a batch
of stream questions (answer -> score prequentially -> learn from feedback by editing the graph) and
re-scores the held-out set with learning frozen. Held-out questions never produce feedback or ingestion.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from desirepath import llm  # noqa: E402
from desirepath.data import OUT  # noqa: E402
from desirepath.db import Hydra  # noqa: E402
from desirepath.graph import Graph, load_prepared  # noqa: E402
from desirepath.heal import Corpus  # noqa: E402
from desirepath.system import Config, System  # noqa: E402

VARIANTS = {
    "desirepath":  dict(heal="demand", plans="learn"),
    "heal_only":   dict(heal="demand", plans="static"),
    "plans_only":  dict(heal="none", plans="learn"),
    "random_heal": dict(heal="random", plans="learn"),
    "static":      dict(heal="none", plans="static"),
    "full_kb":     dict(heal="none", plans="static"),
    "noisy_fb":    dict(heal="demand", plans="learn", feedback_noise=0.2),
    "no_merge":    dict(heal="demand", plans="learn", resolve=False),
}


def reset_hydra(db: Hydra):
    """Fresh HydraDB volume so every variant starts from the same damaged graph.
    HYDRA_SERVICE selects the compose service (hydradb, or hydradb-dev for tests)."""
    svc = os.getenv("HYDRA_SERVICE", "hydradb")
    vol = "hydra_hydra-dev-data" if svc == "hydradb-dev" else "hydra_hydra-data"
    dc = ["docker", "compose", "--profile", "dev"]
    subprocess.run(dc + ["rm", "-s", "-f", svc], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(["docker", "volume", "rm", "-f", vol], cwd=ROOT, capture_output=True)
    subprocess.run(dc + ["up", "-d", svc], cwd=ROOT, check=True, capture_output=True)
    time.sleep(2)
    db.wait_ready()
    for _ in range(30):
        try:
            db.run("MATCH (n:Entity {id: 1}) RETURN n.id AS id")
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    out = {"n": len(rows), "hits1": sum(r["hit"] for r in rows) / n, "f1": sum(r["f1"] for r in rows) / n,
           "ms_p50": sorted(r["ms"] for r in rows)[len(rows) // 2] if rows else 0,
           "queries": sum(r["queries"] for r in rows) / n, "tokens": sum(r["tokens"] for r in rows) / n,
           "from_memory": sum(r["plan_source"] == "memory" for r in rows) / n,
           "via_shortcut": sum(bool(r.get("shortcut")) for r in rows) / n,
           "ms_mean": sum(r["ms"] for r in rows) / n}
    for split in ("seen-topic", "unseen-topic"):
        sub = [r for r in rows if r.get("split") == split]
        if sub:
            out[f"hits1_{split}"] = sum(r["hit"] for r in sub) / len(sub)
    for hop in (2, 3):
        sub = [r for r in rows if r["hop"] == hop]
        if sub:
            out[f"hits1_{hop}hop"] = sum(r["hit"] for r in sub) / len(sub)
    # bootstrap 95% CI on hits@1
    rng = random.Random(0)
    hits = [r["hit"] for r in rows]
    if hits:
        bs = sorted(sum(rng.choice(hits) for _ in hits) / len(hits) for _ in range(1000))
        out["hits1_ci"] = [bs[25], bs[974]]
    return out


def clean(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="desirepath", choices=list(VARIANTS))
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--per-round", type=int, default=150)
    ap.add_argument("--heldout", type=int, default=300)
    ap.add_argument("--offline", action="store_true", help="rules planner + regex extractor (no API key)")
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--serve-repair", action="store_true", help="after the curve, score held-out with query-time repair")
    args = ap.parse_args()

    cfg = Config(**VARIANTS[args.variant])
    if args.offline:
        cfg.planner = cfg.extractor = "rules"
    name = args.variant + ("_offline" if args.offline else "") + (f"_{args.tag}" if args.tag else "")
    if args.variant == "random_heal":
        ref = os.path.join(ROOT, "results", f"desirepath{'_offline' if args.offline else ''}"
                                            f"{'_' + args.tag if args.tag else ''}.json")
        if os.path.exists(ref):
            r = json.load(open(ref))
            cfg.random_docs_per_q = r["healer"]["docs_read"] / max(1, r["rounds"][-1]["round"] * args.per_round)
        print(f"random ingestion budget: {cfg.random_docs_per_q:.2f} docs/question (matched to desirepath)")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    log_path = os.path.join(ROOT, "results", f"{name}.edits.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    wl = json.load(open(os.path.join(OUT, "workload.json")))
    stream, heldout = wl["stream"], wl["heldout"][: args.heldout]
    if args.heldout < len(wl["heldout"]):   # keep both splits when subsampling
        half = args.heldout // 2
        seen = [q for q in wl["heldout"] if q["split"] == "seen-topic"][:half]
        unseen = [q for q in wl["heldout"] if q["split"] == "unseen-topic"][: args.heldout - half]
        heldout = seen + unseen

    db = Hydra()
    ents, kb = load_prepared()
    g = Graph(db, ents)
    triples = kb["full"] if args.variant == "full_kb" else kb["kept"]
    if not args.no_reset:
        print("resetting HydraDB and loading graph ...")
        reset_hydra(db)
        g.load(triples)
    else:
        g.mirror(triples)
    corpus = Corpus()
    sysm = System(db, g, corpus, cfg, log_path)
    rounds = 0 if args.variant in ("static", "full_kb") else args.rounds

    result = {"variant": name, "config": cfg.__dict__, "rounds": [], "started": time.time()}

    def evaluate(r: int, stream_rows=None):
        llm.USAGE.reset()
        t0 = time.time()
        rows = [sysm.answer(q) for q in heldout]
        s = summarize(rows)
        rec = {"round": r, "heldout": s, "eval_s": round(time.time() - t0, 1),
               "graph": {"text_edges": sysm.healer.stats["edges_added"], "docs_read": sysm.healer.stats["docs_read"],
                         "templates": len(sysm.memory.templates), "plans": len(sysm.memory.plans)},
               "edits": dict(sysm.log.counts)}
        if stream_rows is not None:
            rec["stream"] = summarize(stream_rows)
        result["rounds"].append(rec)
        print(f"[{name}] round {r}: held-out hits@1 {s['hits1']:.3f} "
              f"(seen {s.get('hits1_seen-topic', 0):.3f} / unseen {s.get('hits1_unseen-topic', 0):.3f}) "
              f"f1 {s['f1']:.3f} p50 {s['ms_p50']:.0f}ms tokens/q {s['tokens']:.0f} mem {s['from_memory']:.2f} | "
              f"text edges {rec['graph']['text_edges']} docs {rec['graph']['docs_read']}"
              + (f" | stream hits@1 {rec['stream']['hits1']:.3f}" if stream_rows else ""), flush=True)
        if r == 0 or r == rounds:
            json.dump([clean(x) for x in rows], open(os.path.join(ROOT, "results", f"{name}.heldout_r{r}.json"), "w"))
        json.dump(result, open(os.path.join(ROOT, "results", f"{name}.json"), "w"), indent=1)

    evaluate(0)
    for r in range(1, rounds + 1):
        batch = stream[(r - 1) * args.per_round: r * args.per_round]
        srows = []
        for q in batch:
            out = sysm.answer(q, probe=cfg.heal == "demand")
            srows.append(clean(out))
            sysm.learn(q, out)
        evaluate(r, srows)
    # Serving mode, after the curve is final: query-time repair on the held-out set (text only, no feedback).
    if args.serve_repair and cfg.heal != "none" and rounds:
        llm.USAGE.reset()
        rows = [sysm.answer_with_repair(q) for q in heldout]
        s = summarize(rows)
        s["repaired_share"] = sum(bool(r.get("repaired")) for r in rows) / len(rows)
        result["serve_repair"] = s
        json.dump([clean(x) for x in rows], open(os.path.join(ROOT, "results", f"{name}.heldout_repair.json"), "w"))
        print(f"[{name}] serving with query-time repair: hits@1 {s['hits1']:.3f} "
              f"(seen {s.get('hits1_seen-topic', 0):.3f} / unseen {s.get('hits1_unseen-topic', 0):.3f}) "
              f"repaired {s['repaired_share']:.2f} p50 {s['ms_p50']:.0f}ms tokens/q {s['tokens']:.0f}", flush=True)
    result["finished"] = time.time()
    result["healer"] = sysm.healer.stats
    result["resolver"] = sysm.healer.resolver.stats
    result["compressor"] = sysm.compressor.stats
    json.dump(result, open(os.path.join(ROOT, "results", f"{name}.json"), "w"), indent=1)
    sysm.log.close()


if __name__ == "__main__":
    main()
