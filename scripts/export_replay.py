"""Record real traces for the static replay site (docs/data.json).

For each chosen held-out question, starting from the damaged graph each time (text-created edges and
entities are wiped between questions so every trace stands alone):
  plain ask -> plan, answer, top answers, detected gaps, HydraDB queries, latency
  repair    -> text read, edges healed (sample), merges/new entities, re-answer

Needs HydraDB running (use the dev instance) and the recorded Claude cache:
    $env:HYDRA_URL="http://127.0.0.1:18443"; $env:HYDRA_SERVICE="hydradb-dev"
    python scripts/export_replay.py
"""
import json
import os
import shutil
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from desirepath.data import ALL_TYPES, OUT  # noqa: E402
from desirepath.db import Hydra  # noqa: E402
from desirepath.graph import Graph, load_prepared  # noqa: E402
from desirepath.heal import Corpus  # noqa: E402
from desirepath.system import Config, System  # noqa: E402
from eval.run_curve import reset_hydra  # noqa: E402

N_FLIP, N_FAIL, N_RIGHT = 30, 6, 4
DOCS = os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)

db = Hydra()
print("resetting HydraDB ...", flush=True)
reset_hydra(db)
ents0, kb = load_prepared()
Graph(db, ents0).load(kb["kept"], log=lambda *_: None)
corpus = Corpus()
wl = json.load(open(os.path.join(OUT, "workload.json")))
held = wl["heldout"]
byid = {q["qid"]: q for q in held}
stars = json.load(open(os.path.join(ROOT, "results", "demo_questions.json")))


def wipe_text():
    """Back to the damaged graph: drop edges and entities created from text."""
    for t in ALL_TYPES:
        db.write(f"MATCH (a:Entity)-[r:{t}]->(b:Entity) WHERE r.src = 'text' DELETE r")
    db.write("MATCH (n:Entity) WHERE n.id >= 5000000 DETACH DELETE n")


def fresh():
    ents, kb2 = load_prepared()
    g = Graph(db, ents)
    g.mirror(kb2["kept"])
    cfg = Config(heal="demand", plans="learn", planner="llm", extractor="llm")
    return g, System(db, g, corpus, cfg, os.path.join(ROOT, "results", "replay_export.edits.jsonl"))


def view(g, out, q):
    res = out["_res"][-1] if out["_res"] else None
    top = [{"name": g.name[x], "paths": c, "ok": g.name[x] in set(q["answers"])} for x, c in (res.ranked[:6] if res else [])]
    gaps = []
    for r in out["_res"]:
        for lvl, ids in sorted(r.gaps.items()):
            gaps += [g.name[i] for i in ids[:4]]
    return {"plan": out["plan"], "source": out["plan_source"], "pred": out["pred"], "hit": out["hit"], "top": top,
            "gaps": list(dict.fromkeys(gaps))[:6], "queries": out["queries"], "ms": round(out["ms"]),
            "n_answers": len(res.ranked) if res else 0}


def trace(qid):
    wipe_text()
    g, s = fresh()
    q = byid[qid]
    plain = s.answer(q, probe=True)
    row = {"qid": qid, "question": q["question"], "hop": q["hop"], "topic": q["topic"], "split": q["split"],
           "gold": sorted(q["answers"])[:8], "n_gold": len(q["answers"]), "plain": view(g, plain, q)}
    st0 = dict(s.healer.stats)
    rs0 = dict(s.healer.resolver.stats)
    t0 = time.perf_counter()
    added = s.healer.heal(s.repair_targets(plain), cause=f"replay {qid}")
    heal_ms = (time.perf_counter() - t0) * 1000
    again = s.answer(q)
    rep = view(g, again, q)
    rep["healed_edges"] = added
    rep["docs_read"] = s.healer.stats["docs_read"] - st0["docs_read"]
    rep["heal_ms"] = round(heal_ms)
    rep["merges"] = s.healer.resolver.stats["alias_merge"] - rs0["alias_merge"]
    rep["new_entities"] = s.healer.resolver.stats["new_entity"] - rs0["new_entity"]
    rep["sample_edges"] = [{"movie": g.name[m], "rel": r, "value": g.name[v]} for m, r, v in g.written[:10]]
    row["repair"] = rep
    return row


rows, seen = [], set()
t0 = time.time()
flips, fails = [], []
cands = [k for k in stars[:90]]
others = [q["qid"] for q in held if q["qid"] not in set(stars)]
for qid in cands + others[:40]:
    if len(flips) >= N_FLIP and len(fails) >= N_FAIL:
        break
    r = trace(qid)
    if not r["plain"]["hit"] and r["repair"]["hit"]:
        if len(flips) < N_FLIP:
            flips.append(r)
    elif not r["plain"]["hit"] and not r["repair"]["hit"] and r["repair"]["healed_edges"] > 0:
        if len(fails) < N_FAIL:
            fails.append(r)
    print(f"{len(flips)} flips, {len(fails)} non-flips after {time.time() - t0:.0f}s", flush=True)
rights = []
for q in held:
    if len(rights) >= N_RIGHT:
        break
    wipe_text()
    g, s = fresh()
    o = s.answer(q, probe=True)
    if o["hit"]:
        rights.append({"qid": q["qid"], "question": q["question"], "hop": q["hop"], "topic": q["topic"],
                       "split": q["split"], "gold": sorted(q["answers"])[:8], "n_gold": len(q["answers"]),
                       "plain": view(g, o, q), "repair": None})
wipe_text()

# headline numbers, straight from the results files
R = os.path.join(ROOT, "results")
def load(n):
    p = os.path.join(R, n + ".json")
    return json.load(open(p)) if os.path.exists(p) else None
def last(n):
    d = load(n)
    return d["rounds"][-1]["heldout"] if d else None
main = load("desirepath")
summary = {
    "rounds": [{"round": r["round"], "hits": r["heldout"]["hits1"], "tokens": r["heldout"]["tokens"]} for r in main["rounds"]],
    "serve_repair": main["serve_repair"]["hits1"],
    "serve_repair_unseen": main["serve_repair"].get("hits1_unseen-topic"),
    "final_unseen": main["rounds"][-1]["heldout"].get("hits1_unseen-topic"),
    "table": {n: (last(n) or {}).get("hits1") for n in ("static", "plans_only", "heal_only", "no_merge", "random_heal", "noisy_fb", "desirepath", "full_kb")},
    "vector": load("baseline_vector")["heldout"]["hits1"], "llm": load("baseline_llm")["heldout"]["hits1"],
    "edges": main["rounds"][-1]["graph"]["text_edges"],
}
data = {"summary": summary, "questions": [*flips, *fails, *rights],
        "note": "Recorded from real HydraDB runs: the graph is reset to its damaged state before each question."}
json.dump(data, open(os.path.join(DOCS, "data.json"), "w"))
shutil.copy(os.path.join(R, "curve.png"), os.path.join(DOCS, "curve.png"))
print(f"wrote docs/data.json: {len(flips)} fixed by repair, {len(fails)} not fixed, {len(rights)} already right "
      f"({os.path.getsize(os.path.join(DOCS, 'data.json')) / 1e3:.0f} KB)")
