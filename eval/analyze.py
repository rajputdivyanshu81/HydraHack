"""Where do held-out misses come from? Classifies each miss against a reference plan checked on the full KB.

    python -m eval.analyze results/desirepath.heldout_r6.json
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from desirepath.data import FWD, INV, OUT, RELATIONS  # noqa: E402


def adj(triples):
    a = defaultdict(lambda: defaultdict(set))
    for s, r, o in triples:
        a[s][FWD[r]].add(o)
        a[o][INV[r]].add(s)
    return a


def run(a, topic, plan):
    cnt = Counter({topic: 1})
    for rel in plan:
        nxt = Counter()
        for n, c in cnt.items():
            for m in a[n].get(rel, ()):
                nxt[m] += c
        cnt = nxt
    cnt.pop(topic, None)
    return [x for x, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]


def main():
    rows = json.load(open(sys.argv[1]))
    kb = json.load(open(os.path.join(OUT, "kb.json")))
    kind = json.load(open(os.path.join(OUT, "entities.json")))["kind"]
    qs = {q["qid"]: q for q in json.load(open(os.path.join(OUT, "workload.json")))["heldout"]}
    full = adj(kb["full"])
    ref = {}
    cats = Counter()
    examples = defaultdict(list)
    for r in rows:
        q = qs[r["qid"]]
        t, k, gold = q["topic"], kind[q["topic"]], set(q["answers"])
        key = (q["template"], k, q["hop"])
        if key not in ref:
            best, bs = None, -1.0
            for seq in itertools.product(RELATIONS, repeat=q["hop"]):
                at, p = k == "movie", []
                for rel in seq:
                    p.append(FWD[rel] if at else INV[rel]); at = not at
                res = run(full, t, p)
                tp = len(set(res) & gold)
                f = 2 * tp / (len(res) + len(gold)) if tp else 0
                if f > bs:
                    bs, best = f, p
            ref[key] = best
        plan_ok = r["plan"] == ">".join(ref[key] or [])
        if r["hit"]:
            cats["hit"] += 1
        elif not plan_ok:
            cats["miss: wrong plan"] += 1
            examples["wrong plan"].append((q["template"], r["plan"], ">".join(ref[key] or [])))
        elif r["pred"] is None:
            cats["miss: right plan, empty (missing edges)"] += 1
        else:
            cats["miss: right plan, wrong top answer"] += 1
    n = len(rows)
    for c, v in cats.most_common():
        print(f"{c:45s} {v:4d}  {100 * v / n:5.1f}%")
    print("\nwrong-plan examples:")
    for t, used, want in examples["wrong plan"][:12]:
        print(f"  {t[:60]:60s} used {used}  ref {want}")


if __name__ == "__main__":
    main()
