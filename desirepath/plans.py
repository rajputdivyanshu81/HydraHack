"""Plan memory stored in HydraDB, reweighted from feedback.

  (:Template {id, text})-[:USES {id, cost, wins, tries}]->(:Plan {id, rels})

Cheap plans (low cost) are tried first. A correct answer lowers the cost of the plan that produced it;
a wrong one raises it; the plan that would have been right (found by traversal) is added or reinforced.
Similar question wordings share plans via a TF-IDF nearest-template lookup (local, no LLM).
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .db import Hydra
from .graph import EDGE_ID_USES, PLAN_ID, TEMPLATE_ID

COST_INIT, COST_WIN, COST_LOSS, COST_FLOOR, COST_CAP = 1.0, 0.7, 1.3, 0.2, 5.0
SIM_MIN = 0.75
MIN_WINS, MIN_RATE = 2, 0.6   # evidence needed before memory overrides the planner


def signature(text: str) -> frozenset:
    """Which relations a wording mentions. Templates only share plans when this matches exactly."""
    from .answer import KEYWORDS
    q = text.lower()
    return frozenset(r for r, ks in KEYWORDS.items() if any(k in q for k in ks))


def _tokens(text: str) -> list[str]:
    w = re.findall(r"[a-z]+", text.lower())
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]


class PlanMemory:
    def __init__(self, db: Hydra, editlog=None):
        self.db = db
        self.log = editlog
        self.templates: dict[str, int] = {}      # text -> template id
        self.plans: dict[str, int] = {}          # "A>B" -> plan id
        self.uses: dict[tuple[int, int], dict] = {}   # (tid, pid) -> {id, cost, wins, tries}
        self.next_uses = EDGE_ID_USES
        self._vecs: list[Counter] = []
        self._texts: list[str] = []
        self._df: Counter = Counter()

    # --------------------------------------------------------------- similarity
    def _vec(self, text: str) -> dict[str, float]:
        tf = Counter(_tokens(text))
        n = max(1, len(self._texts))
        v = {t: c * math.log(1 + n / (1 + self._df[t])) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def neighbours(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        if not self._texts:
            return []
        sig = signature(text)
        q = self._vec(text)
        sims = []
        for t in self._texts:
            if t != text and signature(t) != sig:
                continue
            s = 1.0 if t == text else sum(q.get(w, 0.0) * x for w, x in self._vec(t).items())
            sims.append((t, s))
        sims.sort(key=lambda x: -x[1])
        return [(t, s) for t, s in sims[:k] if s >= SIM_MIN]

    # --------------------------------------------------------------- graph reads
    def lookup(self, text: str) -> list[tuple[list[str], float, bool]]:
        """Candidate plans from HydraDB for the nearest templates: (rels, score, confident)."""
        out: dict[str, tuple[float, bool]] = {}
        for t, sim in self.neighbours(text):
            rows = self.db.run("MATCH (t:Template {id: $t})-[u:USES]->(p:Plan) "
                               "RETURN p.rels AS rels, u.cost AS cost, u.wins AS wins, u.tries AS tries",
                               {"t": self.templates[t]})
            for r in rows:
                score = r["cost"] / sim          # lower is better
                tries = r["tries"] or 0
                confident = (r["wins"] or 0) >= MIN_WINS and tries and r["wins"] / tries >= MIN_RATE
                if r["rels"] not in out or score < out[r["rels"]][0]:
                    out[r["rels"]] = (score, bool(confident))
        ranked = sorted(out.items(), key=lambda kv: kv[1][0])
        return [(k.split(">"), s, c) for k, (s, c) in ranked]

    # --------------------------------------------------------------- graph writes
    def _template(self, text: str) -> int:
        if text not in self.templates:
            tid = TEMPLATE_ID + len(self.templates) + 1
            self.db.batch("UNWIND $rows AS row MERGE (t {id: row.id}) SET t:Template, t.text = row.text",
                          [{"id": tid, "text": text}])
            self.templates[text] = tid
            self._texts.append(text)
            for tok in set(_tokens(text)):
                self._df[tok] += 1
        return self.templates[text]

    def _plan(self, rels: list[str]) -> int:
        key = ">".join(rels)
        if key not in self.plans:
            pid = PLAN_ID + len(self.plans) + 1
            self.db.batch("UNWIND $rows AS row MERGE (p {id: row.id}) SET p:Plan, p.rels = row.rels",
                          [{"id": pid, "rels": key}])
            self.plans[key] = pid
        return self.plans[key]

    def update(self, text: str, rels: list[str], win: bool, cause: str):
        tid, pid = self._template(text), self._plan(rels)
        u = self.uses.get((tid, pid))
        if u is None:
            u = {"id": self.next_uses, "cost": COST_INIT, "wins": 0, "tries": 0}
            self.next_uses += 1
            self.db.batch("UNWIND $rows AS row MATCH (t:Template {id: row.t}), (p:Plan {id: row.p}) "
                          "CREATE (t)-[u:USES {id: row.id, cost: row.c, wins: row.w, tries: row.n}]->(p)",
                          [{"t": tid, "p": pid, "id": u["id"], "c": u["cost"], "w": 0, "n": 0}])
            self.uses[(tid, pid)] = u
        before = u["cost"]
        u["cost"] = max(COST_FLOOR, u["cost"] * COST_WIN) if win else min(COST_CAP, u["cost"] * COST_LOSS)
        u["tries"] += 1
        u["wins"] += int(win)
        self.db.write("MATCH (t:Template {id: $t})-[u:USES]->(p:Plan {id: $p}) "
                      "SET u.cost = $c, u.wins = $w, u.tries = $n",
                      {"t": tid, "p": pid, "c": float(u["cost"]), "w": u["wins"], "n": u["tries"]})
        if self.log is not None:
            self.log.write("reweight", cause=cause, template=text, plan=">".join(rels),
                           cost_before=round(before, 3), cost_after=round(u["cost"], 3))
