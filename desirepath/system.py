"""DesirePath system: answer by traversal, learn from feedback by editing the graph.

Config
  heal:     none | demand | random       gap-driven ingestion (demand) vs blind ingestion (random)
  plans:    static | learn               planner only, or plan memory reweighted from feedback
  planner:  llm | rules                  cold-start planner
  extractor: llm | rules                 text -> triples
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from . import llm
from .answer import Result, execute, f1, llm_plans, rule_plans, valid_plan
from .compress import Compressor
from .data import ALL_TYPES
from .db import Hydra
from .editlog import EditLog
from .graph import Graph
from .heal import Corpus, Healer
from .plans import COST_INIT, PlanMemory


@dataclass
class Config:
    heal: str = "demand"
    plans: str = "learn"
    planner: str = "llm"
    extractor: str = "llm"
    max_tries: int = 3
    compress: bool = True            # path compression (SHORTCUT edges for hot traversals)
    resolve: bool = True             # entity resolution: alias merge + new entities from text
    feedback_noise: float = 0.0
    random_docs_per_q: float = 2.0   # random-ingestion control: matched to what demand healing read
    seed: int = 0


class System:
    def __init__(self, db: Hydra, graph: Graph, corpus: Corpus, cfg: Config, log_path: str):
        self.db, self.g, self.cfg = db, graph, cfg
        self.log = EditLog(log_path)
        self.healer = Healer(graph, corpus, cfg.extractor, self.log, resolve=cfg.resolve)
        self.memory = PlanMemory(db, self.log)
        self.compressor = Compressor(db, self.log)
        graph.on_write.append(self.compressor.invalidate)
        self.rng = random.Random(cfg.seed)
        self.corpus = corpus

    # ------------------------------------------------------------------ planning
    def planner_plans(self, q: dict, kind: str) -> list[list[str]]:
        masked = q["template"]
        if self.cfg.planner == "llm":
            return llm_plans(masked, kind)
        return rule_plans(masked, kind)

    def candidates(self, q: dict, kind: str) -> tuple[list[list[str]], str]:
        """Ordered candidate plans and where the first came from (memory | planner)."""
        cands, source = [], "planner"
        if self.cfg.plans == "learn":
            mem = [(p, s, c) for p, s, c in self.memory.lookup(q["template"]) if valid_plan(p, kind)]
            good = [p for p, s, c in mem if c and s < COST_INIT]
            if good:                      # confident memory: skip the LLM planner entirely
                return good + [p for p, _, _ in mem if p not in good], "memory"
        for p in self.planner_plans(q, kind):
            if p not in cands:
                cands.append(p)
        return cands, source

    # ------------------------------------------------------------------ answering
    def answer(self, q: dict, probe: bool = False) -> dict:
        t0 = time.perf_counter()
        tok0 = llm.USAGE.input_tokens + llm.USAGE.output_tokens
        topic = self.g.ent_id[q["topic"]]
        kind = self.g.kind.get(q["topic"], "movie")
        cands, source = self.candidates(q, kind)
        res: Result | None = None
        tried = []
        shortcut = False
        # Fallbacks must answer the same kind of question: same final relation as the first plan.
        # (An empty result is honest and triggers healing; a different answer type is just wrong.)
        tryable = [cands[0]] + [p for p in cands[1:] if p[-1] == cands[0][-1] and len(p) == len(cands[0])] \
            if cands else []
        for plan in tryable[: self.cfg.max_tries]:
            if self.cfg.compress and self.compressor.has(topic, plan):
                t1, q1 = time.perf_counter(), self.db.stats["queries"]
                res = Result(plan=plan, ranked=self.compressor.read(topic, plan))
                res.ms, res.queries = (time.perf_counter() - t1) * 1000, self.db.stats["queries"] - q1
                shortcut = True
            else:
                res = execute(self.db, topic, plan, probe_gaps=probe)
            tried.append(res)
            if res.ranked:
                break
        pred = self.g.name[res.ranked[0][0]] if res and res.ranked else None
        gold = set(q["answers"])
        pred_set = {self.g.name[x] for x, _ in (res.ranked if res else [])}
        return {
            "qid": q["qid"], "hop": q["hop"], "split": q.get("split"), "pred": pred,
            "hit": int(pred in gold), "f1": f1(pred_set, gold), "plan": ">".join(res.plan) if res else None,
            "plan_source": source, "shortcut": shortcut, "tries": len(tried),
            "queries": sum(r.queries for r in tried),
            "ms": (time.perf_counter() - t0) * 1000,
            "tokens": llm.USAGE.input_tokens + llm.USAGE.output_tokens - tok0,
            "_res": tried, "_topic": topic, "_kind": kind, "_cands": cands,
        }

    def answer_with_repair(self, q: dict) -> dict:
        """Serving mode: if traversal hits a gap, heal from text (no feedback, no gold) and answer again."""
        first = self.answer(q, probe=True)
        if self.cfg.heal == "none":
            return first
        tok0 = llm.USAGE.input_tokens + llm.USAGE.output_tokens
        t0 = time.perf_counter()
        added = self.healer.heal(self.repair_targets(first), cause=f"query-time repair {q['qid']}")
        heal_ms = (time.perf_counter() - t0) * 1000
        heal_tok = llm.USAGE.input_tokens + llm.USAGE.output_tokens - tok0
        if not added:
            first["repaired"] = False
            return first
        second = self.answer(q)
        second["ms"] += first["ms"] + heal_ms
        second["tokens"] += first["tokens"] + heal_tok
        second["queries"] += first["queries"]
        second["repaired"] = True
        return second

    def repair_targets(self, out: dict, frontier_cap: int = 40) -> list[int]:
        """Where to read text: the topic, nodes with a detected gap (no next edge at all), then the
        first-hop frontier the answer flows through. With random deletion most nodes are *partially*
        incomplete, so zero-edge gaps alone miss most of the damage."""
        targets = [out["_topic"]]
        for r in out["_res"]:
            for lvl in sorted(r.gaps):
                targets += r.gaps[lvl]
        for r in out["_res"][:1]:
            front = r.frontier.get(1, [])
            if len(front) <= frontier_cap:
                targets += front
        return list(dict.fromkeys(targets))

    # ------------------------------------------------------------------ learning
    def learn(self, q: dict, out: dict):
        topic, kind = out["_topic"], out["_kind"]
        gold = set(q["answers"])
        # 1. gap-driven ingestion: read text about where traversal got stuck
        if self.cfg.heal == "demand":
            self.healer.heal(self.repair_targets(out), cause=f"gap on q {q['qid']}")
        elif self.cfg.heal == "random":
            # same reading budget as demand-driven healing, but blind to where queries go
            self._random_credit = getattr(self, "_random_credit", 0.0) + self.cfg.random_docs_per_q
            n = int(self._random_credit)
            self._random_credit -= n
            if n:
                docs = self.rng.sample(range(len(self.corpus.docs)), n * 3)
                ents = [self.g.ent_id[self.corpus.docs[d]["title"]] for d in docs
                        if self.corpus.docs[d]["title"] in self.g.ent_id]
                self.healer.heal(ents, cause="random", budget=n)
        # 2. plan reweighting from feedback
        if self.cfg.plans != "learn":
            return
        if self.cfg.feedback_noise and self.rng.random() < self.cfg.feedback_noise:
            # wrong feedback: a random entity (ids are not contiguous once text-created entities exist)
            gold = {self.g.name[self.rng.choice(tuple(self.g.name))]}
        cands = list(out["_cands"])
        for p in self.path_plans(topic, gold, kind):
            if p not in cands:
                cands.append(p)
        scored = []
        top_hit = {}
        for i, p in enumerate(cands[:8]):
            r = execute(self.db, topic, p)
            names = [self.g.name[x] for x, _ in r.ranked]
            top_hit[">".join(p)] = bool(names) and names[0] in gold
            # rank by answer quality; ties keep the earlier (planner/memory-preferred) candidate
            scored.append((f1(set(names), gold) + (0.5 if names and names[0] in gold else 0), -i, p))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        best_score, _, best = scored[0] if scored else (0, 0, None)
        if self.cfg.compress and best is not None and top_hit.get(">".join(best)):
            self.compressor.confirm(topic, best)
        used = out["plan"].split(">") if out["plan"] else None
        used_score = next((s for s, _, p in scored if p == used), 0)
        if best is not None and best_score > 0:
            self.memory.update(q["template"], best, True, cause=f"feedback on {q['qid']}")
        if used and used != best and used_score < best_score:
            self.memory.update(q["template"], used, False, cause=f"feedback on {q['qid']}")

    def path_plans(self, topic: int, gold: set, kind: str) -> list[list[str]]:
        """Relation sequences HydraDB finds between the topic and correct answers (algo.SPpaths)."""
        rels = ", ".join(f"'{t}'" for t in ALL_TYPES)
        out = []
        for name in list(gold)[:2]:
            tid = self.g.ent_id.get(name)
            if tid is None or tid == topic:
                continue
            rows = self.db.run(f"CALL algo.SPpaths({{sourceNode: {topic}, targetNode: {tid}, relTypes: [{rels}], "
                               f"maxLen: 3, pathCount: 3}}) YIELD path RETURN path")
            for r in rows:
                seq = [e["edge_type"] for e in r["path"]["relationships"]]
                if valid_plan(seq, kind) and len(seq) >= 2 and seq not in out:
                    out.append(seq)
        return out
