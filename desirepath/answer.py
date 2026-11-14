"""Answer a question by traversing HydraDB along a relation plan, and report where the graph has gaps.

A plan is a relation sequence, e.g. ["R_STARRED_ACTORS", "DIRECTED_BY"]:
  topic --R_STARRED_ACTORS--> movies --DIRECTED_BY--> answers
"""
from __future__ import annotations

import itertools
import re
import time
from dataclasses import dataclass, field

from . import llm
from .data import ALL_TYPES, FWD, INV, RELATIONS
from .db import Hydra

REL_DOC = {
    "directed_by": ("movie", "director (person)"),
    "written_by": ("movie", "writer/screenwriter (person)"),
    "starred_actors": ("movie", "actor (person)"),
    "release_year": ("movie", "release year"),
    "in_language": ("movie", "language"),
    "has_tags": ("movie", "descriptive tag / topic keyword"),
    "has_genre": ("movie", "genre"),
    "has_imdb_rating": ("movie", "IMDb rating bucket"),
    "has_imdb_votes": ("movie", "IMDb popularity bucket"),
}


def valid_plan(plan: list[str], topic_kind: str) -> bool:
    """Plans must alternate movie -> value -> movie: forward types leave movies, R_ types enter them."""
    at_movie = topic_kind == "movie"
    for rel in plan:
        if rel not in ALL_TYPES:
            return False
        is_fwd = not rel.startswith("R_")
        if is_fwd != at_movie:
            return False
        at_movie = not at_movie
    return 1 <= len(plan) <= 3


def mask(question: str, topic: str) -> str:
    return question.replace(topic, "NE")


# ---------------------------------------------------------------------- planners
PLAN_SYSTEM = (
    "You translate questions about movies into relation paths over a movie knowledge graph. "
    "Graph: every edge connects a movie to a value. Forward types go movie -> value; the R_ prefixed "
    "inverse types go value -> movie. A path must alternate: from a movie use a forward type, from a "
    "value (person, year, language, genre, tag, rating) use an R_ type.")


def plan_vocab() -> str:
    lines = []
    for r in RELATIONS:
        _, val = REL_DOC[r]
        lines.append(f"- {FWD[r]}: movie -> its {val}")
        lines.append(f"- {INV[r]}: {val} -> movies with that {val.split(' (')[0]}")
    return "\n".join(lines)


PLAN_SCHEMA = llm.obj({"plans": {"type": "array", "items": llm.obj({
    "rels": {"type": "array", "items": {"type": "string", "enum": ALL_TYPES}}})}})


def llm_plans(question_masked: str, topic_kind: str) -> list[list[str]]:
    prompt = (f"Relation types:\n{plan_vocab()}\n\nQuestion: {question_masked}\n"
              f"NE is the topic entity; its type is: {topic_kind}.\n"
              "Return up to 3 candidate relation paths (1 to 3 steps) from NE to the answers, best first. "
              "Answers must not include NE itself. 'Co-' or 'share X with' questions go NE -> X -> back.")
    out = llm.call_json(PLAN_SYSTEM, prompt, PLAN_SCHEMA, max_tokens=1500, tag="plan")
    plans = [p["rels"] for p in out.get("plans", [])]
    return [p for p in plans if valid_plan(p, topic_kind)]


KEYWORDS = {
    "directed_by": ["direct"], "written_by": ["writ", "screenwrit", "wrote", "author"],
    "starred_actors": ["actor", "star", "act", "appear", "cast"], "release_year": ["year", "release", "when"],
    "in_language": ["language"], "has_genre": ["genre", "type of", "types of", "kind"],
    "has_tags": ["tag", "about", "topic", "describ"], "has_imdb_rating": ["rating", "rated"],
    "has_imdb_votes": ["vote", "popular", "famous"],
}


def rule_plans(question_masked: str, topic_kind: str) -> list[list[str]]:
    """Offline planner (no LLM): keyword-matched relations, all valid 2-3 step orderings."""
    q = question_masked.lower()
    hits = [r for r in RELATIONS if any(k in q for k in KEYWORDS[r])] or ["starred_actors", "directed_by"]
    rels = set(hits)
    cands = []
    for n in (2, 3):
        for seq in itertools.product(sorted(rels), repeat=n):
            at_movie = topic_kind == "movie"
            plan = []
            for r in seq:
                plan.append(FWD[r] if at_movie else INV[r]); at_movie = not at_movie
            if valid_plan(plan, topic_kind) and plan not in cands:
                cands.append(plan)
    # prefer plans whose last relation is the last keyword mentioned
    last = max(hits, key=lambda r: max(q.rfind(k) for k in KEYWORDS[r]))
    cands.sort(key=lambda p: (not p[-1].endswith(last.upper()), len(p)))
    return cands[:6]


# ---------------------------------------------------------------------- execution
@dataclass
class Result:
    plan: list[str]
    ranked: list[tuple[int, int]] = field(default_factory=list)   # (entity id, support) best first
    gaps: dict[int, list[int]] = field(default_factory=dict)       # level -> node ids missing next relation
    frontier: dict[int, list[int]] = field(default_factory=dict)
    ms: float = 0.0
    queries: int = 0


def chain(plan: list[str], upto: int, ret: str) -> str:
    parts = ["(n0:Entity {id: $t})"]
    for i, rel in enumerate(plan[:upto], 1):
        parts.append(f"-[:{rel}]->(n{i}:Entity)")
    # Paths must not loop back through the topic ("films sharing an actor with X" excludes X itself).
    inner = [f"n{i}.id <> $t" for i in range(1, upto) if i % 2 == 0]
    where = f" WHERE {' AND '.join(inner)}" if inner else ""
    return f"MATCH {''.join(parts)}{where} RETURN n{ret}.id AS x, count(*) AS c"


def execute(db: Hydra, topic_id: int, plan: list[str], *, probe_gaps: bool = False,
            gap_cap: int = 60) -> Result:
    t0, q0 = time.perf_counter(), db.stats["queries"]
    res = Result(plan=plan)
    rows = db.run(chain(plan, len(plan), str(len(plan))), {"t": topic_id})
    res.ranked = sorted(((r["x"], r["c"]) for r in rows if r["x"] != topic_id), key=lambda x: (-x[1], x[0]))
    if probe_gaps:
        res.frontier[0] = [topic_id]
        for lvl in range(1, len(plan)):
            nodes = [r["x"] for r in db.run(chain(plan, lvl, str(lvl)), {"t": topic_id})]
            res.frontier[lvl] = nodes
            if len(nodes) > gap_cap:
                continue
            cont = {r["x"] for r in db.run(chain(plan, lvl + 1, str(lvl)), {"t": topic_id})}
            missing = [n for n in nodes if n not in cont]
            if missing:
                res.gaps[lvl] = missing
        # topic itself missing the first relation entirely
        if not res.frontier.get(1) and len(plan) >= 1:
            res.gaps[0] = [topic_id]
    res.ms = (time.perf_counter() - t0) * 1000
    res.queries = db.stats["queries"] - q0
    return res


def f1(pred: set, gold: set) -> float:
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if not tp:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


_norm = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    return _norm.sub(" ", s.lower()).strip()
