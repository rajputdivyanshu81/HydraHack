"""Load the (damaged) MetaQA graph into HydraDB and write graph edits.

Schema in HydraDB
  (:Entity {id, name, kind, healed})                     ids 1..N
  (:Entity)-[:DIRECTED_BY {id, cost, src}]->(:Entity)     movie -> value   (one type per relation)
  (:Entity)-[:R_DIRECTED_BY {id, cost, src}]->(:Entity)   value -> movie   (inverse)
  (:Template {id, text})-[:USES {id, cost, wins, tries}]->(:Plan {id, rels})
"""
from __future__ import annotations

import json
import os
import time

from .data import FWD, INV, OUT
from .db import Hydra

EDGE_ID_KB = 10_000_000
EDGE_ID_TEXT = 20_000_000
EDGE_ID_USES = 30_000_000
TEMPLATE_ID = 1_000_000
PLAN_ID = 2_000_000


def load_prepared():
    ents = json.load(open(os.path.join(OUT, "entities.json")))
    kb = json.load(open(os.path.join(OUT, "kb.json")))
    return ents, kb


class Graph:
    """Graph state in HydraDB plus small local mirrors used only for id allocation and dedupe."""

    def __init__(self, db: Hydra, ents: dict):
        self.db = db
        self.ent_id: dict[str, int] = ents["ent_id"]
        self.kind: dict[str, str] = ents["kind"]
        self.name = {v: k for k, v in self.ent_id.items()}
        self.lower_id = {k.lower(): v for k, v in self.ent_id.items()}
        self.triples: set[tuple[int, str, int]] = set()   # (movie_id, relation, value_id)
        self.next_text_edge = EDGE_ID_TEXT
        self.next_uses_edge = EDGE_ID_USES
        self.on_write = []   # callbacks(touched node ids) after edges are written
        self.written = []    # (movie_id, relation, value_id) triples written since creation (for demos/exports)

    def mirror(self, triples):
        """Record already-loaded triples locally (dedupe only) without writing."""
        self.triples.update((self.ent_id[s], r, self.ent_id[o]) for s, r, o in triples)

    # ------------------------------------------------------------ bulk load
    def load(self, triples, src: str = "kb", log=print):
        t0 = time.time()
        rows = [{"id": i, "name": n, "kind": self.kind.get(n, "other"), "healed": 0}
                for n, i in self.ent_id.items()]
        self.db.batch("UNWIND $rows AS row MERGE (n {id: row.id}) "
                      "SET n:Entity, n.name = row.name, n.kind = row.kind, n.healed = row.healed", rows)
        log(f"  nodes: {len(rows)} in {time.time() - t0:.1f}s")
        t1 = time.time()
        n = self.write_triples([(self.ent_id[s], r, self.ent_id[o]) for s, r, o in triples], src=src,
                               first_edge_id=EDGE_ID_KB)
        log(f"  edges: {n} directed edges in {time.time() - t1:.1f}s")

    def write_triples(self, triples, src: str, first_edge_id: int | None = None) -> int:
        """Write (movie, relation, value) triples as forward + inverse typed edges. Skips known triples."""
        new = [t for t in dict.fromkeys(triples) if t not in self.triples]
        if not new:
            return 0
        eid = first_edge_id if first_edge_id is not None else self.next_text_edge
        cost = 1.0 if src == "kb" else 1.0
        by_type: dict[str, list[dict]] = {}
        for m, r, v in new:
            by_type.setdefault(FWD[r], []).append({"eid": eid, "s": m, "d": v, "c": cost, "src": src}); eid += 1
            by_type.setdefault(INV[r], []).append({"eid": eid, "s": v, "d": m, "c": cost, "src": src}); eid += 1
        for etype, rows in by_type.items():
            self.db.batch(f"UNWIND $rows AS row MATCH (a:Entity {{id: row.s}}), (b:Entity {{id: row.d}}) "
                          f"CREATE (a)-[r:{etype} {{id: row.eid, cost: row.c, src: row.src}}]->(b)", rows)
        if first_edge_id is None:
            self.next_text_edge = eid
        self.triples.update(new)
        self.written.extend(new)
        touched = {m for m, _, _ in new} | {v for _, _, v in new}
        for hook in self.on_write:
            hook(touched)
        return 2 * len(new)

    def mark_healed(self, ids: list[int]):
        if ids:
            self.db.batch("UNWIND $rows AS row MATCH (n:Entity {id: row.id}) SET n.healed = row.h",
                          [{"id": i, "h": 1} for i in ids])

    # ------------------------------------------------------------ inspection
    def counts(self) -> dict:
        out = {}
        out["entities"] = self.db.run("MATCH (n:Entity) RETURN count(*) AS n")[0]["n"]
        out["text_edges"] = self.db.run(
            "MATCH (a:Entity)-[r:STARRED_ACTORS]->(b:Entity) WHERE r.src = 'text' RETURN count(*) AS n")[0]["n"]
        return out
