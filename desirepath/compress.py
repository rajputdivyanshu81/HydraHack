"""Path compression: materialise hot multi-hop traversals as direct SHORTCUT edges in HydraDB.

When the same (topic, plan) keeps producing confirmed answers, its k-hop result is written as
  (topic)-[:SHORTCUT {id, plan, n}]->(answer)
so later questions read one hop instead of k. A shortcut can only go stale if a new edge touches a
node the original traversal reached at levels 0..k-1, so we keep that node set and delete the
shortcut the moment healing writes an edge touching it (exact invalidation).
"""
from __future__ import annotations

from collections import defaultdict

from .answer import chain
from .db import Hydra

SHORTCUT_EDGE_ID = 40_000_000
HOT_AFTER = 2          # confirmed wins before a (topic, plan) is compressed
MAX_ANSWERS = 200
MAX_PATH_NODES = 3000  # do not compress hub traversals


class Compressor:
    def __init__(self, db: Hydra, editlog=None):
        self.db = db
        self.log = editlog
        self.wins: dict[tuple[int, str], int] = defaultdict(int)
        self.live: dict[tuple[int, str], set[int]] = {}          # (topic, plan) -> nodes on its paths
        self.by_node: dict[int, set[tuple[int, str]]] = defaultdict(set)
        self.next_id = SHORTCUT_EDGE_ID
        self.stats = {"compressed": 0, "invalidated": 0, "edges": 0}

    def has(self, topic: int, plan: list[str]) -> bool:
        return (topic, ">".join(plan)) in self.live

    def read(self, topic: int, plan: list[str]) -> list[tuple[int, int]]:
        rows = self.db.run("MATCH (a:Entity {id: $t})-[s:SHORTCUT]->(x:Entity) WHERE s.plan = $p "
                           "RETURN x.id AS x, s.n AS c", {"t": topic, "p": ">".join(plan)})
        return sorted(((r["x"], r["c"]) for r in rows), key=lambda x: (-x[1], x[0]))

    def confirm(self, topic: int, plan: list[str]):
        """Called when feedback confirms this plan answered correctly for this topic."""
        if len(plan) < 2:
            return
        key = (topic, ">".join(plan))
        self.wins[key] += 1
        if self.wins[key] >= HOT_AFTER and key not in self.live:
            self._compress(topic, plan, key)

    def _compress(self, topic: int, plan: list[str], key):
        nodes = {topic}
        for lvl in range(1, len(plan)):
            nodes |= {r["x"] for r in self.db.run(chain(plan, lvl, str(lvl)), {"t": topic})}
            if len(nodes) > MAX_PATH_NODES:
                return
        rows = self.db.run(chain(plan, len(plan), str(len(plan))), {"t": topic})
        answers = sorted(((r["x"], r["c"]) for r in rows if r["x"] != topic), key=lambda x: (-x[1], x[0]))
        answers = answers[:MAX_ANSWERS]
        if not answers:
            return
        edge_rows = []
        for x, c in answers:
            edge_rows.append({"eid": self.next_id, "s": topic, "d": x, "p": key[1], "n": c})
            self.next_id += 1
        self.db.batch("UNWIND $rows AS row MATCH (a:Entity {id: row.s}), (b:Entity {id: row.d}) "
                      "CREATE (a)-[r:SHORTCUT {id: row.eid, plan: row.p, n: row.n}]->(b)", edge_rows)
        self.live[key] = nodes
        for n in nodes:
            self.by_node[n].add(key)
        self.stats["compressed"] += 1
        self.stats["edges"] += len(edge_rows)
        if self.log is not None:
            self.log.write("compress", topic=topic, plan=key[1], hops=len(plan), shortcut_edges=len(edge_rows),
                           path_nodes=len(nodes))

    def invalidate(self, touched: set[int]):
        stale = set()
        for n in touched:
            stale |= self.by_node.get(n, set())
        for key in stale:
            if key not in self.live:
                continue
            topic, plan = key
            self.db.write("MATCH (a:Entity {id: $t})-[s:SHORTCUT]->(x:Entity) WHERE s.plan = $p DELETE s",
                          {"t": topic, "p": plan})
            for n in self.live.pop(key):
                self.by_node[n].discard(key)
            self.wins[key] = 0
            self.stats["invalidated"] += 1
            if self.log is not None:
                self.log.write("decompress", topic=topic, plan=plan, cause="healing touched a node on its paths")
