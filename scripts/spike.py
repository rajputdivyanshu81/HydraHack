"""Spike: verify the HydraDB behaviours DesirePath depends on. Prints PASS/FAIL per check."""
import json
import sys
import traceback

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
from desirepath.db import Hydra  # noqa: E402

db = Hydra()
db.wait_ready()
results = {}


def check(name, fn):
    try:
        out = fn()
        if out == [] and name != "cleanup":
            raise AssertionError("empty result")
        results[name] = ("PASS", out)
        print(f"PASS {name}: {json.dumps(out, default=str)[:600]}")
    except Exception as e:  # noqa: BLE001
        results[name] = ("FAIL", str(e))
        print(f"FAIL {name}: {e}")
        traceback.print_exc(limit=1)


BASE = 9_000_000  # id range reserved for the spike


def setup():
    rows = [{"id": BASE + i, "name": n} for i, n in enumerate(["A", "B", "C", "D", "E"])]
    db.batch("UNWIND $rows AS row MERGE (n {id: row.id}) SET n:Spike, n.name = row.name", rows)
    # A->B->D (cheap), A->C->D (expensive), D->E
    edges = [(0, 1, 1.0), (1, 3, 1.0), (0, 2, 0.2), (2, 3, 0.2), (3, 4, 1.0)]
    db.batch("UNWIND $rows AS row MATCH (a:Spike {id: row.s}), (b:Spike {id: row.d}) "
             "CREATE (a)-[r:HOP {id: row.eid, cost: row.c}]->(b)",
             [{"eid": BASE + 100 + i, "s": BASE + s, "d": BASE + d, "c": c} for i, (s, d, c) in enumerate(edges)])
    return db.run("MATCH (a:Spike)-[r:HOP]->(b) RETURN a.id AS a, b.id AS b, r.cost AS c")


check("write+read nodes and edges via UNWIND", setup)
check("read by string property", lambda: db.run("MATCH (n:Spike {name: $x}) RETURN n.id AS id", {"x": "C"}))
check("SET relationship property", lambda: (db.write(
    "MATCH (a {id: $a})-[r:HOP]->(b {id: $b}) SET r.cost = 0.5", {"a": BASE, "b": BASE + 1}),
    db.run("MATCH (a {id: $a})-[r:HOP]->(b {id: $b}) RETURN r.cost AS c", {"a": BASE, "b": BASE + 1}))[1])
check("WHERE on relationship property", lambda: db.run(
    "MATCH (a:Spike)-[r:HOP]->(b) WHERE r.cost < 0.6 RETURN a.id AS a, b.id AS b, r.cost AS c"))
check("ORDER BY aliased rel property", lambda: db.run(
    "MATCH (a:Spike)-[r:HOP]->(b) RETURN b.id AS b, r.cost AS c ORDER BY c LIMIT 2"))
check("2-hop chain with costs", lambda: db.run(
    "MATCH (a {id: $a})-[r1:HOP]->(m)-[r2:HOP]->(b) RETURN m.id AS m, b.id AS b, r1.cost AS c1, r2.cost AS c2",
    {"a": BASE}))
check("bounded var-length", lambda: db.run(
    "MATCH (a {id: $a})-[:HOP*1..3]->(b) RETURN b.id AS b", {"a": BASE}))
check("SSpaths plain", lambda: db.run(
    f"CALL algo.SSpaths({{sourceNode: {BASE}, relTypes: ['HOP'], maxLen: 3}}) YIELD path RETURN path"))
check("SSpaths with costProp + yields", lambda: db.run(
    f"CALL algo.SSpaths({{sourceNode: {BASE}, relTypes: ['HOP'], maxLen: 3, costProp: 'cost', maxCost: 10}}) "
    "YIELD path, pathCost RETURN path, pathCost"))
check("SPpaths weightProp=cost (expect A-C-D)", lambda: db.run(
    f"CALL algo.SPpaths({{sourceNode: {BASE}, targetNode: {BASE + 3}, relTypes: ['HOP'], maxLen: 3, "
    "weightProp: 'cost', pathCount: 1}) YIELD path, pathWeight RETURN path, pathWeight"))
check("SPpaths all paths (pathCount 5)", lambda: db.run(
    f"CALL algo.SPpaths({{sourceNode: {BASE}, targetNode: {BASE + 3}, relTypes: ['HOP'], maxLen: 3, "
    "pathCount: 5}) YIELD path RETURN path"))
check("MSpaths by name", lambda: db.run(
    "CALL algo.MSpaths({sourceLabel: 'Spike', sourceProperty: 'name', sourceValues: ['A'], "
    "targetLabel: 'Spike', targetProperty: 'name', targetValues: ['E'], relTypes: ['HOP'], "
    "relDirection: 'outgoing', maxLen: 4, pathCount: 1, pairwise: false, resultLimit: 10}) YIELD path RETURN path"))
check("count aggregate", lambda: db.run("MATCH (n:Spike) RETURN count(*) AS n"))
check("collect aggregate", lambda: db.run(
    "MATCH (a {id: $a})-[:HOP]->(b) RETURN collect(b.id) AS bs", {"a": BASE}))
check("sum aggregate grouped", lambda: db.run(
    "MATCH (a:Spike)-[r:HOP]->(b) RETURN a.id AS a, sum(r.cost) AS s"))
check("incoming direction pattern", lambda: db.run(
    "MATCH (a {id: $d})<-[:HOP]-(b) RETURN b.id AS b", {"d": BASE + 3}))
check("cleanup", lambda: db.write("MATCH (n:Spike) DETACH DELETE n"))

print("\nSUMMARY:", {k: v[0] for k, v in results.items()})
print("stats", db.stats)
