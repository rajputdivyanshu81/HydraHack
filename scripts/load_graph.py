"""Load the damaged (50%) or full MetaQA graph into a fresh HydraDB.

    python scripts/load_graph.py            # damaged graph (kept triples)
    python scripts/load_graph.py --full     # full KB (upper bound)
"""
import argparse
import sys
import time

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
from desirepath.db import Hydra  # noqa: E402
from desirepath.graph import Graph, load_prepared  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true")
args = ap.parse_args()

db = Hydra()
db.wait_ready()
ents, kb = load_prepared()
g = Graph(db, ents)
t0 = time.time()
g.load(kb["full"] if args.full else kb["kept"])
print(f"loaded in {time.time() - t0:.1f}s; stats {db.stats}")
print(db.run("MATCH (n:Entity) RETURN count(*) AS n"))
print(db.run("MATCH (a:Entity)-[r:DIRECTED_BY]->(b:Entity) RETURN count(*) AS n"))
