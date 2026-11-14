"""Fill the LLM cache for the cold-start planner on every (wording, topic kind) in the workload, in parallel.
Pure speed-up: the planner is a deterministic cached function of the wording, so results are unchanged."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
from desirepath import llm  # noqa: E402
from desirepath.answer import llm_plans  # noqa: E402
from desirepath.data import OUT  # noqa: E402

ents = json.load(open(os.path.join(OUT, "entities.json")))
wl = json.load(open(os.path.join(OUT, "workload.json")))
keys = sorted({(q["template"], ents["kind"].get(q["topic"], "movie")) for q in wl["stream"] + wl["heldout"]})
print(f"{len(keys)} distinct (wording, kind) pairs")
errors = 0


def go(k):
    global errors
    try:
        llm_plans(*k)
    except Exception as e:  # noqa: BLE001
        errors += 1
        print("error", k, str(e)[:120])


with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(go, keys))
print("done", llm.USAGE.snapshot(), "errors", errors)
