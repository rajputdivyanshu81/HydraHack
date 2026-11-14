"""Fill the extraction cache for every wiki passage in parallel. Pure speed-up: running experiments then hit
the cache instead of waiting on the API; results are identical (same prompt -> same cached response)."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])
from desirepath import llm  # noqa: E402
from desirepath.heal import Corpus, extract_llm  # noqa: E402

workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
c = Corpus()
docs = c.docs
t0 = time.time()
done = errors = 0
with ThreadPoolExecutor(max_workers=workers) as pool:
    futs = [pool.submit(extract_llm, d["title"], d["text"]) for d in docs]
    for f in as_completed(futs):
        try:
            f.result()
        except Exception as e:  # noqa: BLE001
            errors += 1
            if errors <= 3:
                print("error:", str(e)[:150], flush=True)
        done += 1
        if done % 1000 == 0:
            print(f"{done}/{len(docs)} in {time.time() - t0:.0f}s  {llm.USAGE.snapshot()}  errors {errors}", flush=True)
print("finished", llm.USAGE.snapshot(), "errors", errors, flush=True)
