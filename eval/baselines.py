"""Baselines on the same held-out questions.

  vector  - vector store RAG: bge-small embeddings over all 18k WikiMovies passages, top-k -> Claude answers
  llm     - Claude with no retrieval (parametric memory)

    python -m eval.baselines --which vector llm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from desirepath import llm  # noqa: E402
from desirepath.answer import f1, norm  # noqa: E402
from desirepath.data import OUT  # noqa: E402
from eval.run_curve import summarize  # noqa: E402

EMB_PATH = os.path.join(ROOT, "results", "cache", "wiki_bge_small.npy")
EMB_MODEL = "BAAI/bge-small-en-v1.5"

ANSWER_SYSTEM = ("Answer questions about movies. Return every answer you can support as a list of short names "
                 "(people, titles, years, languages or genres), most likely first. No explanations.")
ANSWER_SCHEMA = llm.obj({"answers": {"type": "array", "items": {"type": "string"}}})


class VectorStore:
    def __init__(self, docs: list[dict]):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMB_MODEL)
        self.model.max_seq_length = 256
        self.docs = docs
        if os.path.exists(EMB_PATH):
            self.emb = np.load(EMB_PATH)
        else:
            texts = [f"{d['title']}. {d['text']}" for d in docs]
            t0 = time.time()
            self.emb = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
            os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
            np.save(EMB_PATH, self.emb)
            print(f"embedded {len(texts)} passages in {time.time() - t0:.0f}s")

    def search(self, query: str, k: int) -> list[int]:
        q = self.model.encode(["Represent this sentence for searching relevant passages: " + query],
                              normalize_embeddings=True)[0]
        return list(np.argsort(-(self.emb @ q))[:k])


def score(q: dict, answers: list[str]) -> dict:
    gold = {norm(a) for a in q["answers"]}
    preds = [norm(a) for a in answers if a.strip()]
    return {"hit": int(bool(preds) and preds[0] in gold), "f1": f1(set(preds), gold)}


def run(which: str, heldout: list[dict], k: int):
    rows = []
    store = None
    if which == "vector":
        docs = json.load(open(os.path.join(OUT, "wiki.json")))["docs"]
        store = VectorStore(docs)
    llm.USAGE.reset()
    for q in heldout:
        t0 = time.perf_counter()
        tok0 = llm.USAGE.input_tokens + llm.USAGE.output_tokens
        if which == "vector":
            ctx = "\n\n".join(f"[{store.docs[i]['title']}] {store.docs[i]['text']}" for i in store.search(q["question"], k))
            prompt = f"Passages:\n{ctx}\n\nQuestion: {q['question']}"
        else:
            prompt = f"Question: {q['question']}"
        out = llm.call_json(ANSWER_SYSTEM, prompt, ANSWER_SCHEMA, max_tokens=1500, tag=which)
        s = score(q, out.get("answers", []))
        rows.append({"qid": q["qid"], "hop": q["hop"], "split": q["split"], **s, "pred": (out.get("answers") or [None])[0],
                     "ms": (time.perf_counter() - t0) * 1000, "queries": 0, "plan_source": which,
                     "tokens": llm.USAGE.input_tokens + llm.USAGE.output_tokens - tok0})
    s = summarize(rows)
    print(f"[{which}] hits@1 {s['hits1']:.3f} f1 {s['f1']:.3f} tokens/q {s['tokens']:.0f} usage {llm.USAGE.snapshot()}")
    json.dump({"variant": which, "heldout": s, "k": k, "usage": llm.USAGE.snapshot()},
              open(os.path.join(ROOT, "results", f"baseline_{which}.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(ROOT, "results", f"baseline_{which}.rows.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["vector", "llm"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embed-only", action="store_true")
    args = ap.parse_args()
    if args.embed_only:
        VectorStore(json.load(open(os.path.join(OUT, "wiki.json")))["docs"])
        return
    heldout = json.load(open(os.path.join(OUT, "workload.json")))["heldout"]
    for w in args.which:
        run(w, heldout, args.k)


if __name__ == "__main__":
    main()
