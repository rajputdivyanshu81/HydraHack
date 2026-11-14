"""MetaQA + WikiMovies preparation: entity ids, 50% KB deletion, skewed workload, text index.

Run:  python -m desirepath.data      (writes data/prepared/*.json)
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "prepared")

# MetaQA relations (subject is always a movie). Forward type and its inverse.
RELATIONS = ["directed_by", "written_by", "starred_actors", "release_year", "in_language",
             "has_tags", "has_genre", "has_imdb_rating", "has_imdb_votes"]
FWD = {r: r.upper() for r in RELATIONS}            # movie -> value
INV = {r: "R_" + r.upper() for r in RELATIONS}     # value -> movie
ALL_TYPES = list(FWD.values()) + list(INV.values())
# Relations whose values can plausibly be recovered from wiki text.
TEXT_RECOVERABLE = {"directed_by", "written_by", "starred_actors", "release_year", "in_language", "has_genre"}

VALUE_KIND = {"directed_by": "person", "written_by": "person", "starred_actors": "person",
              "release_year": "year", "in_language": "language", "has_tags": "tag",
              "has_genre": "genre", "has_imdb_rating": "rating", "has_imdb_votes": "votes"}

DROP_RATE = 0.5
SEED = 0


def load_kb():
    triples = []
    with open(os.path.join(RAW, "metaqa", "kb_kb.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) == 3 and parts[1] in FWD:
                triples.append(tuple(parts))
    return sorted(set(triples))


def load_questions(hop: int, split: str = "test"):
    path = os.path.join(RAW, "metaqa", f"{hop}-hop_vanilla_qa_{split}.txt")
    qs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            q, a = line.rstrip("\n").split("\t")
            m = re.search(r"\[(.+?)\]", q)
            qs.append({"qid": f"{hop}h-{split}-{i}", "hop": hop, "question": q.replace("[", "").replace("]", ""),
                       "topic": m.group(1), "template": q[:m.start()] + "NE" + q[m.end():],
                       "answers": a.split("|")})
    return qs


def load_wiki():
    """wiki.txt: blocks separated by blank lines; line 1 is '1 Title (film)'."""
    docs, cur = [], []
    with open(os.path.join(RAW, "movieqa", "knowledge_source", "wiki.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    docs.append(cur); cur = []
                continue
            cur.append(line.split(" ", 1)[1] if " " in line else line)
    if cur:
        docs.append(cur)
    out = []
    for d in docs:
        title = re.sub(r"\s*\((?:\d{4} )?(?:[A-Za-z ]+ )?film\)\s*$", "", d[0]).strip()
        out.append({"title": title, "raw_title": d[0], "text": " ".join(d[1:])})
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    triples = load_kb()

    # ---- entity ids (shared by every system variant)
    names = sorted({t[0] for t in triples} | {t[2] for t in triples})
    kind = {}
    for s, r, o in triples:
        kind[s] = "movie"
        kind.setdefault(o, VALUE_KIND[r])
    ent_id = {n: i + 1 for i, n in enumerate(names)}

    # ---- 50% deletion with a fixed seed
    kept, hidden = [], []
    for t in triples:
        (hidden if rng.random() < DROP_RATE else kept).append(t)

    # ---- questions and skewed workload
    # No model is trained, so train+test are just one question pool for the workload.
    qs = load_questions(2) + load_questions(3) + load_questions(2, "train") + load_questions(3, "train")
    qs = [q for q in qs if q["topic"] in ent_id]
    by_topic = defaultdict(list)
    for q in qs:
        by_topic[q["topic"]].append(q)
    topics = sorted(by_topic)
    rng.shuffle(topics)
    # Zipf popularity over topics (real workloads are skewed); stated in README.
    HOT = 600  # size of the frequently-asked topic set
    weights = {t: 1.0 / (rank + 1) ** 1.0 for rank, t in enumerate(topics)}
    unseen_pool = set(topics[HOT:])  # held-out "never seen" topics come only from here
    hot_topics = topics[:HOT]

    def zipf_sample(pool, n, exclude_q):
        w = [weights[t] for t in pool]
        picked, guard = [], 0
        while len(picked) < n and guard < n * 50:
            guard += 1
            t = rng.choices(pool, weights=w, k=1)[0]
            cand = [q for q in by_topic[t] if q["qid"] not in exclude_q]
            if not cand:
                continue
            q = rng.choice(cand)
            exclude_q.add(q["qid"]); picked.append(q)
        return picked

    used = set()
    stream = zipf_sample(hot_topics, 900, used)
    stream_topics = {q["topic"] for q in stream}
    held_seen = zipf_sample(sorted(stream_topics), 150, used)
    unseen = sorted(unseen_pool - stream_topics)
    held_unseen = []
    for t in rng.sample(unseen, 150):
        q = rng.choice(by_topic[t]); used.add(q["qid"]); held_unseen.append(q)
    for q in held_seen:
        q["split"] = "seen-topic"
    for q in held_unseen:
        q["split"] = "unseen-topic"
    heldout = held_seen + held_unseen

    # ---- wiki text: title index + mention index (for gap-driven ingestion)
    wiki = load_wiki()
    title_to_docs = defaultdict(list)
    for i, d in enumerate(wiki):
        title_to_docs[d["title"]].append(i)

    json.dump({"ent_id": ent_id, "kind": kind}, open(os.path.join(OUT, "entities.json"), "w"))
    json.dump({"kept": kept, "hidden": hidden, "full": triples}, open(os.path.join(OUT, "kb.json"), "w"))
    json.dump({"stream": stream, "heldout": heldout}, open(os.path.join(OUT, "workload.json"), "w"), indent=0)
    json.dump({"docs": wiki, "title_to_docs": title_to_docs}, open(os.path.join(OUT, "wiki.json"), "w"))

    print(f"entities {len(ent_id)}  triples {len(triples)}  kept {len(kept)}  hidden {len(hidden)}")
    print("relations kept:", Counter(r for _, r, _ in kept))
    print(f"stream {len(stream)} ({Counter(q['hop'] for q in stream)})  distinct topics {len(stream_topics)}")
    print(f"heldout {len(heldout)} seen-topic {len(held_seen)} unseen-topic {len(held_unseen)}  "
          f"hops {Counter(q['hop'] for q in heldout)}")
    print(f"templates in stream {len({q['template'] for q in stream})}; heldout templates covered by stream: "
          f"{sum(q['template'] in {s['template'] for s in stream} for q in heldout)}/{len(heldout)}")
    movies = {s for s, _, _ in triples}
    print(f"wiki docs {len(wiki)}; KB movies with a wiki doc: {sum(m in title_to_docs for m in movies)}/{len(movies)}")


if __name__ == "__main__":
    main()
