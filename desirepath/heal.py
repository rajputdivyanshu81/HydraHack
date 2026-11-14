"""Gap-driven ingestion: read text about the entities where traversal got stuck, extract the missing
edges, and write them into HydraDB.

Extractors
  llm   - Claude Haiku 5.5, relations restricted to the KB vocabulary (structured output)
  rules - regex patterns over the wiki sentence forms (no LLM; offline mode and a cheap baseline)
"""
from __future__ import annotations

import json
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import llm
from .data import OUT
from .graph import Graph
from .resolve import Resolver

TEXT_RELATIONS = ["directed_by", "written_by", "starred_actors", "release_year", "in_language", "has_genre"]


class Corpus:
    def __init__(self):
        w = json.load(open(os.path.join(OUT, "wiki.json")))
        self.docs = w["docs"]
        self.title_to_docs: dict[str, list[int]] = w["title_to_docs"]
        self.lower_text = [d["text"].lower() for d in self.docs]
        self._mention_cache: dict[str, list[int]] = {}

    def docs_for(self, name: str, kind: str, cap: int) -> list[int]:
        """A movie's own article; for other entities, the articles that mention them."""
        if kind == "movie":
            return self.title_to_docs.get(name, [])[:cap]
        if name not in self._mention_cache:
            needle = name.lower()
            if len(needle) < 4 or kind in ("tag", "rating", "votes"):
                self._mention_cache[name] = []
            else:
                pat = re.compile(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])")
                self._mention_cache[name] = [i for i, t in enumerate(self.lower_text) if needle in t and pat.search(t)]
        return self._mention_cache[name][:cap]


# ---------------------------------------------------------------------- extractors
EXTRACT_SYSTEM = ("You extract facts about one film from its encyclopedia text into a fixed schema. "
                  "Only state facts the text says. Use full names as written in the text.")
EXTRACT_SCHEMA = llm.obj({
    "directed_by": {"type": "array", "items": {"type": "string"}},
    "written_by": {"type": "array", "items": {"type": "string"}},
    "starred_actors": {"type": "array", "items": {"type": "string"}},
    "release_year": {"type": "array", "items": {"type": "string"}},
    "in_language": {"type": "array", "items": {"type": "string"}},
    "has_genre": {"type": "array", "items": {"type": "string"}},
})


def extract_llm(title: str, text: str) -> dict[str, list[str]]:
    prompt = (f"Film: {title}\nText: {text}\n\nFill each field with values from the text (empty list if not "
              "stated): directors, writers (screenplay/story/novel adapted), actors, release year (4 digits), "
              "spoken languages (e.g. English, French), genres (single words like Drama, Comedy, Thriller, "
              "Horror, Romance, Documentary, Animation, Western, Musical, War, Crime, Mystery, Action, "
              "Adventure, Fantasy, Sci-Fi, Family, Biography, Sport, History).")
    return llm.call_json(EXTRACT_SYSTEM, prompt, EXTRACT_SCHEMA, max_tokens=1500, tag="extract")


_NAME = r"[A-Z][\w.'\-]*(?:\s+(?:[A-Z][\w.'\-]*|de|van|von|da|del|la|le))*"
_LIST = rf"({_NAME}(?:\s*,\s*{_NAME})*(?:,?\s+and\s+{_NAME})?)"
RULES = {
    "directed_by": [rf"directed by {_LIST}", rf"written and directed by {_LIST}", rf"director {_NAME}"],
    "written_by": [rf"written by {_LIST}", rf"screenplay by {_LIST}", rf"written and directed by {_LIST}",
                   rf"based on (?:the )?(?:\w+ )*?(?:novel|book|play|story) (?:\w+ )*?by {_LIST}"],
    "starred_actors": [rf"\bstar(?:s|ring|red)? {_LIST}", rf"\bwith {_LIST} in (?:the )?(?:lead|main|title)"],
}
LANGS = ["English", "French", "German", "Italian", "Spanish", "Japanese", "Hindi", "Russian", "Korean",
         "Swedish", "Danish", "Mandarin", "Cantonese", "Portuguese", "Polish", "Dutch", "Turkish", "Greek",
         "Tamil", "Telugu", "Persian", "Norwegian", "Finnish", "Czech", "Hungarian", "Hebrew", "Arabic", "Thai"]
GENRES = {"comedy": "Comedy", "drama": "Drama", "thriller": "Thriller", "horror": "Horror", "romantic": "Romance",
          "romance": "Romance", "documentary": "Documentary", "animated": "Animation", "western": "Western",
          "musical": "Musical", "war film": "War", "crime": "Crime", "mystery": "Mystery", "action": "Action",
          "adventure": "Adventure", "fantasy": "Fantasy", "science fiction": "Sci-Fi", "family": "Family",
          "biographical": "Biography", "sports": "Sport", "historical": "History"}


def _split(names: str) -> list[str]:
    return [n.strip() for n in re.split(r",\s*|\s+and\s+", names) if n.strip()]


def extract_rules(title: str, text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for rel, pats in RULES.items():
        for p in pats:
            for m in re.finditer(p, text):
                out[rel] += _split(m.group(1)) if m.groups() else []
    m = re.search(r"\b(?:is|was) an? (\d{4})\b", text)
    if m:
        out["release_year"].append(m.group(1))
    head = text[:300]
    for lang in LANGS:
        if re.search(rf"\b{lang}(?:-language)?\b", head):
            out["in_language"].append(lang)
    low = head.lower()
    for k, g in GENRES.items():
        if k in low:
            out["has_genre"].append(g)
    return dict(out)


# ---------------------------------------------------------------------- healer
class Healer:
    def __init__(self, graph: Graph, corpus: Corpus, extractor: str = "llm", editlog=None,
                 docs_per_entity: int = 8, docs_per_question: int = 40, resolve: bool = True):
        self.g = graph
        self.c = corpus
        self.extract = extract_llm if extractor == "llm" else extract_rules
        self.extractor = extractor
        self.log = editlog
        self.read_docs: set[int] = set()
        self.healed_entities: set[int] = set()
        self.docs_per_entity = docs_per_entity
        self.docs_per_question = docs_per_question
        self.stats = {"docs_read": 0, "edges_added": 0, "unlinked": 0, "triples_found": 0}
        self.resolver = Resolver(graph, editlog, create_people=resolve)

    def link(self, value: str, rel: str) -> int | None:
        with self._lock:   # resolver mutates indexes and may create nodes
            return self.resolver.link(value, rel)

    def read_doc(self, doc_id: int) -> list[tuple[int, str, int]]:
        d = self.c.docs[doc_id]
        movie = self.g.ent_id.get(d["title"])
        if movie is None or self.g.kind.get(d["title"]) != "movie":
            return []
        try:
            facts = self.extract(d["title"], d["text"])
        except Exception as e:  # noqa: BLE001  (a failed extraction must not kill a run)
            self._count("extract_errors")
            if self.log is not None:
                self.log.write("extract_error", doc=d["title"], error=str(e)[:200])
            return []
        triples = []
        for rel in TEXT_RELATIONS:
            for v in facts.get(rel, []) or []:
                self._count("triples_found")
                vid = self.link(v.strip(), rel)
                if vid is None:
                    self._count("unlinked")
                    continue
                triples.append((movie, rel, vid))
        return triples

    _lock = threading.Lock()

    def _count(self, key: str):
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + 1

    def heal(self, entity_ids: list[int], cause: str, budget: int | None = None) -> int:
        """Read text about these entities (once each), write the missing edges. Returns edges added."""
        budget = budget or self.docs_per_question
        todo: list[int] = []
        touched = []
        for e in entity_ids:
            if e in self.healed_entities:
                continue
            name = self.g.name[e]
            unread = [d for d in self.c.docs_for(name, self.g.kind.get(name, ""), 10_000)
                      if d not in self.read_docs and d not in todo]
            todo += unread[: self.docs_per_entity]
            if len(unread) <= self.docs_per_entity:   # fully read; a hub entity stays open for later gaps
                touched.append(e)
            if len(todo) >= budget:
                break
        todo = todo[:budget]
        triples = []
        if self.extractor == "llm" and len(todo) > 1:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for t in pool.map(self.read_doc, todo):
                    triples += t
        else:
            for d in todo:
                triples += self.read_doc(d)
        self.read_docs.update(todo)
        self.stats["docs_read"] += len(todo)
        added = self.g.write_triples(triples, src="text")
        self.stats["edges_added"] += added
        self.healed_entities.update(touched)
        self.g.mark_healed(touched)
        if self.log is not None and (added or todo):
            self.log.write("ingest", cause=cause, entities=[self.g.name[e] for e in touched][:8],
                           docs=len(todo), edges_added=added)
        return added
