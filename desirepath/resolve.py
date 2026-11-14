"""Entity resolution for text-extracted names: link, merge aliases, or create new entities.

Order: exact name -> case-insensitive -> normalised (accents, punctuation, Jr./Sr.) -> alias
(same first + last name, unique among people) -> new person node (src: text).
Every merge and new node is a logged graph edit.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii)\b")
_PUNCT = re.compile(r"[^a-z0-9 ]+")


def normalise(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = _PUNCT.sub(" ", s.replace("'", ""))
    s = _SUFFIX.sub(" ", s)
    return " ".join(s.split())


def first_last(name: str) -> str | None:
    toks = normalise(name).split()
    toks = [t for t in toks if len(t) > 1]          # drop middle initials
    return f"{toks[0]} {toks[-1]}" if len(toks) >= 2 else None


GENRE_SYNONYMS = {"science fiction": "Sci-Fi", "sci fi": "Sci-Fi", "scifi": "Sci-Fi", "animated": "Animation",
                  "biographical": "Biography", "biopic": "Biography", "sports": "Sport", "historical": "History",
                  "romantic comedy": "Comedy", "suspense": "Thriller", "psychological thriller": "Thriller",
                  "war": "History", "musical comedy": "Musical", "children": "Family", "kids": "Family",
                  "slasher": "Horror", "detective": "Mystery", "noir": "Crime", "film noir": "Crime",
                  "superhero": "Action", "martial arts": "Action"}

WANT = {"directed_by": "person", "written_by": "person", "starred_actors": "person",
        "release_year": "year", "in_language": "language", "has_genre": "genre"}
NEW_ENTITY_ID = 5_000_000


class Resolver:
    def __init__(self, graph, editlog=None, create_people: bool = True):
        self.g = graph
        self.log = editlog
        self.create_people = create_people
        self.norm: dict[tuple[str, str], list[int]] = defaultdict(list)    # (kind, norm) -> ids
        self.alias: dict[str, list[int]] = defaultdict(list)               # first+last -> person ids
        for name, i in graph.ent_id.items():
            self._index(name, i, graph.kind.get(name, ""))
        self.next_id = NEW_ENTITY_ID
        self.stats = {"exact": 0, "normalised": 0, "alias_merge": 0, "new_entity": 0, "dropped": 0}

    def _index(self, name: str, i: int, kind: str):
        n = normalise(name)
        self.norm[(kind, n)].append(i)
        squashed = n.replace(" ", "")
        if squashed != n:
            self.norm[(kind, squashed)].append(i)
        if kind == "person":
            fl = first_last(name)
            if fl:
                self.alias[fl].append(i)

    def link(self, value: str, rel: str) -> int | None:
        kind = WANT[rel]
        g = self.g
        for cand in (g.ent_id.get(value), g.lower_id.get(value.lower())):
            if cand is not None and g.kind.get(g.name[cand]) == kind:
                self.stats["exact"] += 1
                return cand
        n = normalise(value)
        if kind == "genre" and n in GENRE_SYNONYMS:
            n = normalise(GENRE_SYNONYMS[n])
        for key in (n, n.replace(" ", "")):
            hits = list(dict.fromkeys(self.norm.get((kind, key), [])))
            if len(hits) == 1:
                self.stats["normalised"] += 1
                return hits[0]
        if kind == "person":
            fl = first_last(value)
            hits = self.alias.get(fl, []) if fl else []
            if len(hits) == 1:
                self.stats["alias_merge"] += 1
                if self.log is not None:
                    self.log.write("merge", alias=value, into=g.name[hits[0]])
                return hits[0]
            if self.create_people and fl and len(value.split()) <= 5:
                return self._create(value, kind)
        self.stats["dropped"] += 1
        return None

    def _create(self, name: str, kind: str) -> int:
        i = self.next_id
        self.next_id += 1
        g = self.g
        g.db.batch("UNWIND $rows AS row MERGE (n {id: row.id}) "
                   "SET n:Entity, n.name = row.name, n.kind = row.kind, n.healed = row.h, n.src = row.src",
                   [{"id": i, "name": name, "kind": kind, "h": 0, "src": "text"}])
        g.ent_id[name] = i
        g.name[i] = name
        g.kind[name] = kind
        g.lower_id[name.lower()] = i
        self._index(name, i, kind)
        self.stats["new_entity"] += 1
        if self.log is not None:
            self.log.write("new_entity", name=name, entity_kind=kind)
        return i
