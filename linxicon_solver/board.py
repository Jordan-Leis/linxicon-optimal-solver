"""Board simulator — a faithful port of linxicon's client-side graph rules.

Ported from `_build/assets/network-*.js`:
  * `Ee`  -> :meth:`Board._prune`  (top-MAX_LINKS edge pruning)
  * `C`   -> :meth:`Board.shortest_path` (all shortest paths, best total score)
  * `he`  -> path score = sum of edge scores along the path
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .rules import MAX_LINKS, MAX_WORDS, THRESHOLD

SimFn = Callable[[str, str], float]


@dataclass(frozen=True)
class Edge:
    a: str
    b: str
    score: float

    @property
    def id(self) -> str:
        x, y = sorted((self.a, self.b))
        return f"{x}-{y}"

    def other(self, w: str) -> str:
        return self.b if w == self.a else self.a


@dataclass
class SimResult:
    won: bool
    path: list[str]
    words_added: int
    edges: list[Edge] = field(default_factory=list)


class Board:
    def __init__(self, starters: tuple[str, str], sim: SimFn):
        self.tl, self.br = starters
        self.sim = sim
        self.words: list[str] = [self.tl, self.br]
        # all candidate edges ever formed (score >= THRESHOLD), before pruning
        self._candidates: list[Edge] = []
        s = sim(self.tl, self.br)
        if s >= THRESHOLD:
            self._candidates.append(Edge(self.tl, self.br, s))
        self.edges: list[Edge] = self._prune(self._candidates)

    # -- mutation ---------------------------------------------------------
    def add_word(self, word: str) -> list[Edge]:
        """Add a word; returns the new candidate edges it formed."""
        if word in self.words:
            raise ValueError(f'"{word}" already added.')
        if len(self.words) >= MAX_WORDS:
            raise ValueError("Board is full.")
        new = [Edge(w, word, s) for w in self.words if (s := self.sim(w, word)) >= THRESHOLD]
        self.words.append(word)
        self._candidates.extend(new)
        self.edges = self._prune(self._candidates)
        return new

    @staticmethod
    def _prune(candidates: list[Edge]) -> list[Edge]:
        """Port of `Ee`: keep an edge iff it is among the top-MAX_LINKS edges
        (by score, desc) of *either* endpoint."""
        eligible = [e for e in candidates if e.score >= THRESHOLD]
        nodes: list[str] = []
        for e in eligible:
            for n in (e.a, e.b):
                if n not in nodes:
                    nodes.append(n)
        keep: list[Edge] = []
        seen: set[str] = set()
        for n in nodes:
            incident = [e for e in eligible if n in (e.a, e.b)]
            incident.sort(key=lambda e: e.score, reverse=True)
            for e in incident[:MAX_LINKS]:
                if e.id not in seen:
                    seen.add(e.id)
                    keep.append(e)
        return keep

    # -- queries ----------------------------------------------------------
    def has_edge(self, a: str, b: str) -> bool:
        eid = Edge(a, b, 0).id
        return any(e.id == eid for e in self.edges)

    def _adjacency(self) -> dict[str, list[str]]:
        adj: dict[str, list[str]] = {}
        for e in self.edges:
            adj.setdefault(e.a, []).append(e.b)
            adj.setdefault(e.b, []).append(e.a)
        return adj

    def is_won(self) -> bool:
        return len(self.shortest_path()) > 0

    def path_score(self, path: list[str]) -> float:
        """Port of `he`: sum of edge scores along the path."""
        ids = {Edge(path[i], path[i + 1], 0).id for i in range(len(path) - 1)}
        return sum(e.score for e in self.edges if e.id in ids)

    def shortest_path(self) -> list[str]:
        """Port of `C`: among all shortest tl->br paths, the one with the highest
        total score (ties -> last found, matching the JS `>=` reduce). Empty
        list if the starters are not connected."""
        adj = self._adjacency()
        if self.tl not in adj or self.br not in adj:
            return []
        # BFS layering to get distances, then enumerate all shortest paths.
        dist = {self.tl: 0}
        q = deque([self.tl])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if self.br not in dist:
            return []
        paths: list[list[str]] = []

        def walk(u: str, acc: list[str]) -> None:
            if u == self.br:
                paths.append(list(acc))
                return
            for v in adj[u]:
                if dist.get(v) == dist[u] + 1:
                    acc.append(v)
                    walk(v, acc)
                    acc.pop()

        walk(self.tl, [self.tl])
        best: list[str] = []
        best_score = -1.0
        for p in paths:
            s = self.path_score(p)
            if s >= best_score:
                best, best_score = p, s
        return best

    def simulate(self, chain: list[str]) -> SimResult:
        """Add words in order (stopping at a win or the board cap)."""
        added = 0
        for w in chain:
            if len(self.words) >= MAX_WORDS:
                break
            self.add_word(w)
            added += 1
            if self.is_won():
                break
        path = self.shortest_path()
        return SimResult(won=bool(path), path=path, words_added=added, edges=list(self.edges))
