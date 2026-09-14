"""Link graph over the vocabulary and k-best shortest-chain search.

A chain's rank mirrors the game's own ordering (network-*.js `C`/`he`):
fewest words first, then highest total link score.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from .rules import THRESHOLD
from .similarity import Similarity


@dataclass
class Chain:
    words: list[str]
    scores: list[float]

    @property
    def added(self) -> int:
        return len(self.words) - 2

    @property
    def total(self) -> float:
        return float(sum(self.scores))

    @property
    def average(self) -> float:
        return self.total / len(self.scores) if self.scores else 0.0

    @property
    def weakest(self) -> float:
        return min(self.scores) if self.scores else 0.0


class Graph:
    def __init__(self, words: list[str], edges: dict[tuple[int, int], float]):
        self.words = words
        self._index = {w: i for i, w in enumerate(words)}
        self.adj: list[dict[int, float]] = [dict() for _ in words]
        for (a, b), s in edges.items():
            self.adj[a][b] = s
            self.adj[b][a] = s

    def __contains__(self, word: str) -> bool:
        return word in self._index

    def weight(self, a: str, b: str) -> float | None:
        return self.adj[self._index[a]].get(self._index[b])

    def save(self, path: Path) -> None:
        a, b, w = [], [], []
        for i, nbrs in enumerate(self.adj):
            for j, s in nbrs.items():
                if i < j:
                    a.append(i); b.append(j); w.append(s)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, words=np.array(self.words), a=np.array(a, dtype=np.int32),
                            b=np.array(b, dtype=np.int32), w=np.array(w, dtype=np.float32))

    @classmethod
    def load(cls, path: Path) -> "Graph":
        z = np.load(path)
        edges = {(int(i), int(j)): float(s) for i, j, s in zip(z["a"], z["b"], z["w"])}
        return cls(z["words"].tolist(), edges)

    @classmethod
    def build(cls, sim: Similarity, threshold: float = THRESHOLD, chunk: int = 1024, progress=None) -> "Graph":
        """Edges = cosine >= threshold (chunked matmul) plus WordNet boosts >= threshold."""
        v = sim.vectors
        n = len(v.words)
        edges: dict[tuple[int, int], float] = {}
        for start in range(0, n, chunk):
            block = v.matrix[start : start + chunk] @ v.matrix.T
            rows, cols = np.nonzero(block >= threshold)
            for r, c in zip(rows.tolist(), cols.tolist()):
                a = start + r
                if a < c:
                    edges[(a, c)] = float(block[r, c])
            if progress:
                progress(min(start + chunk, n), n)
        for wa, wb, boost in sim.boost_pairs(v.words):
            a, b = v.index(wa), v.index(wb)
            if a == b or boost < threshold:
                continue
            key = (a, b) if a < b else (b, a)
            if edges.get(key, 0.0) < boost:
                edges[key] = boost
        return cls(list(v.words), edges)

    def shortest_chains(self, src: str, dst: str, k: int = 5, observer=None, blocked=frozenset()) -> list[Chain]:
        """k best chains among those with the fewest words (ties by total score).

        `blocked` words (for example ones the game's dictionary rejected) are never
        entered; the starters themselves are exempt.
        """
        if src not in self._index or dst not in self._index:
            return []
        s, t = self._index[src], self._index[dst]
        skip = {self._index[w] for w in blocked if w in self._index} - {s, t}
        if s == t:
            return [Chain([src], [])]
        # BFS layering from src
        dist = {s: 0}
        order = [s]
        q = deque([s])
        examined = 0
        if observer:
            observer(dict(type='discover', word=src, parent=None, depth=0, visited=1, frontier=1, examined=0))
        while q:
            u = q.popleft()
            if u == t:
                break
            for vtx in sorted(self.adj[u], key=lambda i: self.words[i]):
                examined += 1
                if vtx not in dist and vtx not in skip:
                    dist[vtx] = dist[u] + 1
                    order.append(vtx)
                    q.append(vtx)
                    if observer:
                        observer(dict(type='discover', word=self.words[vtx], parent=self.words[u],
                                      depth=dist[vtx], visited=len(dist), frontier=len(q), examined=examined))
        if t not in dist:
            if observer:
                observer(dict(type='complete', visited=len(dist), frontier=len(q), examined=examined, candidates=[]))
            return []
        # k-best DP over the layered DAG (only edges that advance one layer)
        best: dict[int, list[tuple[float, list[int]]]] = {s: [(0.0, [s])]}
        for u in order:
            if u == s or dist[u] > dist[t]:
                continue
            cands: list[tuple[float, list[int]]] = []
            for p, w in self.adj[u].items():
                if dist.get(p) == dist[u] - 1 and p in best:
                    cands.extend((score + w, path + [u]) for score, path in best[p])
            cands.sort(key=lambda x: (-x[0], tuple(self.words[i] for i in x[1])))
            best[u] = cands[:k]
        chains = []
        for _, path in best.get(t, []):
            scores = [self.adj[path[i]][path[i + 1]] for i in range(len(path) - 1)]
            chains.append(Chain([self.words[i] for i in path], scores))
        if observer:
            observer(dict(type='complete', visited=len(dist), frontier=len(q), examined=examined,
                          candidates=[c.words for c in chains]))
        return chains
