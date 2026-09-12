"""Word similarity as linxicon's server computes it.

Reverse-engineered (2026-09-12) by comparing ~1,000 pair scores from the
`update-semantics` server function against candidate models:

* base score  = max(0, cosine(ConceptNet Numberbatch 19.08 vectors))
                exact match on ~96% of pairs
* WordNet boost overrides the base when higher:
    - words sharing a synset (any POS)                 -> 1.0
    - direct hypernym/hyponym (nouns, verbs)           -> 0.6
    - adjective similar_to                             -> 0.6
  (verified 100% on ~75 pairs where cosine alone was below threshold)
* the server occasionally returns other boosts (0.4 for some WordNet
  siblings, 0.3+0.4*cos for some others) that we cannot predict; they only
  ever *raise* a score, so local scores are a safe lower bound.
"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

NUMBERBATCH_URL = "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz"
SYNONYM_BOOST = 1.0
HYPERNYM_BOOST = 0.6


class Vectors:
    """Normalised Numberbatch vectors restricted to a vocabulary."""

    def __init__(self, words: list[str], matrix: np.ndarray):
        self.words = words
        self.matrix = matrix.astype(np.float32)
        self._index = {w: i for i, w in enumerate(words)}

    def __contains__(self, word: str) -> bool:
        return word in self._index

    def index(self, word: str) -> int:
        return self._index[word]

    def cosine(self, a: str, b: str) -> float:
        if a not in self._index or b not in self._index:
            return 0.0
        return float(self.matrix[self._index[a]] @ self.matrix[self._index[b]])

    def cosines_to_all(self, word: str) -> np.ndarray:
        return self.matrix @ self.matrix[self._index[word]]

    @classmethod
    def load(cls, numberbatch_gz: Path, vocab: Iterable[str], cache_dir: Path) -> "Vectors":
        vocab = set(vocab)
        key = hashlib.sha1(("\n".join(sorted(vocab)) + numberbatch_gz.name).encode()).hexdigest()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        words_file, npy_file = cache_dir / f"{key}.words", cache_dir / f"{key}.npy"
        if words_file.exists() and npy_file.exists():
            return cls(words_file.read_text().split("\n"), np.load(npy_file))
        words: list[str] = []
        rows: list[np.ndarray] = []
        with gzip.open(numberbatch_gz, "rt", encoding="utf-8") as f:
            next(f)  # header: "<count> <dim>"
            for line in f:
                w, _, rest = line.partition(" ")
                if w in vocab:
                    v = np.array(rest.split(), dtype=np.float32)
                    words.append(w)
                    rows.append(v / np.linalg.norm(v))
        matrix = np.vstack(rows) if rows else np.zeros((0, 300), dtype=np.float32)
        words_file.write_text("\n".join(words))
        np.save(npy_file, matrix)
        return cls(words, matrix)


# -- WordNet boosts -----------------------------------------------------------

def _lemmas(synset) -> set[str]:
    return {l.name().lower() for l in synset.lemmas() if l.name().isalpha()}


def wordnet_related(word: str) -> dict[str, float]:
    """All words the server boosts relative to `word`, with the boost value."""
    from nltk.corpus import wordnet as wn

    out: dict[str, float] = {}
    for s in wn.synsets(word):
        for w in _lemmas(s):
            out[w] = SYNONYM_BOOST
        related = s.hypernyms() + s.hyponyms() + s.instance_hypernyms() + s.instance_hyponyms() + s.similar_tos()
        for t in related:
            for w in _lemmas(t):
                out.setdefault(w, HYPERNYM_BOOST)
    out.pop(word, None)
    return out


def wordnet_boost(a: str, b: str) -> float:
    return wordnet_related(a).get(b, 0.0)


class Similarity:
    """score(a, b) as the server would report it (lower bound)."""

    def __init__(self, vectors: Vectors, boosts: dict[frozenset, float] | None = None, use_wordnet: bool = True):
        self.vectors = vectors
        self.use_wordnet = use_wordnet
        self._boosts: dict[frozenset, float] = dict(boosts or {})

    def boost(self, a: str, b: str) -> float:
        key = frozenset((a, b))
        if key not in self._boosts:
            self._boosts[key] = wordnet_boost(a, b) if self.use_wordnet else 0.0
        return self._boosts[key]

    def score(self, a: str, b: str) -> float:
        return max(0.0, self.vectors.cosine(a, b), self.boost(a, b))

    def boost_pairs(self, words: Iterable[str]):
        """Yield (a, b, boost) for every known boost among `words`."""
        vocab = set(words)
        for key, boost in self._boosts.items():
            if len(key) == 2 and boost > 0 and key <= vocab:
                a, b = sorted(key)
                yield a, b, boost
        if self.use_wordnet:
            for w in vocab:
                for other, boost in wordnet_related(w).items():
                    if other in vocab:
                        yield w, other, boost
