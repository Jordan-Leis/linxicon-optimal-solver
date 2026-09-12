"""Shared data preparation and cache naming for setup and the solver."""
from __future__ import annotations

from pathlib import Path

from .graph import Graph
from .rules import THRESHOLD
from .similarity import SCORING_VERSION, Similarity, Vectors
from .vocab import DEFAULT_MIN_ZIPF, build_vocab

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
NUMBERBATCH_NAME = 'numberbatch-en-19.08.txt.gz'


def graph_cache_path(cache_dir: Path, vector_key: str, threshold: float, use_wordnet: bool) -> Path:
    return cache_dir / f'graph-{vector_key}-{threshold}-wn{int(use_wordnet)}-{SCORING_VERSION}.npz'


def ensure_wordnet() -> None:
    import nltk.data
    for resource in ('corpora/wordnet', 'corpora/wordnet.zip'):
        try:
            nltk.data.find(resource)
            return
        except LookupError:
            pass
    raise ValueError('WordNet is unavailable. Run python scripts/setup_data.py or use --no-wordnet.')


def prepare(starters=(), *, data_dir: Path = DATA_DIR, min_zipf: float = DEFAULT_MIN_ZIPF,
            use_wordnet: bool = True, progress=None) -> tuple[Similarity, Graph]:
    enable, numberbatch = data_dir/'enable1.txt', data_dir/NUMBERBATCH_NAME
    for path in (enable, numberbatch):
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path.name}. Run python scripts/setup_data.py first.')
    if use_wordnet:
        ensure_wordnet()
    vocabulary = set(build_vocab(enable, min_zipf=min_zipf)) | set(starters)
    if progress:
        progress('Loading vectors…')
    vectors = Vectors.load(numberbatch, vocabulary, data_dir/'cache')
    for starter in starters:
        if starter not in vectors:
            raise ValueError(f'Starter "{starter}" has no Numberbatch vector.')
    sim = Similarity(vectors, use_wordnet=use_wordnet)
    cache = graph_cache_path(data_dir/'cache', vectors.cache_key, THRESHOLD, use_wordnet)
    if cache.exists():
        if progress:
            progress('Loading cached graph…')
        graph = Graph.load(cache)
    else:
        if progress:
            progress(f'Building graph over {len(vectors.words):,} words; this may take a minute…')
        graph = Graph.build(sim)
        # Only a complete archive becomes visible at the final cache path.
        temporary = cache.with_suffix('.tmp.npz')
        try:
            graph.save(temporary)
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
    return sim, graph
