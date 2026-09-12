import gzip
from pathlib import Path

import numpy as np
import pytest

from linxicon_solver.similarity import Vectors, Similarity, wordnet_boost, wordnet_related

REAL_NB = Path(__file__).resolve().parent.parent / "data" / "numberbatch-en-19.08.txt.gz"


@pytest.fixture
def tiny_numberbatch(tmp_path):
    p = tmp_path / "nb.txt.gz"
    with gzip.open(p, "wt") as f:
        f.write("4 3\n")
        f.write("chest 1 0 0\n")
        f.write("setting 0 1 0\n")
        f.write("box 0.8 0.6 0\n")  # cos(box,chest)=0.8, cos(box,setting)=0.6
        f.write("ignored 0 0 1\n")
    return p


def test_vectors_load_filters_to_vocab_and_normalises(tiny_numberbatch, tmp_path):
    v = Vectors.load(tiny_numberbatch, {"chest", "setting", "box", "missing"}, cache_dir=tmp_path / "cache")
    assert set(v.words) == {"chest", "setting", "box"}
    assert "ignored" not in v
    assert v.cosine("box", "chest") == pytest.approx(0.8)
    assert v.cosine("chest", "missing") == 0.0


def test_vectors_cache_roundtrip(tiny_numberbatch, tmp_path):
    cache = tmp_path / "cache"
    v1 = Vectors.load(tiny_numberbatch, {"chest", "box"}, cache_dir=cache)
    tiny_numberbatch.unlink()  # second load must come from cache
    v2 = Vectors.load(tiny_numberbatch, {"chest", "box"}, cache_dir=cache)
    assert v2.words == v1.words
    assert np.allclose(v2.matrix, v1.matrix)


def test_cosines_to_all_returns_vector_aligned_with_words(tiny_numberbatch, tmp_path):
    v = Vectors.load(tiny_numberbatch, {"chest", "setting", "box"}, cache_dir=tmp_path / "c")
    sims = v.cosines_to_all("chest")
    assert sims[v.index("box")] == pytest.approx(0.8)
    assert sims[v.index("setting")] == pytest.approx(0.0)


def test_wordnet_boost_tiers():
    assert wordnet_boost("doctor", "physician") == 1.0  # share a synset
    assert wordnet_boost("chest", "furniture") == 0.6  # direct hypernym
    assert wordnet_boost("jog", "run") == 0.6  # verb hypernym
    assert wordnet_boost("cold", "raw") == 0.6  # adjective similar_to
    assert wordnet_boost("apple", "banana") == 0.0  # siblings: not guaranteed
    assert wordnet_boost("chest", "setting") == 0.0


def test_wordnet_related_lists_boosted_neighbours():
    rel = wordnet_related("chest")
    assert rel["furniture"] == 0.6
    assert rel["thorax"] == 1.0
    assert "chest" not in rel


def test_similarity_score_is_max_of_clamped_cosine_and_boost(tiny_numberbatch, tmp_path):
    v = Vectors.load(tiny_numberbatch, {"chest", "setting", "box"}, cache_dir=tmp_path / "c")
    sim = Similarity(v, boosts={frozenset(("chest", "setting")): 0.6})
    assert sim.score("box", "chest") == pytest.approx(0.8)
    assert sim.score("chest", "setting") == 0.6


@pytest.mark.skipif(not REAL_NB.exists(), reason="real Numberbatch file not downloaded")
def test_calibration_against_live_server_values(tmp_path):
    """Values captured from linxicon's update-semantics endpoint on 2026-09-12."""
    v = Vectors.load(REAL_NB, {"chest", "setting", "keep", "torso", "box", "environment"}, cache_dir=tmp_path / "c")
    sim = Similarity(v)
    assert sim.score("chest", "setting") == pytest.approx(0.0044, abs=5e-4)
    # float32 vectors -> ~1e-5 drift from the server's float64 values
    assert sim.score("chest", "keep") == pytest.approx(0.060531583698313386, abs=1e-4)
    assert sim.score("chest", "torso") == pytest.approx(0.6139977040459459, abs=1e-4)
    assert sim.score("box", "chest") == 0.6  # WordNet hyponym boost overrides cos 0.43
    assert sim.score("environment", "setting") == 0.6
