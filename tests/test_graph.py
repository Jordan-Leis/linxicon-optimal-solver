import numpy as np
import pytest

from linxicon_solver.graph import Graph, Chain


def toy():
    # 0=tl 1=br; 2,3 are 1-hop bridges; 4-5 form a 2-hop bridge with higher scores
    words = ["tl", "br", "a", "b", "c", "d", "lonely"]
    edges = {
        (0, 2): 0.5, (2, 1): 0.5,      # tl-a-br  total 1.0
        (0, 3): 0.9, (3, 1): 0.7,      # tl-b-br  total 1.6
        (0, 4): 0.99, (4, 5): 0.99, (5, 1): 0.99,  # tl-c-d-br total 2.97 (longer)
    }
    return Graph(words, edges)


def test_shortest_chains_ranked_by_total_score_among_fewest_words():
    g = toy()
    chains = g.shortest_chains("tl", "br", k=5)
    assert [c.words for c in chains] == [["tl", "b", "br"], ["tl", "a", "br"]]
    assert chains[0].scores == pytest.approx([0.9, 0.7])
    assert chains[0].added == 1


def test_no_path_returns_empty():
    g = toy()
    assert g.shortest_chains("tl", "lonely") == []


def test_direct_link_is_zero_added_words():
    g = Graph(["x", "y"], {(0, 1): 0.8})
    chains = g.shortest_chains("x", "y")
    assert chains[0].words == ["x", "y"] and chains[0].added == 0


def test_k_limits_results_and_keeps_best():
    words = ["tl", "br"] + [f"m{i}" for i in range(10)]
    edges = {}
    for i in range(10):
        edges[(0, 2 + i)] = 0.4 + i * 0.05
        edges[(2 + i, 1)] = 0.4 + i * 0.05
    g = Graph(words, edges)
    chains = g.shortest_chains("tl", "br", k=3)
    assert [c.words[1] for c in chains] == ["m9", "m8", "m7"]


def test_build_from_vectors_uses_threshold_and_boosts():
    from linxicon_solver.similarity import Vectors, Similarity

    m = np.array([[1, 0, 0], [0, 1, 0], [0.8, 0.6, 0]], dtype=np.float32)  # chest, setting, box
    v = Vectors(["chest", "setting", "box"], m)
    sim = Similarity(v, boosts={frozenset(("chest", "setting")): 1.0}, use_wordnet=False)
    g = Graph.build(sim, threshold=0.65)
    assert g.weight("box", "chest") == pytest.approx(0.8)
    assert g.weight("box", "setting") is None  # cos 0.6 < threshold
    assert g.weight("chest", "setting") == 1.0  # cos 0.0 but boosted -> edge


def test_graph_save_and_load_roundtrip(tmp_path):
    g = toy()
    p = tmp_path / "g.npz"
    g.save(p)
    g2 = Graph.load(p)
    assert g2.words == g.words
    assert g2.weight("tl", "b") == pytest.approx(0.9)
    assert [c.words for c in g2.shortest_chains("tl", "br", k=2)] == [c.words for c in g.shortest_chains("tl", "br", k=2)]
