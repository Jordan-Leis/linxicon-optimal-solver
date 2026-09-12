"""Board simulator tests: a faithful port of linxicon's client-side rules.

Source of truth is the live JS bundle (network-*.js fn `Ee` for pruning,
fn `C`/`he` for shortest path; datetime-*.js for constants).
"""
import pytest

from linxicon_solver.board import Board
from linxicon_solver.rules import THRESHOLD, MAX_LINKS, MAX_WORDS


def make_sim(table):
    """Build a symmetric similarity function from a {frozenset({a,b}): score} table."""
    def sim(a, b):
        return table.get(frozenset((a, b)), 0.0)
    return sim


def test_edge_forms_at_threshold_and_not_below():
    sim = make_sim({frozenset(("tl", "x")): THRESHOLD, frozenset(("br", "x")): THRESHOLD - 1e-6})
    board = Board(("tl", "br"), sim)
    board.add_word("x")
    assert board.has_edge("tl", "x")
    assert not board.has_edge("br", "x")


def test_win_when_starters_connected_through_chain():
    sim = make_sim({frozenset(("tl", "a")): 0.5, frozenset(("a", "b")): 0.5, frozenset(("b", "br")): 0.5})
    board = Board(("tl", "br"), sim)
    assert not board.is_won()
    board.add_word("a")
    board.add_word("b")
    assert board.is_won()


def test_edge_outside_both_endpoints_top5_is_pruned():
    # hub has 6 candidate neighbours; the weakest ("weak") is not in hub's top-5,
    # and hub is not in weak's top-5 either only if weak has 5 stronger links.
    table = {}
    strong = [f"s{i}" for i in range(5)]
    for i, s in enumerate(strong):
        table[frozenset(("hub", s))] = 0.9 - i * 0.01
    table[frozenset(("hub", "weak"))] = 0.5
    # give "weak" five stronger links so hub is not in weak's top-5 either
    for i, s in enumerate(strong):
        table[frozenset(("weak", s))] = 0.8 - i * 0.01
    board = Board(("tl", "br"), make_sim(table))
    for w in ["hub", *strong, "weak"]:
        board.add_word(w)
    assert not board.has_edge("hub", "weak")
    for s in strong:
        assert board.has_edge("hub", s)


def test_edge_survives_if_in_either_endpoints_top5():
    # hub has 6 links; "weak" is hub's 6th, but hub is weak's only link -> survives
    table = {}
    strong = [f"s{i}" for i in range(5)]
    for i, s in enumerate(strong):
        table[frozenset(("hub", s))] = 0.9 - i * 0.01
    table[frozenset(("hub", "weak"))] = 0.5
    board = Board(("tl", "br"), make_sim(table))
    for w in ["hub", *strong, "weak"]:
        board.add_word(w)
    assert board.has_edge("hub", "weak")
    assert len(board.edges) == 6


def test_max_links_constant_is_five():
    assert MAX_LINKS == 5
    assert MAX_WORDS == 50
    assert THRESHOLD == 0.3995


def test_shortest_path_prefers_fewest_nodes_then_highest_score():
    table = {
        frozenset(("tl", "a")): 0.5, frozenset(("a", "br")): 0.5,       # 2-hop path, avg 0.5
        frozenset(("tl", "b")): 0.9, frozenset(("b", "br")): 0.9,       # 2-hop path, avg 0.9
        frozenset(("tl", "c")): 0.99, frozenset(("c", "d")): 0.99, frozenset(("d", "br")): 0.99,  # 3-hop, higher avg
    }
    board = Board(("tl", "br"), make_sim(table))
    for w in "abcd":
        board.add_word(w)
    assert board.shortest_path() == ["tl", "b", "br"]


def test_simulate_reports_win_and_path():
    sim = make_sim({frozenset(("tl", "a")): 0.5, frozenset(("a", "br")): 0.5})
    board = Board(("tl", "br"), sim)
    result = board.simulate(["a"])
    assert result.won is True
    assert result.path == ["tl", "a", "br"]
    assert result.words_added == 1


def test_simulate_stops_at_max_words():
    board = Board(("tl", "br"), make_sim({}))
    result = board.simulate([f"w{i}" for i in range(MAX_WORDS)])
    assert result.won is False
    assert result.words_added == MAX_WORDS - 2


def test_duplicate_word_rejected():
    board = Board(("tl", "br"), make_sim({}))
    board.add_word("x")
    with pytest.raises(ValueError):
        board.add_word("x")
