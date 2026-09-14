from linxicon_solver.vocab import is_valid_word, filter_words


def test_rejects_short_and_long_words():
    assert not is_valid_word("ab")
    assert is_valid_word("abc")
    assert is_valid_word("a" * 15)
    assert not is_valid_word("a" * 16)


def test_rejects_symbols_capitals_and_non_ascii():
    assert not is_valid_word("don't")
    assert not is_valid_word("ice-cream")
    assert not is_valid_word("Paris")
    assert not is_valid_word("café")
    assert is_valid_word("chest")


def test_filter_words_applies_validity_and_min_zipf():
    words = ["chest", "Paris", "ab", "zyzzyva", "setting"]
    zipf = {"chest": 4.5, "paris": 5.0, "ab": 3.0, "zyzzyva": 0.5, "setting": 4.8}
    out = filter_words(words, min_zipf=3.0, zipf=zipf.get)
    assert out == ["chest", "setting"]


def test_filter_words_dedupes_and_sorts():
    out = filter_words(["setting", "chest", "chest"], min_zipf=0, zipf=lambda w: 5.0)
    assert out == ["chest", "setting"]


def test_load_rejected_reads_words_and_ignores_comments(tmp_path):
    from linxicon_solver.vocab import load_rejected
    path = tmp_path / "rejected_words.txt"
    path.write_text("# words the game's dictionary rejected\nfete\n\nFetes \nbahai # seen 2026-09-12\n")
    assert load_rejected(path) == {"fete", "fetes", "bahai"}
    assert load_rejected(tmp_path / "missing.txt") == set()


def test_shipped_rejected_words_include_known_rejections():
    from linxicon_solver.data import DATA_DIR
    from linxicon_solver.vocab import load_rejected
    assert {"fete", "fetes", "bahai", "connexion"} <= load_rejected(DATA_DIR / "rejected_words.txt")
