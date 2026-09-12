"""Candidate vocabulary: ENABLE word list filtered by the game's word rules and
a word-frequency floor (so we propose words the game's dictionary is likely to
accept and a human would recognise)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from .rules import MAX_WORD_LEN, MIN_WORD_LEN

ENABLE_URL = "https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt"
DEFAULT_MIN_ZIPF = 2.0


def is_valid_word(word: str) -> bool:
    """Mirror linxicon's validate-word rules: 3-15 lowercase ASCII letters."""
    return MIN_WORD_LEN <= len(word) <= MAX_WORD_LEN and word.isascii() and word.isalpha() and word.islower()


def filter_words(words: Iterable[str], min_zipf: float, zipf: Callable[[str], float]) -> list[str]:
    """Keep valid words with zipf frequency >= min_zipf; dedupe and sort."""
    return sorted({w for w in words if is_valid_word(w) and zipf(w) >= min_zipf})


def load_enable(path: Path) -> list[str]:
    if not path.exists():
        import requests

        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(ENABLE_URL, timeout=60)
        r.raise_for_status()
        path.write_text(r.text)
    return path.read_text().split()


def build_vocab(enable_path: Path, min_zipf: float = DEFAULT_MIN_ZIPF) -> list[str]:
    from wordfreq import zipf_frequency

    return filter_words(load_enable(enable_path), min_zipf, lambda w: zipf_frequency(w, "en"))
