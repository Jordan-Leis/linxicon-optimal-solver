"""Game constants, reverse-engineered from linxicon.com's client bundle.

Names in comments are the minified identifiers in `_build/assets/datetime-*.js`
so they can be re-verified if the site updates.
"""

THRESHOLD = 0.3995  # `gs` — a link forms when score >= this (UI rounds to "40%")
MAX_LINKS = 5  # `Ms` — each node keeps only its 5 strongest links
MAX_WORDS = 50  # `Ts` — board cap, including the two starter words
CLUE_AT = 25  # `Os` — clue button appears at this many words
MIN_WORD_LEN = 3
MAX_WORD_LEN = 15
