"""Small, rate-limited adapter for the game's SolidStart server functions."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import time

import requests

BASE_URL = "https://linxicon.com"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
SERVER_IDS = {
    "update-semantics": "e9d3b0d5dd4481260a5fcd7bcea33c42ae11b3bd9c026f68d8c95330ee2a8d28",
    "validate-word": "a43d97d5cc6cd123dc9f8181142447dbb75900eaa64aa1d860b980dc5bb4c81e",
}


class ServerError(RuntimeError):
    """Network or protocol failure; verification has not succeeded."""


class MissingVectorError(ServerError):
    """The server returned no vector scores for a requested word."""


class WordRejectedError(ServerError):
    """A word failed the game's dictionary validation."""


def serialize_args(args: list) -> str:
    next_id = 0
    def encode(value):
        nonlocal next_id
        if isinstance(value, str):
            return {"t": 1, "s": value}
        if isinstance(value, list):
            ref = next_id
            next_id += 1
            return {"t": 9, "i": ref, "a": [encode(v) for v in value], "o": 0}
        raise TypeError("Server arguments support only arrays and strings.")
    return json.dumps({"t": encode(args), "f": 127, "m": []}, separators=(",", ":"))


def _check_envelope(text: str) -> None:
    if not re.match(r'^;0x[\da-fA-F]+;', text) or '"server-fn:0"' not in text:
        raise ServerError("Unrecognised server response; the site protocol may have changed.")


def parse_semantics_response(text: str) -> dict[frozenset[str], float]:
    _check_envelope(text)
    pattern = r'from:"node-([a-z]+)",to:"node-([a-z]+)",color:"[^"\\]*",score:([-+\d.eE]+)'
    pairs = {}
    for a, b, value in re.findall(pattern, text):
        try:
            score = float(value)
        except ValueError as exc:
            raise ServerError("Invalid numeric score in server response.") from exc
        if not math.isfinite(score):
            raise ServerError("Non-finite score in server response.")
        pairs[frozenset((a, b))] = score
    if pairs and len(re.findall(r'from:', text)) == len(re.findall(pattern, text)):
        return pairs
    if not pairs and re.search(r'\(\$R=>\$R\[\d+\]=\[\]\)\(\$R\["server-fn:0"\]\)\)\s*$', text):
        return {}
    raise ServerError("Unrecognised semantics response; the endpoint or response format may have changed.")


def parse_validate_response(text: str) -> str | None:
    _check_envelope(text)
    match = re.search(r',\s*(void 0|"(?:[^"\\]|\\.)*")\s*\)\s*$', text)
    if not match:
        raise ServerError("Unrecognised validation response; the site protocol may have changed.")
    if match[1] == "void 0":
        return None
    try:
        return json.loads(match[1])
    except ValueError as exc:
        raise ServerError("Malformed validation error string.") from exc


@dataclass
class Verification:
    words: list[str]
    scores: list[float]
    pairs: dict[frozenset[str], float]

    def score(self, a: str, b: str) -> float:
        return self.pairs[frozenset((a, b))]


class ServerClient:
    def __init__(self, session=None):
        self.session = session if session is not None else requests.Session()
        self._last_call: float | None = None
        self._validated: dict[str, str | None] = {}

    def _call(self, function: str, args: list) -> str:
        if self._last_call is not None:
            delay = 0.5 - (time.monotonic() - self._last_call)
            if delay > 0:
                time.sleep(delay)
        try:
            response = self.session.post(
                BASE_URL + "/_server", data=serialize_args(args), timeout=30,
                headers={"X-Server-Id": SERVER_IDS[function], "X-Server-Instance": "server-fn:0",
                         "Content-Type": "application/json", "User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            raise ServerError(f"{function} request failed: {exc}") from exc
        finally:
            self._last_call = time.monotonic()

    def update_semantics(self, board: list[str], word: str) -> dict[frozenset[str], float]:
        return parse_semantics_response(self._call("update-semantics", [board, word]))

    def validate_word(self, word: str) -> str | None:
        if word not in self._validated:
            self._validated[word] = parse_validate_response(self._call("validate-word", [word]))
        return self._validated[word]

    def verify(self, chain_words: list[str]) -> Verification:
        if len(chain_words) < 2 or len(set(chain_words)) != len(chain_words):
            raise ValueError("A chain needs two distinct starters and no repeated words.")
        for word in chain_words:
            error = self.validate_word(word)
            if error is not None:
                raise WordRejectedError(f'{word}: {error}')
        pairs: dict[frozenset[str], float] = {}
        board = [chain_words[0]]
        for word in [chain_words[-1], *chain_words[1:-1]]:
            new = self.update_semantics(board, word)
            if not new:
                raise MissingVectorError(f'No server vector scores for "{word}" against {", ".join(board)}.')
            for previous in board:
                if frozenset((previous, word)) not in new:
                    raise ServerError(f'Server response is missing the score for {previous} / {word}.')
            pairs.update(new)
            board.append(word)
        scores = [pairs[frozenset((a, b))] for a, b in zip(chain_words, chain_words[1:])]
        return Verification(list(chain_words), scores, pairs)


_default_client = ServerClient()


def update_semantics(board: list[str], word: str) -> dict[frozenset[str], float]:
    return _default_client.update_semantics(board, word)


def validate_word(word: str) -> str | None:
    return _default_client.validate_word(word)


def verify_chain(chain_words: list[str]) -> list[float]:
    return _default_client.verify(chain_words).scores
