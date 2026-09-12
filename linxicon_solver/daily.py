"""Fetch puzzle starters from server-rendered game pages."""
from __future__ import annotations

from dataclasses import dataclass
import re

import requests

from .server import BASE_URL, USER_AGENT


@dataclass(frozen=True)
class Game:
    id: int
    tl: str
    br: str
    similarity: float
    date: str


def parse_game_page(html: str) -> Game:
    match = re.search(
        r'starters:\s*(?:\$R\[\d+\]\s*=\s*)?\{\s*id:\s*(\d+),\s*tl:\s*"([a-z]+)",'
        r'\s*br:\s*"([a-z]+)",\s*similarity:\s*([-+\d.eE]+),\s*date:\s*"(\d{4}-\d{2}-\d{2})"\s*\}', html,
    )
    if not match:
        raise ValueError("Could not find puzzle starters; the game page may have changed.")
    try:
        return Game(int(match[1]), match[2], match[3], float(match[4]), match[5])
    except ValueError as exc:
        raise ValueError("Invalid puzzle data in game page.") from exc


def fetch_game(game_id: int | None = None) -> Game:
    if game_id is not None and game_id <= 0:
        raise ValueError("Game ID must be positive.")
    url = BASE_URL + ("/game" if game_id is None else f"/game/{game_id}?enterGame=")
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"Could not fetch puzzle: {exc}") from exc
    return parse_game_page(response.text)
