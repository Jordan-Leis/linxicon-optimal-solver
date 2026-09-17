"""Fetch puzzle starters from the server-rendered daily game page."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import re

import requests

from .server import BASE_URL, USER_AGENT

DAILY_PATH = '/play/daily'


@dataclass(frozen=True)
class Game:
    id: int
    tl: str
    br: str
    similarity: float
    date: str


def parse_game_page(html: str, *, today: date | None = None) -> Game:
    # Since 2026-09-17 the page hydrates `gameId:948,...,starters:$R[n]={tl,br,similarity}`
    # and no longer states the puzzle date. The puzzle rolls over at 00:00 UTC, so
    # the UTC calendar date at fetch time is the puzzle date.
    ident = re.search(r'\bgameId:\s*(\d+)', html)
    starters = re.search(
        r'starters:\s*(?:\$R\[\d+\]\s*=\s*)?\{\s*tl:\s*"([a-z]+)",\s*br:\s*"([a-z]+)",'
        r'\s*similarity:\s*([-+\d.eE]+)\s*\}', html,
    )
    if not ident or not starters:
        raise ValueError("Could not find puzzle starters; the game page may have changed.")
    try:
        similarity = float(starters[3])
    except ValueError as exc:
        raise ValueError("Invalid puzzle data in game page.") from exc
    when = today or datetime.now(timezone.utc).date()
    return Game(int(ident[1]), starters[1], starters[2], similarity, when.isoformat())


def _fetch_page(path: str) -> str:
    try:
        response = requests.get(BASE_URL + path, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"Could not fetch puzzle: {exc}") from exc
    return response.text


def fetch_game(game_id: int | None = None) -> Game:
    """Fetch today's puzzle. Past puzzles are no longer served by ID: /game/<id> redirects here."""
    if game_id is not None and game_id <= 0:
        raise ValueError("Game ID must be positive.")
    game = parse_game_page(_fetch_page(DAILY_PATH))
    if game_id is not None and game_id != game.id:
        raise ValueError(f"Linxicon serves only today's puzzle (#{game.id}); past puzzles cannot be fetched by ID.")
    return game
