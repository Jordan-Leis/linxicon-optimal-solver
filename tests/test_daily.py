from pathlib import Path

from linxicon_solver.daily import parse_game_page

SAVED = Path(__file__).resolve().parent.parent / "source-from-web" / "Game #942.html"


def test_parse_saved_game_page():
    game = parse_game_page(SAVED.read_text(encoding="utf-8", errors="ignore"))
    assert game.id == 942
    assert game.tl == "chest"
    assert game.br == "setting"
    assert game.similarity == 0.0044
    assert game.date == "2026-09-11"

import pytest
import requests
from linxicon_solver import daily


@pytest.mark.parametrize('reference', ['', '$R[987]='])
def test_parse_reference_variations(reference):
    game = parse_game_page('starters:' + reference + '{id:12,tl:"apple",br:"banana",similarity:4.2e-3,date:"2026-09-12"}')
    assert game.id == 12 and game.similarity == 0.0042


def test_missing_game_has_clear_error():
    with pytest.raises(ValueError, match='puzzle'):
        parse_game_page('<html>Not found</html>')


@pytest.mark.parametrize(('game_id', 'suffix'), [(None, '/game'), (942, '/game/942?enterGame=')])
def test_fetch_game_urls(monkeypatch, game_id, suffix):
    calls = []
    class Response:
        text = SAVED.read_text()
        def raise_for_status(self): pass
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()
    monkeypatch.setattr(daily.requests, 'get', get)
    assert daily.fetch_game(game_id).id == 942
    assert calls[0][0] == 'https://linxicon.com' + suffix
    assert calls[0][1]['timeout'] == 30


def test_fetch_failure(monkeypatch):
    def get(*args, **kwargs):
        raise requests.Timeout('timed out')
    monkeypatch.setattr(daily.requests, 'get', get)
    with pytest.raises(ValueError, match='fetch puzzle'):
        daily.fetch_game()


def test_today_follows_homepage_play_form_after_soft_404(monkeypatch):
    pages = {
        'https://linxicon.com/game': '<h1>404 - Page not found</h1>',
        'https://linxicon.com/': '<form action="/game/943"><button name="enterGame">Play</button></form>',
        'https://linxicon.com/game/943?enterGame=': 'starters:$R[7]={id:943,tl:"apple",br:"banana",similarity:0.1,date:"2026-09-12"}',
    }
    visited = []
    class Response:
        def __init__(self, text): self.text = text
        def raise_for_status(self): pass
    def get(url, **kwargs):
        visited.append(url)
        return Response(pages[url])
    monkeypatch.setattr(daily.requests, 'get', get)
    assert daily.fetch_game().id == 943
    assert visited == list(pages)


def test_today_missing_play_form_is_clear(monkeypatch):
    class Response:
        text = '<h1>404 - Page not found</h1>'
        def raise_for_status(self): pass
    monkeypatch.setattr(daily.requests, 'get', lambda *a, **kw: Response())
    with pytest.raises(ValueError, match="today's puzzle"):
        daily.fetch_game()
