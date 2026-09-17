from datetime import date
from pathlib import Path

import pytest
import requests

from linxicon_solver import daily
from linxicon_solver.daily import parse_game_page

# linxicon.com/play/daily as served on 2026-09-17: the puzzle id lives in a
# separate `gameId` field and the page carries no date.
SAVED = Path(__file__).resolve().parent.parent / "source-from-web" / "Play daily #948.html"
SNIPPET = ('$R[12]($R[6],$R[13]={gameId:948,gameType:"game",playerId:"abc",twlId:"",'
           'starters:$R[14]={tl:"calculate",br:"record",similarity:0.0797},words:$R[15]=[]})')


def test_parse_saved_daily_page():
    game = parse_game_page(SAVED.read_text(encoding="utf-8", errors="ignore"), today=date(2026, 9, 17))
    assert game.id == 948
    assert game.tl == "calculate"
    assert game.br == "record"
    assert game.similarity == 0.0797
    assert game.date == "2026-09-17"


@pytest.mark.parametrize('reference', ['', '$R[987]='])
def test_parse_reference_variations(reference):
    game = parse_game_page('gameId:12,starters:' + reference + '{tl:"apple",br:"banana",similarity:4.2e-3}',
                           today=date(2026, 9, 12))
    assert game.id == 12 and game.similarity == 0.0042 and game.date == "2026-09-12"


def test_date_defaults_to_utc_today(monkeypatch):
    from datetime import datetime
    class Clock:
        @staticmethod
        def now(tz):
            assert tz is daily.timezone.utc
            return datetime(2026, 9, 17, 0, 5, tzinfo=tz)
    monkeypatch.setattr(daily, 'datetime', Clock)
    assert parse_game_page(SNIPPET).date == "2026-09-17"


@pytest.mark.parametrize('html', ['<html>Not found</html>', 'gameId:948,words:[]',
                                  'starters:{tl:"apple",br:"banana",similarity:0.1}'])
def test_missing_game_has_clear_error(html):
    with pytest.raises(ValueError, match='puzzle'):
        parse_game_page(html)


@pytest.mark.parametrize('game_id', [None, 948])
def test_fetch_game_reads_the_daily_page(monkeypatch, game_id):
    calls = []
    class Response:
        text = SNIPPET
        def raise_for_status(self): pass
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()
    monkeypatch.setattr(daily.requests, 'get', get)
    assert daily.fetch_game(game_id).id == 948
    assert calls[0][0] == 'https://linxicon.com/play/daily'
    assert calls[0][1]['timeout'] == 30
    assert calls[0][1]['headers']['User-Agent'] == daily.USER_AGENT


def test_fetch_other_game_id_is_refused(monkeypatch):
    class Response:
        text = SNIPPET
        def raise_for_status(self): pass
    monkeypatch.setattr(daily.requests, 'get', lambda *a, **kw: Response())
    with pytest.raises(ValueError, match=r"only today's puzzle \(#948\)"):
        daily.fetch_game(900)


def test_fetch_failure(monkeypatch):
    def get(*args, **kwargs):
        raise requests.Timeout('timed out')
    monkeypatch.setattr(daily.requests, 'get', get)
    with pytest.raises(ValueError, match='fetch puzzle'):
        daily.fetch_game()
