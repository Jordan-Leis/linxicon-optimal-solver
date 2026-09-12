from types import SimpleNamespace

import numpy as np
import pytest

from linxicon_solver import cli
from linxicon_solver.board import Board
from linxicon_solver.daily import Game
from linxicon_solver.graph import Graph
from linxicon_solver.server import Verification, WordRejectedError
from linxicon_solver.similarity import Similarity, Vectors


@pytest.fixture
def prepared(monkeypatch):
    words = ['chest', 'setting', 'box', 'place']
    vectors = Vectors(words, np.eye(4, dtype=np.float32))
    boosts = {frozenset((a, b)): score for a, b, score in [
        ('chest', 'box', .8), ('box', 'setting', .7),
        ('chest', 'place', .6), ('place', 'setting', .5)]}
    sim = Similarity(vectors, boosts=boosts, use_wordnet=False)
    graph = Graph.build(sim)
    calls = []
    def prepare(starters, **kwargs):
        calls.append((starters, kwargs))
        return sim, graph
    monkeypatch.setattr(cli, 'prepare', prepare)
    return sim, graph, calls


def test_default_cli_normalizes_and_simulates(prepared, capsys):
    assert cli.main(['CHEST', 'Setting']) == 0
    output = capsys.readouterr().out
    assert 'chest → box → setting' in output
    assert 'added: 1' in output and 'WIN' in output
    assert 'total:' in output and 'average:' in output and 'weakest:' in output
    assert prepared[2][0][0] == ('chest', 'setting')
    assert prepared[2][0][1]['min_zipf'] == 2.0


@pytest.mark.parametrize('args', [[], ['chest'], ['chest', 'chest'], ['ab', 'chest'],
    ['chest', 'setting', '--today'], ['--today', '--game', '942'],
    ['chest', 'setting', '--alternates', '0'], ['chest', 'setting', '--min-zipf', 'nan'],
    ['chest', 'setting', '--vocab', 'full', '--min-zipf', '2']])
def test_invalid_arguments(args):
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2


def test_alternate_limit_and_no_simulation(prepared, monkeypatch, capsys):
    monkeypatch.setattr(Board, 'simulate', lambda *a: pytest.fail('simulation disabled'))
    assert cli.main(['chest', 'setting', '--alternates', '1', '--no-simulate', '--vocab', 'full', '--no-wordnet']) == 0
    output = capsys.readouterr().out
    assert 'chest → box → setting' in output and 'chest → place → setting' not in output
    assert prepared[2][0][1]['min_zipf'] == 0.0
    assert prepared[2][0][1]['use_wordnet'] is False


@pytest.mark.parametrize('args, expected', [(['--today'], None), (['--game', '942'], 942)])
def test_daily_modes(prepared, monkeypatch, capsys, args, expected):
    def fetch(game_id=None):
        assert game_id == expected
        return Game(942, 'chest', 'setting', .0044, '2026-09-11')
    monkeypatch.setattr(cli, 'fetch_game', fetch)
    assert cli.main(args) == 0
    assert '942' in capsys.readouterr().out


def test_missing_vector(prepared, capsys):
    assert cli.main(['chest', 'unknown']) == 1
    assert 'unknown' in capsys.readouterr().err


def test_no_path(prepared, capsys):
    prepared[1].adj = [dict() for _ in prepared[1].words]
    assert cli.main(['chest', 'setting']) == 1
    assert 'No path' in capsys.readouterr().err


def test_direct_link(prepared, capsys):
    assert cli.main(['chest', 'box']) == 0
    assert 'added: 0' in capsys.readouterr().out


def test_board_cap(prepared, monkeypatch, capsys):
    monkeypatch.setattr(cli, 'MAX_WORDS', 2)
    assert cli.main(['chest', 'setting']) == 1
    assert 'board cap' in capsys.readouterr().err


def test_verification_replays_server_board_and_continues_after_rejection(prepared, monkeypatch, capsys):
    class Client:
        def verify(self, words):
            if 'box' in words:
                raise WordRejectedError('box: rejected')
            # Server starters connect directly, so replay should add zero words.
            pairs = {frozenset((a, b)): .9 for i, a in enumerate(words) for b in words[i+1:]}
            return Verification(words, [.9] * (len(words)-1), pairs)
    monkeypatch.setattr(cli, 'ServerClient', Client)
    assert cli.main(['chest', 'setting', '--verify']) == 0
    output = capsys.readouterr().out
    assert 'REJECTED' in output and 'server board: WIN' in output
    assert 'server added: 0' in output and 'server 0.900000' in output


def test_verify_no_simulate(prepared, monkeypatch, capsys):
    class Client:
        def verify(self, words):
            return Verification(words, [.9] * (len(words)-1), {})
    monkeypatch.setattr(cli, 'ServerClient', Client)
    monkeypatch.setattr(Board, 'simulate', lambda *a: pytest.fail('simulation disabled'))
    assert cli.main(['chest', 'setting', '--verify', '--no-simulate']) == 0
    assert 'server board: SKIPPED' in capsys.readouterr().out


def test_setup_failure_is_actionable(monkeypatch, capsys):
    def prepare(*a, **kw):
        raise FileNotFoundError('Run python scripts/setup_data.py')
    monkeypatch.setattr(cli, 'prepare', prepare)
    assert cli.main(['chest', 'setting']) == 1
    assert 'setup_data.py' in capsys.readouterr().err
