import json
from dataclasses import replace

import numpy as np
import pytest

from linxicon_solver import export_daily as export
from linxicon_solver.daily import Game
from linxicon_solver.graph import Graph
from linxicon_solver.server import Verification, ServerError, WordRejectedError
from linxicon_solver.similarity import Similarity, Vectors


@pytest.fixture
def model():
    vectors = Vectors(['start', 'end', 'bridge', 'other'], np.array([[1., 0], [0, 1], [.8, .6], [-1, 0]], dtype=np.float32))
    sim = Similarity(vectors, use_wordnet=False)
    return Game(10, 'start', 'end', 0., '2026-09-12'), sim, Graph.build(sim)


class Client:
    def verify(self, words):
        pairs = {frozenset((a, b)): (.01 if {a,b} == {'start','end'} else .7)
                 for i,a in enumerate(words) for b in words[i+1:]}
        return Verification(words, [pairs[frozenset((a,b))] for a,b in zip(words, words[1:])], pairs)


def test_export_matches_search_neighbors_and_board(model):
    game, sim, graph = model
    result = export.build_daily(game, sim, graph, Client(), revision='test')
    assert result['schema_version'] == 1
    assert result['candidates'][0]['words'] == ['start','bridge','end']
    assert result['candidates'][0]['status'] == 'verified'
    assert result['candidates'][0]['server_frames'][-1]['path'] == ['start','bridge','end']
    start = next(n for n in result['nodes'] if n['word'] == 'start')
    assert start['neighbors'][0]['word'] == 'bridge'
    assert start['neighbors'][0]['score'] == pytest.approx(.8)
    assert result['search']['visited'] == 3
    assert len(result['nodes']) <= 500
    assert all('cosine' in e and 'boost' in e and 'local' in e for e in result['edges'])


def test_geometry_and_trace_are_reproducible(model):
    first = export.build_daily(*model, Client(), revision='test')
    second = export.build_daily(*model, Client(), revision='test')
    assert first['nodes'] == second['nodes']
    assert first['search'] == second['search']


def test_missing_server_values_remain_null(model):
    class Offline:
        def verify(self, words): raise ServerError('offline')
    result = export.build_daily(*model, Offline(), revision='test')
    assert result['verification_status'] == 'unavailable'
    assert result['candidates'][0]['status'] == 'unavailable'
    assert all(e['server'] is None for e in result['edges'])


def test_rejections_and_false_local_boosts(model):
    class Rejected:
        def verify(self, words): raise WordRejectedError('bridge is rejected')
    result = export.build_daily(*model, Rejected(), revision='test')
    assert result['candidates'][0]['status'] == 'rejected'
    class Low:
        def verify(self, words):
            pairs = {frozenset((a,b)):.01 for i,a in enumerate(words) for b in words[i+1:]}
            return Verification(words, [.01,.01],pairs)
    result = export.build_daily(*model, Low(), revision='test')
    assert result['candidates'][0]['status'] == 'rejected'
    assert result['candidates'][0]['local_frames'][-1]['path']
    assert not result['candidates'][0]['server_frames'][-1]['path']


def test_direct_and_disconnected_graphs(model):
    game, sim, graph = model
    direct = export.build_daily(replace(game, br='bridge'), sim, graph, Client(), revision='test')
    assert direct['candidates'][0]['added'] == 0
    assert len(direct['candidates'][0]['local_frames']) == 1
    empty = export.build_daily(replace(game, br='other'), sim, graph, Client(), revision='test')
    assert empty['candidates'] == [] and empty['verification_status'] == 'no_path'


def test_manifest_and_integrity(tmp_path, model):
    result = export.build_daily(*model, Client(), revision='test')
    manifest = export.write_bundle(result, tmp_path)
    assert manifest['file'].startswith('daily-10-')
    assert export.load_bundle(tmp_path)['game']['id'] == 10
    path = tmp_path/manifest['file']
    path.write_text('{}')
    with pytest.raises(ValueError): export.load_bundle(tmp_path)


def test_schema_rejects_nonfinite_or_wrong_references(model):
    result = export.build_daily(*model, Client(), revision='test')
    result['nodes'][0]['x'] = float('nan')
    with pytest.raises(ValueError): export.validate_bundle(result)


def test_rejected_words_are_read_from_candidate_errors():
    bundle = {'candidates': [
        {'status': 'rejected', 'error': 'fete: "fete" not found in dictionary.'},
        {'status': 'rejected', 'error': 'fetes: "fetes" not found in dictionary.'},
        {'status': 'rejected', 'error': 'damn: "damn" is not allowed.'},
        {'status': 'rejected', 'error': 'Server scores do not connect the starters.'},
        {'status': 'rejected', 'error': 'bahai: no vector on the server'},
        {'status': 'verified', 'error': None}]}
    assert export.rejected_words(bundle) == {'fete', 'fetes', 'damn'}


def test_build_daily_avoids_blocked_words(model):
    game, sim, graph = model
    result = export.build_daily(game, sim, graph, Client(), revision='test', blocked={'bridge'})
    assert all('bridge' not in c['words'] for c in result['candidates'])
