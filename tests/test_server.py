import json

from linxicon_solver.server import (
    parse_semantics_response,
    parse_validate_response,
    serialize_args,
)

SEMANTICS_RESPONSE = (
    ';0x00000115;((self.$R=self.$R||{})["server-fn:0"]=[],($R=>$R[0]=['
    '$R[1]={id:"chest-keep",from:"node-chest",to:"node-keep",color:"#555555",score:0.060531583698313386},'
    '$R[2]={id:"keep-setting",from:"node-keep",to:"node-setting",color:"#555555",score:0.09924945538023916}'
    '])($R["server-fn:0"]))'
)


def test_parse_semantics_response_extracts_pair_scores():
    scores = parse_semantics_response(SEMANTICS_RESPONSE)
    assert scores[frozenset(("chest", "keep"))] == 0.060531583698313386
    assert scores[frozenset(("keep", "setting"))] == 0.09924945538023916


def test_parse_validate_response():
    assert parse_validate_response(';0x00000030;((self.$R=self.$R||{})["server-fn:0"]=[],void 0)') is None
    err = parse_validate_response(';0x0000004e;((self.$R=self.$R||{})["server-fn:0"]=[],"\\"xyzzy\\" not found in dictionary.")')
    assert err == '"xyzzy" not found in dictionary.'


def test_serialize_args_matches_seroval_json_format():
    body = json.loads(serialize_args([["chest", "setting"], "keep"]))
    assert body == {
        "t": {"t": 9, "i": 0, "a": [
            {"t": 9, "i": 1, "a": [{"t": 1, "s": "chest"}, {"t": 1, "s": "setting"}], "o": 0},
            {"t": 1, "s": "keep"}], "o": 0},
        "f": 127, "m": [],
    }

import pytest
import requests
from linxicon_solver.server import ServerClient, ServerError, MissingVectorError, WordRejectedError


def test_parsers_reject_unknown_payloads():
    for text in ['<html>down</html>', '', 'void 0', 'score:NaN']:
        with pytest.raises(ServerError):
            parse_semantics_response(text)
        with pytest.raises(ServerError):
            parse_validate_response(text)


def test_empty_semantics_is_recognised():
    assert parse_semantics_response(';0x1;((self.$R=self.$R||{})["server-fn:0"]=[],($R=>$R[0]=[])($R["server-fn:0"]))') == {}


def test_client_serializes_and_spaces_calls(monkeypatch):
    from linxicon_solver import server
    clock = [10.0]
    sleeps, calls = [], []
    monkeypatch.setattr(server.time, 'monotonic', lambda: clock[0])
    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay
    monkeypatch.setattr(server.time, 'sleep', sleep)
    class Response:
        text = ';0x1;((self.$R=self.$R||{})["server-fn:0"]=[],void 0)'
        def raise_for_status(self): pass
    class Session:
        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response()
    client = ServerClient(Session())
    assert client.validate_word('chest') is None
    assert client.validate_word('setting') is None
    assert client.validate_word('chest') is None
    assert len(calls) == 2
    assert sleeps == [0.5]
    assert calls[0][1]['timeout'] == 30
    assert json.loads(calls[0][1]['data']) == json.loads(serialize_args(['chest']))
    assert calls[0][1]['headers']['X-Server-Instance'] == 'server-fn:0'


def test_verification_seeds_starters_and_collects_all_pairs(monkeypatch):
    client = ServerClient()
    validated, added = [], []
    monkeypatch.setattr(client, 'validate_word', lambda w: validated.append(w))
    def update(board, word):
        added.append((list(board), word))
        return {frozenset((other, word)): 0.7 for other in board}
    monkeypatch.setattr(client, 'update_semantics', update)
    result = client.verify(['chest', 'box', 'place', 'setting'])
    assert validated == ['chest', 'box', 'place', 'setting']
    assert added == [(['chest'], 'setting'), (['chest', 'setting'], 'box'), (['chest', 'setting', 'box'], 'place')]
    assert result.scores == [0.7, 0.7, 0.7]
    assert len(result.pairs) == 6


def test_verification_rejects_bad_words_and_missing_pairs(monkeypatch):
    client = ServerClient()
    monkeypatch.setattr(client, 'validate_word', lambda w: 'not allowed')
    with pytest.raises(WordRejectedError, match='chest'):
        client.verify(['chest', 'setting'])
    monkeypatch.setattr(client, 'validate_word', lambda w: None)
    monkeypatch.setattr(client, 'update_semantics', lambda board, word: {})
    with pytest.raises(MissingVectorError, match='setting'):
        client.verify(['chest', 'setting'])
    monkeypatch.setattr(client, 'update_semantics', lambda board, word: {frozenset(('chest', word)): 0.5})
    with pytest.raises(ServerError, match='missing'):
        client.verify(['chest', 'box', 'setting'])


def test_http_failure_is_actionable():
    class Session:
        def post(self, *args, **kwargs):
            raise requests.Timeout('timed out')
    with pytest.raises(ServerError, match='timed out'):
        ServerClient(Session()).validate_word('chest')
