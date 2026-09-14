from dataclasses import replace
import json

import pytest

from linxicon_solver import publish_daily as publish
from linxicon_solver.export_daily import write_bundle, load_bundle
from tests.test_export_daily import model, Client
from linxicon_solver.export_daily import build_daily


def test_first_deployment_and_unchanged_day(tmp_path, model):
    game,sim,graph=model
    calls=[]
    def generate(game):
        calls.append(game.id)
        return build_daily(game,sim,graph,Client(),revision='abc')
    assert publish.update(tmp_path,game,'abc',generate)['changed'] is True
    assert publish.update(tmp_path,game,'abc',generate)['changed'] is False
    assert calls==[10]


def test_rollover_and_revision_change_regenerate(tmp_path, model):
    game,sim,graph=model
    generate=lambda game:build_daily(game,sim,graph,Client(),revision='abc')
    publish.update(tmp_path,game,'abc',generate)
    assert publish.update(tmp_path,replace(game,id=11,date='2026-09-13'),'abc',generate)['changed']
    assert load_bundle(tmp_path)['game']['id']==11
    assert publish.update(tmp_path,replace(game,id=11),'new',generate)['changed']


def test_failed_generation_preserves_previous_complete_artifact(tmp_path, model):
    game,sim,graph=model
    write_bundle(build_daily(game,sim,graph,Client(),revision='abc'),tmp_path)
    before=(tmp_path/'latest.json').read_bytes()
    def broken(game): raise RuntimeError('server offline')
    result=publish.update(tmp_path,replace(game,id=11),'abc',broken)
    assert result['outcome']=='stale' and not result['changed']
    assert (tmp_path/'latest.json').read_bytes()==before
    assert load_bundle(tmp_path)['game']['id']==10


def test_temporary_verification_retry_cap(tmp_path, model):
    game,sim,graph=model
    def generate(game):
        result=build_daily(game,sim,graph,Client(),revision='abc')
        result['verification_status']='unavailable'
        return result
    for attempt in range(1,4):
        assert publish.update(tmp_path,game,'abc',generate)['changed']
        assert json.loads((tmp_path/'latest.json').read_text())['attempts']==attempt
    result=publish.update(tmp_path,game,'abc',generate)
    assert not result['changed'] and result['outcome']=='retry_limit'


def test_definitive_rejection_is_not_retried(tmp_path, model):
    game,sim,graph=model
    from linxicon_solver.server import WordRejectedError
    class Rejected:
        def verify(self,words):raise WordRejectedError('rejected')
    def generate(game):return build_daily(game,sim,graph,Rejected(),revision='abc')
    publish.update(tmp_path,game,'abc',generate)
    assert not publish.update(tmp_path,game,'abc',generate)['changed']


def test_restore_validates_before_replacing_snapshot(tmp_path, model, monkeypatch):
    game,sim,graph=model
    target=tmp_path/'site';remote=tmp_path/'remote'
    write_bundle(build_daily(game,sim,graph,Client(),revision='old'),target)
    manifest=write_bundle(build_daily(replace(game,id=11),sim,graph,Client(),revision='new'),remote)
    class Response:
        def __init__(self,raw):self.content=raw
        def raise_for_status(self):pass
        def json(self):return json.loads(self.content)
    def get(url,**kwargs):return Response((remote/url.rsplit('/',1)[1]).read_bytes())
    monkeypatch.setattr(publish.requests,'get',get)
    assert publish.restore_published('https://example.test/data',target)
    assert load_bundle(target)['game']['id']==11
    before=(target/'latest.json').read_bytes()
    (remote/manifest['file']).write_bytes(b'corrupt')
    assert not publish.restore_published('https://example.test/data',target)
    assert (target/'latest.json').read_bytes()==before


def test_rejected_list_merges_seed_previous_and_new_rejections(tmp_path):
    (tmp_path/'rejected.json').write_text(json.dumps({'schema_version':1,'words':['bahai','fete']}))
    result=publish.update_rejected(tmp_path,seed={'connexion','fete'},learned={'fetes'})
    data=json.loads((tmp_path/'rejected.json').read_text())
    assert data['schema_version']==1 and data['words']==['bahai','connexion','fete','fetes']
    assert result==set(data['words'])
    assert publish.update_rejected(tmp_path,seed=set(),learned=set())==set(data['words'])
    assert publish.load_rejected_list(tmp_path)=={'bahai','connexion','fete','fetes'}
    assert publish.load_rejected_list(tmp_path/'nowhere')==set()


def test_restore_also_fetches_the_published_rejected_list(tmp_path, model, monkeypatch):
    game,sim,graph=model
    target=tmp_path/'site';remote=tmp_path/'remote'
    write_bundle(build_daily(game,sim,graph,Client(),revision='old'),target)
    write_bundle(build_daily(game,sim,graph,Client(),revision='new'),remote)
    (remote/'rejected.json').write_text(json.dumps({'schema_version':1,'words':['fete']}))
    class Response:
        def __init__(self,raw):self.content=raw
        def raise_for_status(self):pass
        def json(self):return json.loads(self.content)
    def get(url,**kwargs):return Response((remote/url.rsplit('/',1)[1]).read_bytes())
    monkeypatch.setattr(publish.requests,'get',get)
    assert publish.restore_published('https://example.test/data',target)
    assert publish.load_rejected_list(target)=={'fete'}
