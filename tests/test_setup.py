import requests
import pytest

from scripts import setup_data


class Response:
    def __init__(self, fail=False): self.fail = fail
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_content(self, chunk_size):
        yield b'data'
        if self.fail:
            raise requests.ConnectionError('interrupted')
        yield b'more'


def test_download_reuses_complete_file(tmp_path, monkeypatch):
    calls = []
    def get(*args, **kwargs):
        calls.append(kwargs)
        return Response()
    monkeypatch.setattr(setup_data.requests, 'get', get)
    destination = tmp_path/'nested'/'data.gz'
    setup_data.download('https://example.test/data.gz', destination)
    setup_data.download('https://example.test/data.gz', destination)
    assert destination.read_bytes() == b'datamore'
    assert len(calls) == 1 and calls[0]['stream'] is True
    assert not destination.with_suffix('.gz.part').exists()


def test_interrupted_download_does_not_publish_file(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_data.requests, 'get', lambda *a, **kw: Response(fail=True))
    destination = tmp_path/'data.gz'
    with pytest.raises(requests.ConnectionError):
        setup_data.download('https://example.test/data.gz', destination)
    assert not destination.exists()
    assert not destination.with_suffix('.gz.part').exists()


def test_setup_downloads_data_wordnet_and_builds_with_shared_defaults(tmp_path, monkeypatch):
    downloads, prepared, corpora = [], [], []
    monkeypatch.setattr(setup_data, 'DATA_DIR', tmp_path)
    def missing():
        raise ValueError('missing WordNet')
    monkeypatch.setattr(setup_data, 'ensure_wordnet', missing)
    monkeypatch.setattr(setup_data, 'download', lambda url, path: downloads.append(path.name))
    monkeypatch.setattr(setup_data.nltk, 'download', lambda name, **kw: corpora.append(name) or True)
    monkeypatch.setattr(setup_data, 'prepare', lambda **kw: prepared.append(kw))
    assert setup_data.main([]) == 0
    assert downloads == ['enable1.txt', 'numberbatch-en-19.08.txt.gz']
    assert corpora == ['wordnet']
    assert prepared[0]['min_zipf'] == 2.0 and prepared[0]['use_wordnet'] is True


def test_setup_full_and_no_wordnet(tmp_path, monkeypatch):
    prepared = []
    monkeypatch.setattr(setup_data, 'download', lambda *a: None)
    monkeypatch.setattr(setup_data.nltk, 'download', lambda *a, **kw: pytest.fail('disabled'))
    monkeypatch.setattr(setup_data, 'prepare', lambda **kw: prepared.append(kw))
    assert setup_data.main(['--vocab', 'full', '--no-wordnet']) == 0
    assert prepared[0]['min_zipf'] == 0 and prepared[0]['use_wordnet'] is False


def test_corpus_failure_does_not_build(monkeypatch, capsys):
    def missing():
        raise ValueError('missing WordNet')
    monkeypatch.setattr(setup_data, 'ensure_wordnet', missing)
    monkeypatch.setattr(setup_data, 'download', lambda *a: None)
    monkeypatch.setattr(setup_data.nltk, 'download', lambda *a, **kw: False)
    monkeypatch.setattr(setup_data, 'prepare', lambda **kw: pytest.fail('missing WordNet'))
    assert setup_data.main([]) == 1
    assert 'WordNet' in capsys.readouterr().err


def test_setup_reuses_installed_wordnet_without_network(monkeypatch):
    monkeypatch.setattr(setup_data, 'download', lambda *a: None)
    monkeypatch.setattr(setup_data, 'ensure_wordnet', lambda: None, raising=False)
    monkeypatch.setattr(setup_data.nltk, 'download', lambda *a, **kw: pytest.fail('installed WordNet must be reused'))
    monkeypatch.setattr(setup_data, 'prepare', lambda **kw: None)
    assert setup_data.main([]) == 0
