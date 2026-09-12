import gzip

import pytest

from linxicon_solver import data
from linxicon_solver.graph import Graph
from linxicon_solver.similarity import Vectors


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    (tmp_path / 'enable1.txt').write_text('apple\nbanana\n')
    with gzip.open(tmp_path / data.NUMBERBATCH_NAME, 'wt') as f:
        f.write('3 2\napple 1 0\nbanana 0.8 0.6\nexcluded 0 1\n')
    monkeypatch.setattr(data, 'build_vocab', lambda path, min_zipf: ['apple', 'banana'])
    return tmp_path


def test_prepare_includes_starters_and_reuses_graph(tiny_data, monkeypatch):
    sim, graph = data.prepare(('apple', 'excluded'), data_dir=tiny_data, use_wordnet=False)
    assert 'excluded' in sim.vectors and 'excluded' in graph
    assert list((tiny_data/'cache').glob('graph-*.npz'))
    monkeypatch.setattr(Graph, 'build', lambda *a, **kw: pytest.fail('cache should be reused'))
    _, again = data.prepare(('apple', 'excluded'), data_dir=tiny_data, use_wordnet=False)
    assert again.words == graph.words


def test_graph_cache_identity_changes_with_scoring_inputs(tiny_data, monkeypatch):
    v = Vectors.load(tiny_data/data.NUMBERBATCH_NAME, ['apple'], tiny_data/'cache')
    original = data.graph_cache_path(tiny_data/'cache', v.cache_key, .3995, True)
    assert original != data.graph_cache_path(tiny_data/'cache', v.cache_key, .4, True)
    assert original != data.graph_cache_path(tiny_data/'cache', v.cache_key, .3995, False)
    assert original != data.graph_cache_path(tiny_data/'cache', 'other', .3995, True)
    monkeypatch.setattr(data, 'SCORING_VERSION', 'next-version')
    assert original != data.graph_cache_path(tiny_data/'cache', v.cache_key, .3995, True)


def test_missing_data_never_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match='setup_data.py'):
        data.prepare(('apple', 'banana'), data_dir=tmp_path, use_wordnet=False)


def test_missing_wordnet_has_actionable_error(tiny_data, monkeypatch):
    import nltk.data
    def missing(*a):
        raise LookupError('missing')
    monkeypatch.setattr(nltk.data, 'find', missing)
    with pytest.raises(ValueError, match='WordNet.*setup_data.py'):
        data.prepare(('apple', 'banana'), data_dir=tiny_data)
