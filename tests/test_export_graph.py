import gzip
import hashlib
import json
import struct

import pytest

from linxicon_solver import export_graph as export
from linxicon_solver.graph import Graph


def tiny_graph():
    words = ['apple', 'banana', 'cherry', 'date', 'elder', 'fig']
    edges = {(0, 1): .5, (0, 2): .75, (1, 2): 1., (2, 5): .4, (3, 4): .9}
    return Graph(words, edges)


def test_pack_round_trips_words_edges_and_scores():
    graph = tiny_graph()
    payload = export.pack(graph)
    words, edges = export.unpack(payload)
    assert words == graph.words
    assert set(edges) == {(0, 1), (0, 2), (1, 2), (2, 5), (3, 4)}
    assert edges[(1, 2)] == pytest.approx(1., abs=1/65535)
    assert edges[(2, 5)] == pytest.approx(.4, abs=1/65535)


def test_layout_is_aligned_and_rows_are_delta_encoded():
    payload = export.pack(tiny_graph())
    assert payload[:4] == b'LXG1'
    word_count, edge_count, words_length = struct.unpack_from('<III', payload, 4)
    assert (word_count, edge_count) == (6, 5)
    words_end = 16 + words_length
    aligned = (words_end + 3) // 4 * 4
    degree = struct.unpack_from('<6H', payload, aligned)
    assert degree == (2, 1, 1, 1, 0, 0)
    neighbor_offset = aligned + 6 * 2
    neighbor_offset = (neighbor_offset + 3) // 4 * 4
    neighbors = struct.unpack_from('<5H', payload, neighbor_offset)
    # apple's row is [banana, cherry] -> first absolute (1), then delta (2-1).
    assert neighbors[:2] == (1, 1)


def test_degree_above_uint16_is_refused():
    graph = Graph(['hub'] + [f'w{i}' for i in range(70000)], {(0, i): .5 for i in range(1, 70001)})
    with pytest.raises(ValueError):
        export.pack(graph)


def test_write_bundle_names_file_by_hash_and_is_idempotent(tmp_path):
    graph = tiny_graph()
    manifest = export.write_bundle(graph, tmp_path, threshold=.3995, min_zipf=2., scoring_version='v', vector_key='k', revision='r')
    raw = (tmp_path / manifest['file']).read_bytes()
    assert manifest['file'] == f"graph-{manifest['sha256'][:16]}.bin"
    assert hashlib.sha256(raw).hexdigest() == manifest['sha256']
    assert gzip.decompress(raw)[:4] == b'LXG1'
    assert manifest['words'] == 6 and manifest['edges'] == 5
    assert json.loads((tmp_path / 'graph.json').read_text())['file'] == manifest['file']
    again = export.write_bundle(graph, tmp_path, threshold=.3995, min_zipf=2., scoring_version='v', vector_key='k', revision='r')
    assert again['sha256'] == manifest['sha256'] and again['changed'] is False
    assert manifest['changed'] is True
    assert len(list(tmp_path.glob('graph-*.bin'))) == 1
