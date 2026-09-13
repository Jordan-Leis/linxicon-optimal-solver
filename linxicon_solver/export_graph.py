"""Export the common-vocabulary link graph as a compact bundle the website can solve over.

Format ``LXG1`` (little-endian, every section 4-byte aligned), gzip-compressed on disk:

    magic 'LXG1' | uint32 word_count | uint32 edge_count | uint32 words_length
    words: UTF-8, newline-joined, padded to 4 bytes (index = position)
    degree:   uint16[word_count]   neighbors b > a per word
    neighbor: uint16[edge_count]   rows in word order, ascending; first absolute, rest deltas
    score:    uint16[edge_count]   round(score * 65535)

Each undirected edge is stored once, on the row of its lower index.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from .data import prepare
from .export_daily import revision
from .graph import Graph
from .rules import THRESHOLD
from .similarity import SCORING_VERSION
from .vocab import DEFAULT_MIN_ZIPF

SCHEMA_VERSION = 1
MAGIC = b'LXG1'


def _pad(data: bytes) -> bytes:
    return data + b'\0' * (-len(data) % 4)


def pack(graph: Graph) -> bytes:
    words = '\n'.join(graph.words).encode()
    degree, neighbor, score = [], [], []
    for a, nbrs in enumerate(graph.adj):
        row = sorted(b for b in nbrs if b > a)
        if len(row) > 65535:
            raise ValueError(f'"{graph.words[a]}" has more than 65535 links.')
        degree.append(len(row))
        previous = 0
        for b in row:
            neighbor.append(b - previous)
            score.append(int(round(nbrs[b] * 65535)))
            previous = b
    if len(graph.words) > 65535:
        raise ValueError('More than 65535 words cannot be indexed with 16 bits.')
    header = MAGIC + struct.pack('<III', len(graph.words), len(neighbor), len(words))
    return (header + _pad(words) + _pad(np.array(degree, dtype='<u2').tobytes())
            + _pad(np.array(neighbor, dtype='<u2').tobytes()) + np.array(score, dtype='<u2').tobytes())


def unpack(payload: bytes) -> tuple[list[str], dict[tuple[int, int], float]]:
    if payload[:4] != MAGIC:
        raise ValueError('Not an LXG1 graph.')
    word_count, edge_count, words_length = struct.unpack_from('<III', payload, 4)
    offset = 16
    words = payload[offset:offset + words_length].decode().split('\n') if words_length else []
    offset += len(_pad(payload[offset:offset + words_length]))
    degree = np.frombuffer(payload, dtype='<u2', count=word_count, offset=offset)
    offset += len(_pad(degree.tobytes()))
    neighbor = np.frombuffer(payload, dtype='<u2', count=edge_count, offset=offset)
    offset += len(_pad(neighbor.tobytes()))
    score = np.frombuffer(payload, dtype='<u2', count=edge_count, offset=offset)
    edges, position = {}, 0
    for a, count in enumerate(degree.tolist()):
        previous = 0
        for _ in range(count):
            b = previous + int(neighbor[position])
            edges[(a, b)] = float(score[position]) / 65535
            previous, position = b, position + 1
    return words, edges


def write_bundle(graph: Graph, output: Path, *, threshold: float, min_zipf: float, scoring_version: str,
                 vector_key: str, revision: str) -> dict:
    raw = gzip.compress(pack(graph), compresslevel=9, mtime=0)
    digest = hashlib.sha256(raw).hexdigest()
    filename = f'graph-{digest[:16]}.bin'
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'graph.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get('sha256') == digest and (output / filename).exists():
            return dict(previous, changed=False)
    (output / filename).write_bytes(raw)
    manifest = dict(schema_version=SCHEMA_VERSION, file=filename, sha256=digest, words=len(graph.words),
                    edges=sum(1 for a, nbrs in enumerate(graph.adj) for b in nbrs if b > a),
                    threshold=threshold, min_zipf=min_zipf, scoring_version=scoring_version, vector_key=vector_key,
                    solver_revision=revision, generated_at=datetime.now(timezone.utc).isoformat())
    temporary = output / 'graph.tmp'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(manifest_path)
    for stale in output.glob('graph-*.bin'):
        if stale.name != filename:
            stale.unlink()
    return dict(manifest, changed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        sim, graph = prepare(progress=print)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1
    manifest = write_bundle(graph, args.output, threshold=THRESHOLD, min_zipf=DEFAULT_MIN_ZIPF,
                            scoring_version=SCORING_VERSION, vector_key=sim.vectors.cache_key, revision=revision())
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
