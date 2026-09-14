"""Export an actual solve as a bounded, versioned static atlas replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import re

from .board import Board
from .daily import fetch_game
from .data import prepare
from .rules import MAX_WORDS, THRESHOLD
from .server import MissingVectorError, ServerClient, ServerError, WordRejectedError
from .similarity import SCORING_VERSION, _synsets

SCHEMA_VERSION = 1
MAX_NODES = 500


def revision() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent,
                                   text=True).strip()


def _projection(matrix):
    centered = matrix.astype(np.float64) - matrix.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    for axis in axes[:2]:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    points = centered @ axes[:2].T
    if points.shape[1] < 2:
        points = np.pad(points, ((0, 0), (0, 2-points.shape[1])))
    scale = max(float(np.abs(points).max()), 1e-9)
    return np.round(points/scale, 6)


def _neighbors(vectors, word, count=10):
    scores = vectors.cosines_to_all(word)
    # Stable alphabetical tie-break, including at the boundary of the list.
    order = np.lexsort((np.asarray(vectors.words), -scores))
    return [dict(word=vectors.words[i], score=round(float(scores[i]), 7))
            for i in order if vectors.words[i] != word][:count]


def _relationship(sim, a, b, boost):
    if boost <= 0:
        return None
    if not sim.use_wordnet:
        return dict(kind='lexical', label='Lexical boost')
    sa, sb = _synsets(a), _synsets(b)
    shared = sorted(set(sa) & set(sb), key=lambda s: s.name())
    if shared:
        return dict(kind='synonym', label='Shared WordNet meaning', synset=shared[0].name(), definition=shared[0].definition())
    for source, targets in ((sa, set(sb)), (sb, set(sa))):
        for s in source:
            for method, label in [('hypernyms', 'Direct broader / narrower meaning'), ('hyponyms', 'Direct broader / narrower meaning'),
                                  ('instance_hypernyms', 'Instance relationship'), ('instance_hyponyms', 'Instance relationship'),
                                  ('similar_tos', 'Similar adjective')]:
                related = sorted(set(getattr(s, method)()) & targets, key=lambda t: t.name())
                if related:
                    return dict(kind=method, label=label, synset=s.name(), related=related[0].name())
    return dict(kind='lemma', label='WordNet lemma or inflection relationship')


REJECTION = re.compile(r'^(?P<word>[a-z]+): "(?P=word)" (not found in dictionary|is not allowed)')


def rejected_words(bundle) -> set[str]:
    """Words the game's dictionary rejected while verifying this bundle's candidates."""
    found = set()
    for candidate in bundle.get('candidates', []):
        match = REJECTION.match(candidate.get('error') or '')
        if match:
            found.add(match.group('word'))
    return found


def build_daily(game, sim, graph, client, *, revision: str, blocked=frozenset()) -> dict:
    started = time.perf_counter()
    events = []
    chains = graph.shortest_chains(game.tl, game.br, k=5, observer=events.append, blocked=blocked)
    search_seconds = time.perf_counter() - started
    discoveries = [e for e in events if e['type'] == 'discover']
    complete = next((e for e in reversed(events) if e['type'] == 'complete'),
                    dict(visited=len(discoveries), frontier=0, examined=0))
    candidates, server_pairs = [], {}
    unavailable = False
    verification_start = time.perf_counter()
    for chain in chains:
        local_frames = []
        Board((game.tl, game.br), sim.score).simulate(chain.words[1:-1], observer=local_frames.append)
        candidate = dict(words=chain.words, scores=chain.scores, added=chain.added, total=chain.total,
                         average=chain.average, weakest=chain.weakest, local_frames=local_frames,
                         server_frames=[], server_scores=None, server_pairs=[], status='unavailable', error=None)
        if len(chain.words) > MAX_WORDS:
            candidate.update(status='rejected', error='Chain exceeds the board cap.')
        elif unavailable:
            candidate['error'] = 'Verification paused after a temporary server failure.'
        else:
            try:
                verified = client.verify(chain.words)
                frames = []
                result = Board((game.tl, game.br), verified.score).simulate(chain.words[1:-1], observer=frames.append)
                pairs = [dict(a=sorted(key)[0], b=sorted(key)[1], score=value) for key,value in sorted(verified.pairs.items(), key=lambda kv: sorted(kv[0]))]
                candidate.update(status='verified' if result.won else 'rejected', server_frames=frames,
                                 server_scores=verified.scores, server_pairs=pairs,
                                 error=None if result.won else 'Server scores do not connect the starters.')
                server_pairs.update(verified.pairs)
            except (WordRejectedError, MissingVectorError) as exc:
                candidate.update(status='rejected', error=str(exc))
            except ServerError as exc:
                candidate['error'] = str(exc)
                unavailable = True
        candidates.append(candidate)
    verification_seconds = time.perf_counter()-verification_start

    selected = dict.fromkeys([game.tl, game.br, *(w for c in chains for w in c.words)])
    layers = defaultdict(list)
    for event in discoveries:
        layers[event['depth']].append(event)
    for layer in sorted(layers):
        entries = layers[layer]
        for index in np.linspace(0, len(entries)-1, min(20, len(entries)), dtype=int):
            if len(selected) < 200:
                selected[entries[index]['word']] = None
    seeds = list(selected)
    neighbors = {}
    for word in seeds:
        neighbors[word] = _neighbors(sim.vectors, word)
        for neighbor in neighbors[word]:
            if len(selected) < MAX_NODES:
                selected[neighbor['word']] = None
    words = sorted(selected)
    coords = _projection(sim.vectors.matrix[[sim.vectors.index(w) for w in words]])
    discovered = {e['word']: (i,e) for i,e in enumerate(discoveries)}
    nodes = []
    for word, (x,y) in zip(words, coords):
        if word not in neighbors:
            neighbors[word] = _neighbors(sim.vectors, word)
        index, event = discovered.get(word, (None, {}))
        nodes.append(dict(word=word, x=float(x), y=float(y), depth=event.get('depth'),
                          discovery=index, parent=event.get('parent'), neighbors=neighbors[word]))

    # Candidate boards include every pair, even those below the game threshold.
    keys = dict.fromkeys(frozenset((a,b)) for c in candidates for i,a in enumerate(c['words']) for b in c['words'][i+1:])
    for event in discoveries:
        if event['word'] in selected and event['parent'] in selected:
            keys[frozenset((event['word'], event['parent']))] = None
    contextual = []
    indices = {graph._index[w] for w in words}
    for a in words:
        for j, score in graph.adj[graph._index[a]].items():
            b = graph.words[j]
            if j in indices and a < b:
                contextual.append((score, a, b))
    for _, a, b in sorted(contextual, key=lambda e: (-e[0],e[1],e[2])):
        if len(keys) >= 1200:
            break
        keys[frozenset((a,b))] = None
    edges = []
    for key in sorted(keys, key=lambda key: sorted(key)):
        a, b = sorted(key)
        cosine = sim.vectors.cosine(a,b)
        boost = sim.boost(a,b)
        edges.append(dict(a=a,b=b,cosine=round(cosine,7),boost=boost,local=max(0.,cosine,boost),
                          relationship=_relationship(sim,a,b,boost),server=server_pairs.get(key)))
    trace = [dict(event, order=i) for i,event in enumerate(discoveries) if event['word'] in selected]
    result = dict(schema_version=SCHEMA_VERSION, game=asdict(game), solver_revision=revision,
                  generated_at=datetime.now(timezone.utc).isoformat(),
                  config=dict(min_zipf=2., wordnet=True, threshold=THRESHOLD, scoring_version=SCORING_VERSION),
                  timings=dict(search=search_seconds, verification=verification_seconds, export=time.perf_counter()-started),
                  nodes=nodes, edges=edges, candidates=candidates,
                  search=dict(events=trace, visited=complete['visited'], frontier=complete['frontier'], examined=complete['examined'],
                              layers=[dict(depth=depth, discovered=len(items)) for depth,items in sorted(layers.items())]),
                  verification_status=('no_path' if not chains else 'unavailable' if unavailable else 'complete'),
                  license='CC-BY-SA-4.0')
    validate_bundle(result)
    return result


def validate_bundle(data):
    if data.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported atlas schema.')
    json.dumps(data, allow_nan=False)
    words = {n['word'] for n in data['nodes']}
    if len(words) != len(data['nodes']) or len(words) > MAX_NODES:
        raise ValueError('Invalid atlas nodes.')
    if not {data['game']['tl'], data['game']['br']} <= words:
        raise ValueError('Atlas is missing starters.')
    for edge in data['edges']:
        if edge['a'] not in words or edge['b'] not in words:
            raise ValueError('Unknown edge endpoint.')
    for candidate in data['candidates']:
        if not set(candidate['words']) <= words:
            raise ValueError('Candidate is missing from map.')
        if candidate['status'] == 'verified' and (not candidate['server_frames'] or not candidate['server_frames'][-1]['path']):
            raise ValueError('Verified candidate has no server board path.')


def write_bundle(data, output: Path, *, attempts=1):
    validate_bundle(data)
    raw = json.dumps(data, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    filename = f"daily-{data['game']['id']}-{digest[:16]}.json"
    output.mkdir(parents=True, exist_ok=True)
    (output/filename).write_bytes(raw)
    manifest = dict(schema_version=SCHEMA_VERSION, file=filename, sha256=digest, game=data['game'],
                    solver_revision=data['solver_revision'], generated_at=data['generated_at'],
                    verification_status=data['verification_status'], attempts=attempts)
    temporary = output/'latest.tmp'
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(output/'latest.json')
    return manifest


def load_bundle(output: Path):
    manifest = json.loads((output/'latest.json').read_text())
    name = manifest['file']
    if Path(name).name != name:
        raise ValueError('Invalid bundle path.')
    raw = (output/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['sha256']:
        raise ValueError('Atlas bundle checksum mismatch.')
    result = json.loads(raw)
    validate_bundle(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    game = fetch_game()
    sim, graph = prepare((game.tl,game.br), progress=print)
    from .vocab import load_rejected
    from .data import DATA_DIR
    result = build_daily(game, sim, graph, ServerClient(), revision=revision(), blocked=load_rejected(DATA_DIR/'rejected_words.txt'))
    result['timings']['total'] = time.perf_counter()-start
    manifest = write_bundle(result, args.output)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
