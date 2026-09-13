# Semantic Atlas export

The static atlas at `https://jordanleis.com/linxicon-solver/` replays real daily
search and verification events. Generate a standalone bundle with:

```bash
OPENBLAS_NUM_THREADS=2 python -m linxicon_solver.export_daily --output /tmp/atlas-data
```

`latest.json` points to a SHA-256 checked, content-hashed daily file. Schema 1
contains puzzle metadata, solver revision, configuration, timings, up to 500
projected words, cosine neighbors, local scoring evidence, sampled discovery
events with exact aggregate counts, candidates, and local/server board frames.
Missing server values are null. Coordinates use sign-normalized PCA; sampling
and ties are deterministic. Playback speed is independent of recorded timing.
Derived Numberbatch data is CC BY-SA 4.0; publication must include the Numberbatch
attribution and WordNet license notices bundled with the website.

The website's scheduled publishing workflow calls:

```bash
python -m linxicon_solver.publish_daily --output SITE/linxicon-solver/data \
  --previous-url https://jordanleis.com/linxicon-solver/data/ \
  --status /tmp/atlas-status.json
```

This checks the current puzzle before preparing large datasets, reuses an
unchanged complete result, and retries temporary verification failures up to
three times per puzzle and exporter revision. `--force` explicitly requests a
refresh. A generation failure preserves the last complete bundle; it is never
relabelled with a newer puzzle date. The website pins a reviewed solver commit.
Graph search and `Board.simulate()` also accept optional observer callbacks;
normal CLI return values remain unchanged.

## Graph export for in-browser solving

The website's "Any pair" tab solves arbitrary word pairs in the browser over a
copy of the common-vocabulary link graph. Generate it with:

```bash
OPENBLAS_NUM_THREADS=2 python -m linxicon_solver.export_graph --output SITE/linxicon-solver/data
```

This writes `graph-<sha256 prefix>.bin` (about 5 MB: a gzip-compressed `LXG1`
binary holding 51,029 words, 1,679,661 links with 16-bit scores) and a
`graph.json` manifest with the hash, counts, threshold, vocabulary floor,
scoring version and solver revision. The bundle is deterministic for a given
graph cache; a second run with the same content reports `changed: false` and
writes nothing. It is committed to the website once and only needs
regenerating when `SCORING_VERSION`, the threshold or the vocabulary changes.
The browser reproduces the local model only: the daily starters are added to
the vocabulary before the daily graph is built, so words outside the common
vocabulary cannot be solved there, and nothing is checked against the game.
