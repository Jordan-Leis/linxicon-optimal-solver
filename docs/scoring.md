# Scoring, limits, data and validation

## Scoring and limits

Reverse-engineering captured on September 12, 2026 found that the site's FAQ description of MiniLM did not match measured scores. The implemented model uses:

- `max(0, cosine)` of normalized ConceptNet Numberbatch 19.08 English vectors.
- A **1.0** boost for shared WordNet synsets.
- A **0.6** boost for direct hypernym/hyponym or adjective `similar_to` relationships.
- Exact-form WordNet lemma lookup first, falling back to morphological lookup only when the input is not itself a lemma. Boost lookup is symmetric.

The strongest applicable score wins. The original captured research found exact base-score agreement on about 96% of pairs and additional server boosts that only increased scores in that sample. **Later live verification disproved a universal lower-bound claim:** `leaves → segment` scored 0.6 locally but approximately 0.054383 on the server. The precise cause of this WordNet boost discrepancy remains unresolved. Use `--verify` before relying on a chain. Local float32 computation also introduces small rounding differences near the 0.3995 threshold.

Board simulation uses the captured client rules: a **0.3995** link threshold, a **50-word** cap including starters, and top-five pruning where an edge survives if it is in either endpoint's five strongest links. Among shortest paths, the game prefers the highest total score. Total and average ranking agree when path lengths are equal.

“Optimal” means optimal within the selected vocabulary and local model. The site's extra boosts and broader dictionary may allow shorter chains. The local dictionary is a candidate filter, not a complete mirror of the server's dictionary: profanity and other rejected words may occur. Inflections remain eligible. Some locally available vectors are absent on the server. Use verification and alternates to detect these mismatches; the solver does not automatically search longer paths after rejecting the displayed candidates.

Missing starter vectors, missing pair scores, HTTP errors, and unrecognized server formats fail explicitly. Server function identifiers are captured in `linxicon_solver/server.py`; a site deployment may require updating them and their response fixtures.

## Data and caches

Data lives under the checkout's `data/`, regardless of the invocation directory. ENABLE is committed; Numberbatch, the venv, and generated caches are gitignored. Saved game pages in `source-from-web/` remain reference material.

Vector caches are keyed by the requested vocabulary and the fixed Numberbatch release filename. Graph names also include threshold, WordNet setting, and scoring implementation version. Starters are added before vector loading, so starters outside the selected vocabulary create a different cache. Clear `data/cache/` to force a rebuild, including if replacing vector contents under the same filename. Full-vocabulary builds require more time and memory than the default.

## Validation

```bash
python -m pytest tests/ -q
```

Tests use tiny vector fixtures and mocked HTTP calls. WordNet tests require the installed corpus; the calibration test uses captured server values and is skipped if Numberbatch is absent. The test suite does not call the game server.

### Results recorded September 12, 2026

- **79 tests passed**. Repeating setup successfully reused the installed corpus and existing caches without network access. Installed dependencies passed `pip check`.
- A default graph contains **51,029 words and 1,679,661 edges**; a fresh build took **27.3 seconds** using two OpenBLAS threads.
- Cached `chest → setting` solving returned five locally winning chains with three added words. One result was `chest → bureau → office → place → setting`, with scores 1.0, 1.0, 1.0, 0.6.
- An isolated warm run took approximately **10.0 seconds**: vocabulary/corpus preparation 3.0 s, vector loading 0.1 s, graph loading 2.0 s, search 2.8 s, and simulation 2.1 s. **The under-five-second target is not met.** A full subprocess measured 12.5 s while tests ran concurrently. These timings used `OPENBLAS_NUM_THREADS=2`; performance depends on the machine and vocabulary.
- `python -m linxicon_solver --today --verify` fetched **game #943**, dated **2026-09-12**, with starters **sick → segment**. All three available shortest local candidates were checked. The first (`sick → pass → leaves → segment`) failed on the server, despite a local win.
- The highest-ranked successful candidate was **sick → green → leafs → segment**. All four words passed validation; server link scores were **0.600000, 0.478765, 0.600000**. The complete server-score board replay reported **WIN after two additions**. A third candidate through `greenest → leafs` also passed verification.
- **Manual browser confirmation: pending.** On [game #943](https://linxicon.com/game/943?enterGame=), add **green**, then **leafs** (exact spelling). Server-score replay alone is not recorded as confirmation of the game's win screen.
