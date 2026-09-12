# Linxicon Optimal Solver

Find chains connecting two Linxicon starter words, ranked by the fewest added words and then the highest total link score. Search runs locally using ConceptNet Numberbatch 19.08 and WordNet. Optional server verification checks dictionary acceptance, retrieves actual scores, and replays the board.

## Setup

Python 3.12 is the tested version. From this checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/setup_data.py
```

Setup downloads the ENABLE dictionary if absent, the approximately 325 MB Numberbatch archive, and the NLTK `wordnet` corpus. It then builds vector and graph caches. A GPU is unnecessary. Existing downloads, installed WordNet, and matching caches are reused; interrupted downloads are not published as complete files. After setup, explicit-word solves need no internet connection.

If your Python installation lacks `ensurepip`, install your distribution's Python venv support or provision a venv with pip separately. Set `NLTK_DATA` to a writable corpus directory if the default location is unsuitable.

## Usage

```bash
python -m linxicon_solver chest setting
python -m linxicon_solver --today
python -m linxicon_solver --game 942
python -m linxicon_solver --today --verify --alternates 1
python -m linxicon_solver chest setting --vocab full
python -m linxicon_solver chest setting --min-zipf 3.0
python -m linxicon_solver chest setting --no-wordnet --no-simulate
```

| Option | Behavior |
| --- | --- |
| Two positional words | Solve a supplied pair; normalize uppercase input. |
| `--today` / `--game ID` | Fetch today's or a numbered puzzle; cannot be combined with positional words. Today falls back to the homepage Play form when `/game` serves no puzzle. |
| `--alternates N` | Print up to **N total candidates**, default 5, all at the shortest local length. |
| `--vocab common` | Default: ENABLE words with English Zipf frequency at least **2.0**. |
| `--vocab full` | Include all valid ENABLE words without a frequency floor. |
| `--min-zipf FLOAT` | Override the common vocabulary floor; cannot be combined with `--vocab full`. |
| `--no-wordnet` | Use Numberbatch cosine alone. |
| `--verify` | Validate all words in every displayed candidate and compare local/server scores. |
| `--no-simulate` | Skip local and server board replay; verification still checks words and link scores. |

Setup accepts the same `--vocab`, `--min-zipf`, and `--no-wordnet` options. Use matching options to prebuild a particular graph.

Output includes the chain, individual link scores, added-word count, total, average, weakest link, and simulation outcome. Both starters are already on the board; enter only the intermediate words, in the printed order. A direct starter link requires zero additions.

`--verify` preserves local ranking, marks rejected candidates, and reports the board path found using the complete server pair-score table. Extra server links can produce a shorter board path than the candidate. Validation results are reused across alternates within the command. Requests are spaced at least 0.5 seconds apart; the server is never queried during search. Start with `--alternates 1` to minimize requests.

Exit status is 0 when at least one candidate succeeds, 1 for unavailable data, no solution, or verification failure, and 2 for invalid arguments. With simulation disabled, success means a candidate exists (or its words and links pass verification); it does not certify a board win.

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
