# Command-line usage

See the [README](../README.md) for setup. Setup reuses existing downloads, the installed WordNet corpus and matching caches; interrupted downloads are not published as complete files. After setup, explicit-word solves need no internet connection. If your Python installation lacks `ensurepip`, install your distribution's Python venv support or provision a venv with pip separately. Set `NLTK_DATA` to a writable corpus directory if the default location is unsuitable.

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
| `--today` / `--game ID` | Fetch today's puzzle from `linxicon.com/play/daily`; cannot be combined with positional words. The game no longer serves past puzzles, so `--game ID` only succeeds when ID is today's and otherwise exits with a clear error. The puzzle date is the UTC date at fetch time (the page no longer states it). |
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
