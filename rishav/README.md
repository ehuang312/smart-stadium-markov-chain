# Rishav's Markov chain work

Owner: **Rishav Ranganathan**. Everything in this folder is Rishav's; every file name starts with `rishav_` so it can't be confused with the main model. Nothing outside `rishav/` was changed.

## How this relates to the rest of the repo

- The repo's current model is Ethan's v2 (`markovChain.py`, `versions/markovChain_v2.py`), trained on real cfbfastR play-by-play.
- Rishav's code is a separate track: an upgrade of **last year's** `archive/last_year/MarkovChain.py`, which reads the `allplays.csv` format. It does not replace or modify the main model, `versions/` or `archive/`.
- Several of Rishav's proposals are already in Ethan's v2: backing off to coarser states, predicting from the upcoming play's pre-snap situation, and adding score and clock. The rest (input checks, the cross-drive rule, a faster drop-in for last year's code) apply to last year's code.
- **Every accuracy number here comes from simulated games**, not real data. `allplays.csv` lives on a secure server, so the harness simulates games in its format.
- "v2" and "v3" in Rishav's documents mean Rishav's proposal track, **not** `versions/markovChain_v2.py`.

## Files

| Path | What it is |
|---|---|
| `rishav_markov.py` | Drop-in upgrade of last year's `Markov` class. It writes the same four CSVs, and adds backoff smoothing, a cross-drive option, input checks, context mode, 5-class mapping and conflation |
| `rishav_evaluate.py` | Runs last year's file unmodified against `rishav_markov.py` on simulated games: an exact-reproduction check and a step-by-step accuracy comparison |
| `rishav_simulate_allplays.py` | Simulated games written in last year's `allplays.csv` format, plus a simulated video classifier |
| `paper_based/` | Rishav's first rewrite, written from the paper before seeing last year's code, compared with the repo's v1 |
| `docs/rishav_last_year_code_review.md` | Line-by-line review of last year's `MarkovChain.py` against the paper |
| `docs/rishav_v2_proposal.md` | The six proposed changes, with simulated estimates |
| `docs/rishav_v3_roadmap.md` | Proposed next phases, with the gate to pass before each one |

## Running it

From the repo root (needs `numpy` and `pandas`):

```bash
python rishav/rishav_evaluate.py --check      # rishav_markov.py reproduces last year's 4 CSVs exactly
python rishav/rishav_evaluate.py --seeds 10   # accuracy, one change at a time (~1.5 min)
python rishav/paper_based/rishav_paper_evaluate.py --seeds 10
```

## Simulated results

10 seeds, held-out games, with a simulated video classifier at about 73% accuracy, conflated with the chain:

| Step | Final accuracy | Change |
|---|---|---|
| Last year's code, best chain (DownTypeFP) | 83.9% | baseline |
| + tuned conflation weight | 84.6% | +0.7 |
| + backoff smoothing across the four chains | 84.7% | +0.8 |
| + keep every within-game transition | 85.0% | +1.1 |
| + context mode (upcoming play's down, distance, field position) | 88.2% | +4.2 |

## Fewer zeros, lighter files

Last year's finest chain is a 1,680 x 1,688 matrix that is about 99.9% zeros (11.4 MB as a CSV). `rishav_markov.py` keeps that layout for compatibility, and adds two compact outputs, on simulated data:

| Output | Size | Zeros |
|---|---|---|
| `Markov.build('DownTypeFPDist')`: last year's full matrix | 1,680 x 1,688, 11.4 MB | ~99.9% |
| `Markov.build_transition_list(...)`: only transitions that happened | 3,556 rows, 190 KB | none stored |
| `Markov.build_compact(...)`: states that occurred, smoothed P(next play type) | 381 rows x 8, 62 KB | no exact zeros |

`PlayPredictor` also stores only situations it has seen (about 3,400-4,400 cells instead of about 14,000), with identical predictions.

`--check` also confirms that `rishav_markov.py` writes CSVs with the same labels and values as last year's file, about 9x faster.
