# Agent instructions

This repo holds the play-type Markov chain for the Georgia Tech VIP Smart Stadium project. It is shared between teammates; each person's AI coding agent should follow these rules.

## Rules

1. **Never push without explicit approval.** Commit locally as you work, but run `git push` only when your user says so in that conversation. Approval for one push doesn't carry over to the next.
2. **Version every model change.** When the model changes, create a new file `versions/markovChain_vN.py` with the next number, then copy it over `markovChain.py`. Follow [versions/README.md](versions/README.md) and add a row to its table.
3. **Never delete old models.** Don't delete, rename or overwrite anything in `versions/` or `archive/`, including last year's `archive/last_year/MarkovChain.py`.
4. **Keep a local chatlog.** After every message from your user, append an entry to `CHATLOG.md`: the date, your user's name, their message, and a one- or two-line summary of what you did or answered. If you don't know your user's name, ask once and remember it. `CHATLOG.md` is gitignored: it stays on your machine and is never committed or pushed.
5. **Don't commit data or secrets.** `data/` is gitignored (season files are 70–130 MB). API keys go in `.env`, which is also gitignored.

## Project facts

- **Model:** `PlayMarkovChain` in `markovChain.py`. `fit(plays)` takes the DataFrame from `playByPlay.load_cfb_pbp`, passed through `prefilter_plays` first (the loader leaves impossible downs and distances as recorded). `next_play_type_proba(situation)` returns 5 probabilities in the order R, P, F, U, O (run, pass, field goal, punt, other), always summing to 1 and never exactly 0.
- **Conflation:** `conflation.conflate_probabilities([classifier_probs, mc_probs])` multiplies the two vectors and renormalizes. Each source's probabilities are first raised to at least `floor` (default 1e-4), so one source's zero can't veto a class the other is sure of. The classifier's output must use the same R, P, F, U, O order. The video classifier doesn't exist yet; the notebook simulates one.
- **Data:** cfbfastR play-by-play via `playByPlay.download_cfb_pbp(season)`. Only 2017 has been used so far. `load_cfb_pbp` accepts a list of seasons.
- **"Other" (O)** is almost entirely pre-snap fouls in this data. PATs aren't separate rows, so there is no PAT class.
- **Last year's training file `allplays.csv`** lives on a secure server. Don't ask for it or plan around it.

## Evaluating changes

- Split train and test by whole games, never by plays.
- Report log loss as well as accuracy: conflation uses the whole probability vector, so calibration matters.
- Report Georgia Tech offense separately; the project's footage is Georgia Tech games.
- The notebook's section 6 has a 5-fold cross-validation harness and ways to register new markers in `MARKERS`.

## Commands

```bash
pip install -r requirements.txt
python -m pytest tests -q
python exportOutputs.py      # writes outputs/*.csv (gitignored) and models/markov_v*_<seasons>.json
```

The notebook expects to run from the repo root or from `notebooks/`; it finds the root by looking for `conflation.py`.
