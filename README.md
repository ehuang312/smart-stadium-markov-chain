# Smart Stadium Markov Chain

Play-type prediction for the Georgia Tech VIP Smart Stadium project. The Markov chain predicts the type of the upcoming play (run, pass, field goal, punt or other) from its pre-snap situation, and its probabilities are conflated with a video classifier's to give the final prediction (paper Sections 4.2–4.3).

Maintained by Ethan Huang.

## Current model: v2

On the 2017 FBS season (5-fold cross-validation by game, 140,279 plays):

| | v1 | v2 |
|---|---|---|
| Accuracy, all plays | 54.3% | **64.8%** |
| Accuracy, Georgia Tech offense | 67.3% | **80.3%** |
| Log loss (lower is better) | 0.870 | **0.723** |
| With a simulated 71% classifier (conflated) | 84.8% | **87.3%** |

v2 predicts from the upcoming play's down, distance, field zone, the offense's run tendency and the game situation (score and clock): 1,200 states in all. Rare states back off to broader ones, so no play type is ever given exactly 0%. Full details are in [docs/MARKOV_CHAIN_V2.md](docs/MARKOV_CHAIN_V2.md).

## Quick start

```bash
pip install -r requirements.txt
```

```python
from playByPlay import download_cfb_pbp, load_cfb_pbp
from markovChain import PlayMarkovChain

plays = load_cfb_pbp(download_cfb_pbp(2017))   # downloads once (~72 MB) into data/cfb_pbp/
mc = PlayMarkovChain().fit(plays)

situation = dict(offense='Georgia Tech', down=3, distance=8, yards_to_goal=65,
                 score_diff=0, half=1, period=2, seconds_left_half=600)
mc.next_play_type_proba(situation)   # [0.27, 0.70, 0.00, 0.00, 0.03] over R, P, F, U, O
```

Run the tests with `python -m pytest tests -q`. The notebook [notebooks/conflatedPlayPrediction.ipynb](notebooks/conflatedPlayPrediction.ipynb) runs the whole workflow, including the comparison against v1 and the marker cross-validation.

## Layout

| Path | What it is |
|---|---|
| `markovChain.py` | The current model (always the latest version) |
| `playByPlay.py` | Downloads and cleans cfbfastR play-by-play data |
| `conflation.py` | Fuses the Markov chain with classifier probabilities |
| `versions/` | Every model version as its own file: `markovChain_v1.py`, `markovChain_v2.py`, … |
| `archive/` | Older code kept for reference: v1 with its toy-data notebook, and last year's `MarkovChain.py` |
| `notebooks/` | The main notebook |
| `tests/` | Unit tests for the loader and the model |
| `docs/` | Change document, one-pager (Markdown and PDF), and the interactive state network |
| `SETUP.md` | Step-by-step environment setup for new teammates |
| `AGENTS.md` | Instructions for AI coding agents working in this repo |

## Documents

- [docs/MARKOV_CHAIN_V2.md](docs/MARKOV_CHAIN_V2.md): what changed from v1, new features and accuracy comparisons
- [docs/one_pager.md](docs/one_pager.md) and [docs/Markov_Chain_v2_One_Pager.pdf](docs/Markov_Chain_v2_One_Pager.pdf): how to use the model and plug it into the full pipeline
- [docs/play_call_state_network.html](docs/play_call_state_network.html): interactive network of all 1,200 states (download and open in a browser)

## Data

Training data is cfbfastR college play-by-play from the [SportsDataverse release](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfbfastR_cfb_pbp) (2014–2026, no API key), built mostly from [CollegeFootballData.com](https://collegefootballdata.com). It isn't committed: `download_cfb_pbp(season)` fetches it into `data/`, which is gitignored.

## Versioning

Each model update is saved as a new numbered file in `versions/`, and `markovChain.py` is updated to match. Old versions and `archive/` are never deleted. See [versions/README.md](versions/README.md).

## Asking Claude on GitHub

Collaborators can write `@claude` in an issue, a pull request or a review comment to get help from Claude, which replies in the thread and can push a branch with changes. Only the repo owner and invited collaborators can trigger it ([.github/workflows/claude.yml](.github/workflows/claude.yml)), and pull requests from outside contributors need the owner's approval before any workflow runs.
