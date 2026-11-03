# Model versions

Every version of the Markov chain is kept here as its own file. `markovChain.py` at the repo root is always a copy of the newest one, so the notebook and tests run against it.

| Version | File | Date | Summary | 2017 accuracy | Log loss |
|---|---|---|---|---|---|
| v1 | `markovChain_v1.py` | Before Oct 2026 | Original repo model: 280 states (down, play type, 10 field buckets) of the previous play; trained on simulated games | 54.3% | 0.870 |
| v2 | `markovChain_v2.py` | 2026-10-14 | Predicts from the upcoming play's pre-snap situation: down, distance, field zone, team tendency, game situation (1,200 states), with back-off smoothing | 64.8% | 0.723 |
| v2.1 | `markovChain_v2_1.py` | 2026-11-02 | v2 plus `prefilter_plays`: drops 320 plays with impossible fields (down 0, snap on a goal line, unknown distance) and repairs 228 (goal-to-go distance from yards to goal, distance capped at the goal line). Model unchanged. GT 80.5% | 64.9% | 0.721 |

Accuracy and log loss are 5-fold cross-validation by game on the 2017 FBS season (140,279 test plays; 139,959 for v2.1 after prefiltering, where v2 scores 64.9% / 0.7215 on the same plays).

## Adding a version

1. Copy the current model to `markovChain_vN.py` (next number) and make the changes there.
2. Copy it over `markovChain.py` at the repo root.
3. Rerun `python -m pytest tests -q` and the notebook's sections 4 and 6, and add a row to the table above.
4. Never delete or overwrite an older `markovChain_v*.py` file.

`../archive/` also holds v1's original toy-data notebook and last year's `MarkovChain.py`.
