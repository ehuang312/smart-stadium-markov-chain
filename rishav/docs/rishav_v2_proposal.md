# What v2 should change (Rishav's proposal)

Author: Rishav Ranganathan. Proposed changes to last year's `archive/last_year/MarkovChain.py`. "v2" here is Rishav's proposal track, not `versions/markovChain_v2.py`; several of these ideas are already in Ethan's v2.

Ground rule: keep the same class, method names and four CSVs, and put a smarter predictor inside.

| # | Proposal | What it does | Simulated estimate |
|---|---|---|---|
| 1 | Add backoff smoothing | The four chains back each other up, so a rare state leans on a coarser one instead of returning an empty row | Steadier probabilities (log loss 1.10 to 0.84) |
| 2 | Count every play | Stop dropping turnovers, missed field goals and turnovers on downs just because the drive changes | +0.3 pts |
| 3 | Add context mode | Predict from the upcoming play's down, distance and ball spot, from the feed, instead of averaging over them | +3.1 pts (biggest) |
| 4 | Tune conflation | Learn how much to trust the chain versus the video, with a floor so one zero can't veto a class | +0.7 pts |
| 5 | Check the inputs | Raise a clear error on a bad down, a ball spot with no H/V letter or an odd distance | Prevents silent bugs |
| 6 | Build it once | Replace four copies of the same loop with one vectorized builder | Much faster builds |

Also: map to the vision model's 5 classes, add a live `predict_next()`, and stop running on import.

Expected accuracy, step by step (simulated games, 10 seeds): 83.9% today, then 84.6%, 84.7%, 85.0%, and 88.2% with context mode. These are estimates to confirm on real data.

## How smoothing works

Each prediction starts from the finest chain and fills in from coarser ones in proportion to how little data the fine one has:

`blended = (fine-chain count + alpha x coarser probability) / (fine-chain total + alpha)`

A state seen 500 times is mostly its own data; one seen 3 times mostly borrows; one never seen uses the coarser chain entirely. Each level is blended with the one above it, down to the overall play-type frequency, so the four existing chains become the rungs of one model. Alpha tuned to about 20-100 in simulation.
