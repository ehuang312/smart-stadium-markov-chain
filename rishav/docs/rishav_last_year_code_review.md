# Review of last year's MarkovChain.py (Rishav)

Author: Rishav Ranganathan. Reviewed file: `archive/last_year/MarkovChain.py`. The indexing below was checked by re-implementing it separately (`rishav/rishav_markov.py`): all four matrices match exactly on simulated data.

## What it does

`Markov()` reads `allplays.csv` and drops rows with no type, drive-start rows (`#`) and penalties (`E`). Four methods build the paper's four chains and write a CSV each:

| Method | State | Matrix | Output |
|---|---|---|---|
| `createDownMarkov` | down | 4 x 5 | `Down.csv` (paper Table 1) |
| `createDownTypeMarkov` | down x type | 28 x 36 | `DownType.csv` |
| `createDownTypeFPMarkov` | + field position | 280 x 288 | `DownTypeFP.csv` |
| `createDownTypeFPDistMarkov` | + distance | 1,680 x 1,688 | `DownTypeFPDist.csv` |

After the states come a Total column (1 if the row has data, else 0) and 7 columns of P(next play type | state). Those 7 columns are what gets conflated with the video model.

## Data format

- The CSV header has 6 names and each row 7 values, so pandas uses the first number as the index; `playID` is then like `"4,5,27"`, and the code treats its first number as the drive.
- `context` like `"V,3,15,H44"` = possessor, down, distance, ball spot (the letter is whose half of the field).
- Field position: own side is buckets 0-4, opponent side is buckets 5-9 (9 minus the bucket); yard lines of 50+ are capped at 49. Distance buckets: 1-3, 4-6, 7-9, 10-14, 15-19, 20+.

## The cross-drive rule

A transition counts only within the same drive, or from a punt or kickoff into the next drive. So it drops:

- interceptions and fumbles
- missed field goals (so the FG row only learns FG to kickoff)
- turnovers on downs (so 4th-down run/pass rows only learn successful conversions)

The Down chain is stricter: same drive only. That's why Table 1 shows 4th to 1st = 1.000. Punts, field goals and failed attempts all end the drive and are thrown out, so only successful conversions remain.

## Problems, by impact

1. **No smoothing.** An unseen state gives an all-zero row; in conflation that zeroes every class (NaN after normalizing), and an argmax silently picks Run. Zeros inside rows that have data veto classes outright. With 2.8 million cells and about 15,000 plays, this likely explains why the finest chain does worst in Fig. 9.
2. **The cross-drive rule** above, applied differently in the Down chain.
3. **Silent mis-indexing.** A down of `0` or blank gives index -1, which numpy wraps to the 4th-down block.
4. **Midfield.** A ball spot written as plain `"50"` is read as letter `5`, yardage `0`: the opponent's goal line.
5. **No train/test split or evaluation code.** If Fig. 9 was scored on the training plays, it's in-sample.
6. **Game boundaries** rely on drive numbers restarting each game.
7. **Housekeeping:** the parsing loop is copied four times; `createDownTypeMarkov` re-reads the CSV; `playType_counts` is unused; row-by-row `.iloc` is slow (about 7-9 s for all four chains on 10,000 plays); the module-level call runs on import.

## Code vs. paper

- **Matches:** the four chains; 1,680 states from 4 downs, 7 types, 10 field and 6 distance buckets; Table 1's layout; kickoffs and PATs coded as 1st down.
- **In the paper, not here:** the five-category data cleaning, merging into the vision model's 5 classes, conflation, the one-play delay, the GT vs NCAA comparison, the code behind Figs. 9-10.
- **Here, not in the paper:** the drive-based rule, dropping penalty and drive-start rows, the 49-yard cap, the aggregate columns.
