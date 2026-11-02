# Markov chain v3 roadmap (Rishav)

Author: Rishav Ranganathan. Proposed next steps after the v2 proposal. "v3" is Rishav's proposal track, not a version in `versions/`. Each phase starts once the previous gate is met.

Note: `allplays.csv` is on a secure server (see `AGENTS.md`), so Phase 1 may need to run there, or on the repo's cfbfastR data instead.

| Phase | Work | Gate to the next phase |
|---|---|---|
| 1. Real data | Run the upgrade on real play-by-play; confirm the four CSVs match last year's; fix anything the input checks flag; get a real accuracy number | The upgrade matches last year's CSVs on real data |
| 2. Into the pipeline | Real vision outputs and labels; report vision-only accuracy; turn on context mode; test error carryover when the previous play is the model's own prediction | Beats 87.2% on the same test clips |
| 3. Extend | Score, quarter and time as backoff levels; GT vs NCAA on held-out games; audio as a third conflation input; a conflation weight per class | Each add-on helps on held-out games |
| 4. Write it up | Update paper Sections 4.2-4.3; redo Figs. 9-10 on held-out games; fix Table 1's note (its 4th to 1st = 1.000 comes from the same-drive filter); report the ablation | — |

## Open questions for the team

1. Is playID's first number the drive number, as last year's code assumes?
2. Are the PAT and kickoff recorded at the end of the scoring drive?
3. Does the live feed give down and distance before the snap?
4. In the current pipeline, is the previous play the official one or the model's own prediction?
5. Which chain produced the paper's 87.2%?
6. Which play types end up in "Other" in `allplays.csv`?
7. Why did 56,596 parsed plays become 14,906?
