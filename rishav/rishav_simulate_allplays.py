"""
Simulated play-by-play in the same allplays.csv format the team's code reads,
used only to measure the code changes. Not real data.

Row format (matches the CBS example in the paper):
    24,"4,5,27",P,,,"H,3,6,H42",
    index, playID "drive,play-in-drive,play-in-game", type, clock, token, context, text

Drive convention, inferred from the original code's comments: a drive is one
possession; a PAT and the following kickoff belong to the scoring drive, so a
drive ends in a punt (U), a kickoff (K), a turnover, a missed FG, a turnover
on downs, or the end of a half. '#' rows start drives; 'E' rows are penalties.
"""
import csv

import numpy as np

from rishav_markov import CLASSIFIER_CLASSES

TEAMS = ('H', 'V')


def _choose_play(rng, down, togo, yl, run_rate):
    if down == 4:
        go = 0.6 if togo <= 1 else 0.35 if togo <= 3 else 0.08
        if yl < 40:
            go *= 0.2
        if rng.random() >= go:
            return 'F' if 117 - yl <= 53 else 'U'
    p_run = run_rate + {1: 0.05, 2: 0.0, 3: -0.1, 4: 0.0}[down]
    if togo <= 2:
        p_run += 0.2
    elif togo >= 8 and down >= 2:
        p_run -= 0.25
    return 'R' if rng.random() < np.clip(p_run, 0.05, 0.95) else 'P'


def simulate_game(rng, n_plays=135, run_rates=(0.68, 0.48)):
    """One game as a list of (playID, type, context). Team H is run-heavy (GT-like)."""
    rows = []
    s = {'drive': 0, 'pid': 0, 'n': 0, 'off': 0, 'down': 1, 'togo': 10, 'yl': 25}

    def context(off, down, togo, yl):
        letter, num = (TEAMS[off], yl) if yl <= 50 else (TEAMS[1 - off], 100 - yl)
        return f'{TEAMS[off]},{down},{togo},{letter}{num:02d}'

    def emit(ptype, ctx):
        s['pid'] += 1
        s['n'] += 1
        rows.append((f"{s['drive']},{s['pid']},{s['n']}", ptype, ctx))

    def start_drive(off, yl):
        yl = int(np.clip(yl, 1, 99))
        s['drive'] += 1
        s['pid'] = 0
        s.update(off=off, down=1, yl=yl, togo=min(10, 100 - yl))
        emit('#', context(off, 1, s['togo'], yl))

    def kickoff(kicker, from_yl=35):
        emit('K', context(kicker, 1, 10, from_yl))
        start_drive(1 - kicker, rng.integers(18, 40))

    start_drive(1, 35)
    kickoff(1)
    scrimmage, second_half = 0, False
    while scrimmage < n_plays:
        if not second_half and scrimmage >= n_plays // 2:
            second_half = True
            start_drive(0, 35)
            kickoff(0)
            continue
        off, down, togo, yl = s['off'], s['down'], s['togo'], s['yl']
        ctx = context(off, down, togo, yl)
        scrimmage += 1

        if rng.random() < 0.05:                            # penalty row; replay the down
            emit('E', ctx)
            new_yl = int(np.clip(yl + (5 if rng.random() < 0.45 else -5), 1, 99))
            left = togo - (new_yl - yl)
            if left <= 0:
                s.update(yl=new_yl, down=1, togo=min(10, 100 - new_yl))
            else:
                s.update(yl=new_yl, togo=min(left, 100 - new_yl))
            continue

        play = _choose_play(rng, down, togo, yl, run_rates[off])
        emit(play, ctx)
        if play == 'U':
            land = yl + int(rng.normal(41, 7))
            start_drive(1 - off, 20 if land >= 100 else 100 - land)
            continue
        if play == 'F':
            if rng.random() < np.clip(1.45 - 0.018 * (117 - yl), 0.05, 0.97):
                kickoff(off)
            else:
                start_drive(1 - off, max(100 - yl, 20))
            continue

        turnover = False
        if play == 'R':
            gain = int(round(rng.normal(4.0, 4.5)))
            if rng.random() < 0.05:
                gain += int(rng.integers(12, 60))
            turnover = rng.random() < 0.012
        else:
            r = rng.random()
            if r < 0.06:
                gain = -int(rng.integers(3, 11))          # sack
            elif r < 0.09:
                gain, turnover = 0, True                  # interception
            elif r < 0.68:
                gain = max(0, int(round(rng.normal(11, 7))))
                if rng.random() < 0.07:
                    gain += int(rng.integers(15, 60))
            else:
                gain = 0                                  # incomplete

        new_yl = yl + gain
        if turnover:
            start_drive(1 - off, 100 - int(np.clip(new_yl, 1, 99)))
        elif new_yl >= 100:                               # touchdown
            emit('X', context(off, 1, 3, 97))
            kickoff(off)
        elif new_yl <= 0:                                 # safety: free kick from own 20
            kickoff(off, from_yl=20)
        elif gain >= togo:
            s.update(yl=new_yl, down=1, togo=min(10, 100 - new_yl))
        elif down == 4:                                   # turnover on downs
            start_drive(1 - off, 100 - new_yl)
        else:
            s.update(yl=new_yl, down=down + 1, togo=togo - gain)
    return rows


def write_allplays(games, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['playID', 'type', 'clock', 'token', 'context', 'text'])
        k = 0
        for game in games:
            for pid, ptype, ctx in game:
                w.writerow([k, pid, ptype, '', '', ctx, ''])
                k += 1


_CONFUSER = {'run': 'pass', 'pass': 'run', 'pat_fg': 'punt', 'punt': 'pat_fg', 'kickoff': 'punt'}


def fake_vision_proba(rng, true_class, accuracy=0.97, bad_clip_rate=0.06):
    """Softmax of a simulated video classifier (~73% accurate at the defaults)."""
    k = len(CLASSIFIER_CLASSES)
    t = CLASSIFIER_CLASSES.index(true_class)
    if rng.random() < bad_clip_rate:                      # ad / wrong camera: almost no signal
        logits = rng.normal(0, 0.6, k)
    else:
        if rng.random() < accuracy:
            pred = t
        elif rng.random() < 0.8:
            pred = CLASSIFIER_CLASSES.index(_CONFUSER[true_class])
        else:
            pred = int(rng.choice([i for i in range(k) if i != t]))
        logits = rng.normal(0, 0.7, k)
        logits[pred] += rng.uniform(0.3, 3.0)
        if pred != t:
            logits[t] += rng.uniform(0.0, 1.2)
    e = np.exp(logits - logits.max())
    return e / e.sum()
