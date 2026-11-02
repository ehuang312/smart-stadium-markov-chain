"""
Synthetic stand-ins for the real data, used only to measure the code changes.

simulate_game():      play-by-play sequences with realistic football structure
                      (downs, field position, punts, FGs, PATs, kickoffs,
                      penalties, turnovers, a run-heavy home team).
fake_vision_proba():  a noisy stand-in for the ResNet50+LSTM classifier that
                      confuses run/pass and the kicking plays, with some
                      unusable "ad" clips, roughly at the paper's accuracy level.

Neither is real data. Swap in the CBS XML sequences and the real classifier's
outputs to get real numbers (see evaluate.compare).
"""
import numpy as np

from rishav_markov_chain_from_paper import CLASSIFIER_CLASSES, make_state


def _choose_play(rng, down, togo, yl, run_rate):
    if rng.random() < 0.05:
        return 'O'  # penalty / no play
    if down == 4:
        kick_distance = 117 - yl
        go = 0.6 if togo <= 1 else 0.35 if togo <= 3 else 0.08
        if yl < 40:
            go *= 0.2
        if rng.random() >= go:
            return 'F' if kick_distance <= 53 else 'U'
    p_run = run_rate + {1: 0.05, 2: 0.0, 3: -0.1, 4: 0.0}[down]
    if togo <= 2:
        p_run += 0.2
    elif togo >= 8 and down >= 2:
        p_run -= 0.25
    return 'R' if rng.random() < np.clip(p_run, 0.05, 0.95) else 'P'


def simulate_game(rng, n_plays=135, run_rates=(0.68, 0.48)):
    """One game as a list of make_state() tuples. Team 0 is run-heavy (GT-like)."""
    states = []
    g = {'off': 0, 'down': 1, 'togo': 10, 'yl': 25}

    def change_possession(new_yl):
        g['off'] = 1 - g['off']
        g['yl'] = int(np.clip(new_yl, 1, 99))
        g['down'], g['togo'] = 1, 10

    phase = 'kickoff'
    while len(states) < n_plays:
        if phase == 'kickoff':
            states.append(make_state(1, 'K', 35, 10))
            change_possession(rng.integers(18, 40))
            phase = 'scrimmage'
            continue
        if phase == 'pat':
            states.append(make_state(1, 'X', 97, 3))
            phase = 'kickoff'
            continue

        down, togo, yl = g['down'], g['togo'], g['yl']
        play = _choose_play(rng, down, togo, yl, run_rates[g['off']])
        states.append(make_state(down, play, yl, togo))

        if play == 'U':
            land = yl + int(rng.normal(41, 7))
            change_possession(20 if land >= 100 else 100 - land)
            continue
        if play == 'F':
            if rng.random() < np.clip(1.45 - 0.018 * (117 - yl), 0.05, 0.97):
                phase = 'kickoff'
            else:
                change_possession(max(100 - yl, 20))
            continue
        if play == 'O':
            pen = 5 if rng.random() < 0.45 else -5
            g['yl'] = int(np.clip(yl + pen, 1, 99))
            g['togo'] = togo - pen
            if g['togo'] <= 0:
                g['down'], g['togo'] = 1, min(10, 100 - g['yl'])
            else:
                g['togo'] = min(g['togo'], 100 - g['yl'])
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
            change_possession(100 - int(np.clip(new_yl, 1, 99)))
            continue
        if new_yl >= 100:
            phase = 'pat'
            continue
        if new_yl <= 0:
            phase = 'kickoff'                             # safety; conceding team kicks
            continue
        g['yl'] = new_yl
        if gain >= togo:
            g['down'], g['togo'] = 1, min(10, 100 - new_yl)
        else:
            g['down'], g['togo'] = down + 1, togo - gain
            if g['down'] > 4:
                change_possession(100 - new_yl)
    return states


_CONFUSER = {'run': 'pass', 'pass': 'run', 'pat_fg': 'punt', 'punt': 'pat_fg', 'kickoff': 'punt'}


def fake_vision_proba(rng, true_class, accuracy=0.97, bad_clip_rate=0.06):
    """Softmax output of a simulated video classifier for one clip."""
    k = len(CLASSIFIER_CLASSES)
    t = CLASSIFIER_CLASSES.index(true_class)
    if rng.random() < bad_clip_rate:                      # ad / wrong camera: nearly no signal
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
            logits[t] += rng.uniform(0.0, 1.2)            # true class usually keeps some mass
    e = np.exp(logits - logits.max())
    return e / e.sum()
