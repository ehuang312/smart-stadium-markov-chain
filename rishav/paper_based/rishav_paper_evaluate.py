"""
Rishav's paper-based harness: compares rishav_markov_chain_from_paper.py with the repo's v1 (versions/markovChain_v1.py) on simulated games.

Metrics (all on held-out games, 5 classifier classes, penalties excluded):
  chain_acc / chain_logloss : the Markov chain alone predicting the play type
  conflated_acc             : final system accuracy (vision conflated with chain)

Variants compared:
  vision only
  A  original code, pure conflation (weight = 1)
  B  original code, tuned conflation weight
  C  new code, paper's 'previous' mode (distance + backoff smoothing), tuned weight
  D  new code, 'context' mode (current down/distance/field known), tuned weight

Run from the repo root:  python rishav/paper_based/rishav_paper_evaluate.py --seeds 10
To use real data, call compare() with your own train/val/test game lists and
the real classifier's probability vectors for the val/test plays.
"""
import argparse
import importlib.util
import os

import numpy as np

# baseline: the repo's v1 model, loaded unmodified from versions/
_V1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'versions', 'markovChain_v1.py')
_spec = importlib.util.spec_from_file_location('markovChain_v1', _V1)
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)
OriginalChain = _v1.PlayMarkovChain

from rishav_markov_chain_from_paper import (CLASSIFIER_CLASSES, DEFAULT_CLASS_MAP, PLAY_TYPES,
                          PlayMarkovChain, conflate, context_of, to_classes)
from rishav_paper_simulate import fake_vision_proba, simulate_game

WEIGHT_GRID = np.round(np.arange(0.0, 2.01, 0.1), 2)
ALPHA_GRID = (0.5, 1, 2, 5, 10, 20, 50)


def labeled_pairs(games):
    """(prev_state, cur_state, class_index) for every play the vision model classifies."""
    out = []
    for game in games:
        for prev, cur in zip(game, game[1:]):
            label = DEFAULT_CLASS_MAP[cur[1]]
            if label is not None:
                out.append((prev, cur, CLASSIFIER_CLASSES.index(label)))
    return out


def chain_probas(predict, pairs):
    return np.array([predict(prev, cur) for prev, cur, _ in pairs])


def logloss(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def acc(P, y):
    return float(np.mean(P.argmax(1) == y))


def best_weight(V, M, y):
    scores = [acc(conflate(V, M, w), y) for w in WEIGHT_GRID]
    return float(WEIGHT_GRID[int(np.argmax(scores))])


def compare(train, val, test, vision_val, vision_test):
    """All variants on one split. vision_* are (n_plays, 5) arrays aligned with labeled_pairs()."""
    val_pairs, test_pairs = labeled_pairs(val), labeled_pairs(test)
    y_val = np.array([y for *_, y in val_pairs])
    y_test = np.array([y for *_, y in test_pairs])
    res = {'vision only': {'conflated_acc': acc(vision_test, y_test)}}

    # original code: 3-element states, dense matrix, 1e-3 smoothing
    orig = OriginalChain()
    orig.fit([[s[:3] for s in g] for g in train])
    orig_pred = lambda prev, cur: to_classes(orig.next_play_type_proba(prev[:3]), PLAY_TYPES)
    Mv, Mt = chain_probas(orig_pred, val_pairs), chain_probas(orig_pred, test_pairs)
    chain_stats = {'chain_acc': acc(Mt, y_test), 'chain_logloss': logloss(Mt, y_test)}
    res['A original, pure conflation'] = {**chain_stats, 'weight': 1.0,
                                          'conflated_acc': acc(conflate(vision_test, Mt, 1.0), y_test)}
    w = best_weight(vision_val, Mv, y_val)
    res['B original, tuned weight'] = {**chain_stats, 'weight': w,
                                       'conflated_acc': acc(conflate(vision_test, Mt, w), y_test)}

    # new code, both modes; alpha tuned on validation log loss
    for name, mode in (('C new, previous mode', 'previous'), ('D new, context mode', 'context')):
        best = None
        for a in ALPHA_GRID:
            chain = PlayMarkovChain(mode=mode, alpha=a).fit(train)
            pred = lambda prev, cur, c=chain: c.classifier_proba(prev, context_of(cur))
            Mv = chain_probas(pred, val_pairs)
            ll = logloss(Mv, y_val)
            if best is None or ll < best[0]:
                best = (ll, a, pred, Mv)
        _, a, pred, Mv = best
        Mt = chain_probas(pred, test_pairs)
        w = best_weight(vision_val, Mv, y_val)
        res[name] = {'chain_acc': acc(Mt, y_test), 'chain_logloss': logloss(Mt, y_test),
                     'alpha': a, 'weight': w,
                     'conflated_acc': acc(conflate(vision_test, Mt, w), y_test)}
    return res


def run_seed(seed, n_games, vision_accuracy):
    rng = np.random.default_rng(seed)
    games = [simulate_game(rng) for _ in range(n_games)]
    n_tr, n_va = int(n_games * 0.64), int(n_games * 0.18)
    train, val, test = games[:n_tr], games[n_tr:n_tr + n_va], games[n_tr + n_va:]
    vis = lambda gs: np.array([fake_vision_proba(rng, CLASSIFIER_CLASSES[y], vision_accuracy)
                               for *_, y in labeled_pairs(gs)])
    return compare(train, val, test, vis(val), vis(test))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--games', type=int, default=110, help='~110 games x 135 plays ≈ the paper\'s 14,906 plays')
    ap.add_argument('--vision-accuracy', type=float, default=0.97)
    args = ap.parse_args()

    runs = [run_seed(s, args.games, args.vision_accuracy) for s in range(args.seeds)]
    names = list(runs[0])
    print(f'{args.seeds} seeds, {args.games} games each, held-out test games\n')
    print(f'{"variant":<30}{"chain acc":>12}{"chain logloss":>15}{"FINAL acc":>14}{"Δ vs A":>14}')
    base = np.array([r['A original, pure conflation']['conflated_acc'] for r in runs])
    for n in names:
        fa = np.array([r[n]['conflated_acc'] for r in runs])
        row = f'{n:<30}'
        if 'chain_acc' in runs[0][n]:
            ca = np.array([r[n]['chain_acc'] for r in runs])
            cl = np.array([r[n]['chain_logloss'] for r in runs])
            row += f'{ca.mean():>9.1%} ±{ca.std()*100:.1f}{cl.mean():>12.3f}   '
        else:
            row += f'{"—":>12}{"—":>15}'
        d = (fa - base) * 100
        row += f'{fa.mean():>9.1%} ±{fa.std()*100:.1f}{d.mean():>+9.1f} ±{d.std():.1f} pts'
        print(row)
    for n in names[3:]:
        print(f'\n{n}: alpha per seed {[r[n]["alpha"] for r in runs]}, weight {[r[n]["weight"] for r in runs]}')


if __name__ == '__main__':
    main()
