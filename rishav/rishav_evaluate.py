"""
Rishav's evaluation harness. Measures what the changes in rishav_markov.py do
to accuracy, using last year's file (archive/last_year/MarkovChain.py),
unmodified, as the baseline. Data is simulated in the allplays.csv format.

Run from the repo root:
  python rishav/rishav_evaluate.py --check        # new code reproduces the original CSVs exactly + timing
  python rishav/rishav_evaluate.py --seeds 10     # accuracy comparison

Each seed simulates ~110 games (~15k plays, like the paper's 14,906), splits
them by game into train / validation / test, trains every chain on train,
tunes alpha and the conflation weight on validation, and scores test.

Test plays: every consecutive pair of plays within a game (including across
drives, since the live system always has a previous play), scored on the
vision model's 5 classes. The previous play's state comes from the official
feed, as in the paper.
"""
import argparse
import contextlib
import importlib.util
import io
import os
import shutil
import tempfile
import time
import uuid

import numpy as np
import pandas as pd

from rishav_markov import (CHAINS, CLASSIFIER_CLASSES, DEFAULT_CLASS_MAP, Markov, PlayPredictor,
                    conflate, load_plays, state_index, to_classes, transitions)
from rishav_simulate_allplays import fake_vision_proba, simulate_game, write_allplays

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = os.path.join(HERE, '..', 'archive', 'last_year', 'MarkovChain.py')  # last year's code, never modified
ORIGINAL_ATTR = {'Down': 'down_matrix', 'DownType': 'down_type_matrix',
                 'DownTypeFP': 'down_type_fp_matrix', 'DownTypeFPDist': 'down_type_fp_dist_matrix'}
WEIGHTS = np.round(np.arange(0.0, 2.01, 0.1), 2)
ALPHAS = (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500)


def run_original(csv_path, write_csvs=False, out_dir=None):
    """Run the team's file, unmodified, on csv_path (it reads ./allplays.csv)."""
    tmp = out_dir or tempfile.mkdtemp()
    shutil.copy(csv_path, os.path.join(tmp, 'allplays.csv'))
    cwd, to_csv = os.getcwd(), pd.DataFrame.to_csv
    try:
        os.chdir(tmp)
        if not write_csvs:
            pd.DataFrame.to_csv = lambda *a, **k: None       # skip writing files; doesn't change the math
        with contextlib.redirect_stdout(io.StringIO()):
            spec = importlib.util.spec_from_file_location(f'orig_{uuid.uuid4().hex}', ORIGINAL)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)                      # the file's last line builds DownTypeFP on import
            m = mod.Markov()
            t0 = time.perf_counter()
            for name in CHAINS:
                getattr(m, f'create{name}Markov')()
            m.build_seconds = time.perf_counter() - t0
    finally:
        os.chdir(cwd)
        pd.DataFrame.to_csv = to_csv
        if out_dir is None:
            shutil.rmtree(tmp, ignore_errors=True)
    return m


def simulate_split(seed, n_games=110):
    rng = np.random.default_rng(seed)
    games = [simulate_game(rng) for _ in range(n_games)]
    a, b = int(n_games * 0.64), int(n_games * 0.82)
    return rng, {'train': games[:a], 'val': games[a:b], 'test': games[b:]}


def check(seed=0):
    """New Markov(cross_drive='legacy') must give the same CSVs as the original."""
    _, split = simulate_split(seed)
    with tempfile.TemporaryDirectory() as orig_dir, tempfile.TemporaryDirectory() as new_dir:
        csv_path = os.path.join(new_dir, 'allplays.csv')
        write_allplays(split['train'], csv_path)
        orig = run_original(csv_path, write_csvs=True, out_dir=orig_dir)

        new = Markov(csv_path)
        t0 = time.perf_counter()
        for name in CHAINS:
            getattr(new, f'create{name}Markov')(out=os.path.join(new_dir, f'{name}.csv'))
        new_seconds = time.perf_counter() - t0

        print(f'{len(new.plays):,} plays after filtering\n')
        for name in CHAINS:
            a = pd.read_csv(os.path.join(orig_dir, f'{name}.csv'), index_col=0)
            b = pd.read_csv(os.path.join(new_dir, f'{name}.csv'), index_col=0)
            same_labels = list(a.columns) == list(b.columns) and list(a.index) == list(b.index)
            same_values = np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
            same_memory = np.allclose(getattr(orig, ORIGINAL_ATTR[name]), getattr(new, ORIGINAL_ATTR[name]),
                                      equal_nan=True)
            print(f'{name:<16} shape {a.shape}  labels match: {same_labels}  '
                  f'CSV values match: {same_values}  in-memory matrix match: {same_memory}')
        print(f'\nbuild time for all 4 chains: '
              f'original {orig.build_seconds:.2f}s (CSV writing skipped), new {new_seconds:.2f}s (including CSV writing)')


def labeled_pairs(plays):
    i, j = transitions(plays, 'all')
    labels = [DEFAULT_CLASS_MAP[t] for t in plays['type'].to_numpy()[j]]
    keep = np.array([lab is not None for lab in labels])
    y = np.array([CLASSIFIER_CLASSES.index(lab) for lab, k in zip(labels, keep) if k])
    return i[keep], j[keep], y


def acc(P, y):
    return float(np.mean(P.argmax(1) == y))


def logloss(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def best_weight(V, M, y):
    return float(WEIGHTS[int(np.argmax([acc(conflate(V, M, w), y) for w in WEIGHTS]))])


def run_seed(seed, vision_accuracy):
    rng, split = simulate_split(seed)
    with tempfile.TemporaryDirectory() as tmp:
        data = {}
        for part, games in split.items():
            path = os.path.join(tmp, f'{part}.csv')
            write_allplays(games, path)
            data[part] = load_plays(path)
        orig = run_original(os.path.join(tmp, 'train.csv'))

    sets = {}
    for part in ('val', 'test'):
        i, j, y = labeled_pairs(data[part])
        V = np.array([fake_vision_proba(rng, CLASSIFIER_CLASSES[k], vision_accuracy) for k in y])
        sets[part] = (data[part], i, j, y, V)
    _, _, _, y_te, V_te = sets['test']
    _, _, _, y_va, V_va = sets['val']
    res = {'vision only': {'final': acc(V_te, y_te)}}

    # --- the team's original chains (zero rows -> uniform, the most charitable reading)
    orig_val, orig_test, unseen = {}, {}, {}
    for name in ('DownType', 'DownTypeFP', 'DownTypeFPDist'):
        matrix = getattr(orig, ORIGINAL_ATTR[name])
        for part, store in (('val', orig_val), ('test', orig_test)):
            plays, i, _, _, _ = sets[part]
            rows = matrix[state_index(plays, CHAINS[name])[i], -7:]
            store[name] = to_classes(rows)
            if part == 'test':
                unseen[name] = float(np.mean(rows.sum(1) == 0))
        M = orig_test[name]
        res[f'original {name}'] = {'chain': acc(M, y_te), 'logloss': logloss(M, y_te),
                                   'unseen': unseen[name], 'final': acc(conflate(V_te, M, 1.0), y_te)}
    best = max(orig_val, key=lambda n: acc(conflate(V_va, orig_val[n], 1.0), y_va))
    base = dict(res[f'original {best}'], picked=best)
    res['A  original best chain, w=1'] = base
    w = best_weight(V_va, orig_val[best], y_va)
    res['B  + tuned conflation weight'] = dict(base, final=acc(conflate(V_te, orig_test[best], w), y_te), w=w)

    # --- new predictor, adding one change at a time
    steps = (('C  + backoff blend of the 4 chains', 'previous', 'kicks'),
             ('D  + all cross-drive transitions', 'previous', 'all'),
             ('E  + context mode', 'context', 'all'))
    for label, mode, cross in steps:
        best_fit = None
        for a in ALPHAS:
            model = PlayPredictor(mode=mode, alpha=a, cross_drive=cross).fit(data['train'])
            plays, i, j, _, _ = sets['val']
            Mv = model.classifier_proba(plays, i, j)
            if best_fit is None or logloss(Mv, y_va) < best_fit[0]:
                best_fit = (logloss(Mv, y_va), a, model, Mv)
        _, a, model, Mv = best_fit
        plays, i, j, _, _ = sets['test']
        Mt = model.classifier_proba(plays, i, j)
        w = best_weight(V_va, Mv, y_va)
        res[label] = {'chain': acc(Mt, y_te), 'logloss': logloss(Mt, y_te), 'unseen': 0.0,
                      'final': acc(conflate(V_te, Mt, w), y_te), 'alpha': a, 'w': w}
    return res


def report(runs):
    names = list(runs[0])
    base = np.array([r['A  original best chain, w=1']['final'] for r in runs])
    print(f'{len(runs)} seeds; mean ± std over seeds; held-out test games\n')
    print(f'{"variant":<38}{"chain acc":>11}{"chain logloss":>15}{"unseen rows":>13}'
          f'{"FINAL acc":>12}{"Δ vs A (pts)":>16}')
    for n in names:
        if n == 'A  original best chain, w=1':
            print('-' * 105)
        f = np.array([r[n]['final'] for r in runs])
        row = f'{n:<38}'
        if 'chain' in runs[0][n]:
            c = np.array([r[n]['chain'] for r in runs])
            ll = np.array([r[n]['logloss'] for r in runs])
            u = np.array([r[n]['unseen'] for r in runs])
            row += f'{c.mean():>8.1%} ±{c.std()*100:.1f}{ll.mean():>12.2f}   {u.mean():>10.1%}'
        else:
            row += f'{"—":>11}{"—":>15}{"—":>13}'
        d = (f - base) * 100
        row += f'{f.mean():>9.1%} ±{f.std()*100:.1f}{d.mean():>+10.1f} ±{d.std():.1f}'
        print(row)
    picks = [r['A  original best chain, w=1']['picked'] for r in runs]
    print(f'\nchain picked for A (by validation): {pd.Series(picks).value_counts().to_dict()}')
    for n in names[-3:]:
        print(f'{n}: alpha {[r[n]["alpha"] for r in runs]}, weight {[r[n]["w"] for r in runs]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='verify the new code reproduces the original CSVs')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--vision-accuracy', type=float, default=0.97, help='0.97 ≈ 73%% simulated vision accuracy')
    args = ap.parse_args()
    if args.check:
        check()
    else:
        report([run_seed(s, args.vision_accuracy) for s in range(args.seeds)])
