'''
train the Markov chain and write its outputs to disk

    python exportOutputs.py                     # 2017 season
    python exportOutputs.py --seasons 2016 2017

Writes
    outputs/predictions_<seasons>.csv      every play's 5 probabilities, predicted vs actual type.
                                           Out-of-fold: 5-fold cross-validation by game, so each
                                           play is predicted by a model that never saw its game.
    outputs/situation_table_<seasons>.csv  the probabilities for all 1,200 situations, from the
                                           model trained on every play
    models/markov_v<version>_<seasons>.json  that trained model; reload with PlayMarkovChain.load
'''
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from markovChain import PlayMarkovChain, prefilter_plays, PLAY_TYPES, MARKERS, MODEL_VERSION
from playByPlay import download_cfb_pbp, load_cfb_pbp

ROOT = Path(__file__).resolve().parent
PROB_COLUMNS = {'R': 'prob_run', 'P': 'prob_pass', 'F': 'prob_field_goal', 'U': 'prob_punt', 'O': 'prob_other'}
PLAY_COLUMNS = ['season', 'week', 'game_id', 'play_number', 'offense', 'defense', 'period', 'half',
                'seconds_left_half', 'down', 'distance', 'yards_to_goal', 'score_diff', 'prev_play_type']
N_FOLDS = 5


def out_of_fold_predictions(plays: pd.DataFrame) -> pd.DataFrame:
    '''predict every scrimmage play with a model trained on the other folds' games'''
    games = plays.game_id.unique()
    fold = plays.game_id.map(dict(zip(games, np.random.default_rng(0).permutation(len(games)) % N_FOLDS)))
    target = plays.play_type.isin(PLAY_TYPES)
    parts = []
    for f in range(N_FOLDS):
        test = plays[(fold == f) & target]
        model = PlayMarkovChain().fit(plays[fold != f])
        proba = model.predict_proba(test)
        part = test[PLAY_COLUMNS].copy()
        situations = pd.DataFrame([model.describe(play) for play in test.to_dict('records')],
                                  index=test.index).add_suffix('_bucket')  # e.g. down_bucket = '3rd'
        part = part.join(situations)
        part[list(PROB_COLUMNS.values())] = proba.round(6)
        part['predicted'] = np.array(PLAY_TYPES)[proba.argmax(1)]
        part['actual'] = test.play_type
        part['correct'] = part.predicted == part.actual
        part['fold'] = f
        parts.append(part)
    return pd.concat(parts).sort_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seasons', type=int, nargs='+', default=[2017])
    parser.add_argument('--outputs', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--models', type=Path, default=ROOT / 'models')
    args = parser.parse_args()
    tag = '_'.join(map(str, args.seasons))

    paths = [download_cfb_pbp(s, ROOT / 'data' / 'cfb_pbp') for s in args.seasons]
    plays, prefilter_counts = prefilter_plays(load_cfb_pbp(paths), report=True)
    print(f'{len(plays)} plays from {plays.game_id.nunique()} games after prefiltering')

    predictions = out_of_fold_predictions(plays)
    args.outputs.mkdir(parents=True, exist_ok=True)
    predictions_path = args.outputs / f'predictions_{tag}.csv'
    predictions.to_csv(predictions_path, index=False)
    gt = predictions.offense == 'Georgia Tech'
    print(f'wrote {predictions_path} ({len(predictions)} plays; out-of-fold accuracy '
          f'{predictions.correct.mean():.1%}, Georgia Tech {predictions.correct[gt].mean():.1%})')

    model = PlayMarkovChain().fit(plays)
    table = model.situation_table().rename(columns=PROB_COLUMNS).round(6)
    table_path = args.outputs / f'situation_table_{tag}.csv'
    table.to_csv(table_path, index=False)
    print(f'wrote {table_path} ({len(table)} situations, {(table.n_plays > 0).sum()} seen in training)')

    model_path = model.save(args.models / f'markov_v{MODEL_VERSION.replace(".", "_")}_{tag}.json', metadata={
        'seasons': args.seasons, 'n_plays': len(plays), 'n_games': int(plays.game_id.nunique()),
        'trained_on': date.today().isoformat(), 'prefilter': prefilter_counts,
        'markers': {m: list(MARKERS[m].labels) for m in model.markers}})
    print(f'wrote {model_path} ({model_path.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
