import json
from bisect import bisect_right
from collections import Counter, namedtuple
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_VERSION = '2.2'


PLAY_TYPES = ('R', 'P', 'F', 'U', 'O')  # run, pass, field goal, punt, other -- what the model predicts
PREV_PLAY_TYPES = ('S', 'R', 'P', 'F', 'U', 'K', 'O')  # 'S' = start of game, 'K' = kickoff

LATE_GAME_SECONDS = 480    # last 8 minutes of the 4th quarter
END_OF_HALF_SECONDS = 120  # last 2 minutes of the 1st half
RUN_RATE_EDGES = (0.45, 0.60)  # team tendency: pass-heavy < 45% runs <= balanced < 60% <= run-heavy


def team_run_rates(plays) -> dict:
    '''
    each offense's share of runs among its run + pass plays

    Args:
        plays: play dicts (or a DataFrame) with offense and play_type fields

    Returns:
        dict of offense -> run rate
    '''
    runs, total = {}, {}
    for play in _records(plays):
        if play['play_type'] in ('R', 'P'):
            team = play['offense']
            total[team] = total.get(team, 0) + 1
            runs[team] = runs.get(team, 0) + (play['play_type'] == 'R')
    return {team: runs[team] / total[team] for team in total}


def prefilter_plays(plays: pd.DataFrame, report: bool = False):
    '''
    remove or repair plays whose pre-snap fields are impossible, before fit() / predict_proba()

    Kickoffs pass through untouched (their down and distance are placeholders). For every
    other play, in this order:
        1. down not 1-4                  -> dropped (e.g. touchdowns recorded as down 0)
        2. yards_to_goal not 1-99        -> dropped (a snap can't be on a goal line)
        3. distance < 1, within 10 yards -> distance = yards_to_goal (goal-to-go with the
           of the goal line                 distance left blank)
        4. distance < 1, anywhere else   -> dropped (the real distance can't be recovered)
        5. distance > yards_to_goal      -> distance = yards_to_goal (the line to gain can't
                                            be past the goal line)
    prev_play_type is left as loaded, so it still names the play that really came before.

    Args:
        plays: DataFrame from playByPlay.load_cfb_pbp
        report: also return how many plays each rule touched

    Returns:
        cleaned copy of plays (down as int), or (plays, counts) if report is True
    '''
    plays = plays.copy()
    scrimmage = plays.play_type != 'K'
    bad_down = scrimmage & ~plays.down.isin([1, 2, 3, 4])
    bad_spot = scrimmage & ~bad_down & ~plays.yards_to_goal.between(1, 99)
    keep = ~bad_down & ~bad_spot
    no_distance = keep & scrimmage & ~(plays.distance >= 1)
    goal_to_go = no_distance & (plays.yards_to_goal <= 10)
    lost_distance = no_distance & ~goal_to_go
    keep &= ~lost_distance
    plays.loc[goal_to_go, 'distance'] = plays.loc[goal_to_go, 'yards_to_goal']
    past_goal = keep & scrimmage & (plays.distance > plays.yards_to_goal)
    plays.loc[past_goal, 'distance'] = plays.loc[past_goal, 'yards_to_goal']

    plays = plays[keep].reset_index(drop=True)
    plays['down'] = plays.down.astype(int)
    if not report:
        return plays
    counts = {'dropped: down not 1-4': int(bad_down.sum()),
              'dropped: yards_to_goal not 1-99': int(bad_spot.sum()),
              'dropped: no distance outside goal-to-go': int(lost_distance.sum()),
              'repaired: goal-to-go distance': int(goal_to_go.sum()),
              'repaired: distance past the goal line': int(past_goal.sum())}
    return plays, counts


def _records(plays):
    '''play dicts from a DataFrame (or pass a list of dicts through); much faster than to_dict('records')'''
    if not hasattr(plays, 'columns'):
        return plays
    columns = list(plays.columns)
    return [dict(zip(columns, row)) for row in zip(*(plays[c].tolist() for c in columns))]


def _game_situation(play, model):
    late = play['half'] == 2 and play['period'] <= 4 and play['seconds_left_half'] <= LATE_GAME_SECONDS
    if late and play['score_diff'] < 0:
        return 1
    if late and play['score_diff'] > 0:
        return 2
    if play['half'] == 1 and play['seconds_left_half'] <= END_OF_HALF_SECONDS:
        return 3
    return 0


def _team_tendency(play, model):
    rate = model.run_rates.get(play['offense'])
    return 1 if rate is None else bisect_right(RUN_RATE_EDGES, rate)


# a marker turns one raw pre-snap field (or a few) into a small number of buckets
Marker = namedtuple('Marker', ['labels', 'bucket'])

MARKERS = {
    'down': Marker(('1st', '2nd', '3rd', '4th'),
                   lambda play, model: min(4, max(1, int(play['down']))) - 1),
    'distance': Marker(('1', '2-3', '4-6', '7-10', '11+'),
                       lambda play, model: bisect_right((2, 4, 7, 11), play['distance'])),
    'field_zone': Marker(('red zone', 'opp 21-35', 'opp 36-49', 'own side', 'own deep (inside 20)'),
                         lambda play, model: bisect_right((21, 36, 50, 80), play['yards_to_goal'])),
    'team_tendency': Marker(('pass-heavy', 'balanced', 'run-heavy'), _team_tendency),
    'game_situation': Marker(('normal', 'trailing late', 'leading late', 'end of 1st half'), _game_situation),
    'prev_play_type': Marker(PREV_PLAY_TYPES,
                             lambda play, model: PREV_PLAY_TYPES.index(play['prev_play_type'])),
}

# most informative first: back-off drops markers from the end of this list
DEFAULT_MARKERS = ('down', 'distance', 'field_zone', 'team_tendency', 'game_situation')


class PlayMarkovChain:
    '''
    predicts the type of the upcoming play from its pre-snap situation, following the
    play-prediction design in Section 4.2 of the Smart Stadium paper

    Each play's context is a tuple of marker buckets (down, distance to go, field zone,
    offense run tendency, game situation by default -- see MARKERS). Counts of play types
    are kept for every context and for each coarser prefix of it, and predictions back off
    toward the coarser context when a context has few plays, so rare situations still get
    sensible probabilities without growing the state space.

    Adding 'prev_play_type' to the markers restores the classic Markov dependence on the
    previous play; on 2017 data it adds almost nothing once the pre-snap situation is known.
    The original (down, play_type, field_bucket) transition-matrix version is in
    archive/v1_markov_chain/.
    '''

    def __init__(self, markers: tuple = DEFAULT_MARKERS, play_types: tuple = PLAY_TYPES,
                 backoff_strength: float = 25.0):
        unknown = set(markers) - set(MARKERS)
        if unknown:
            raise ValueError(f'unknown markers {unknown}; choose from {list(MARKERS)}')
        self.markers = tuple(markers)
        self.play_types = tuple(play_types)
        self.backoff_strength = backoff_strength
        self.run_rates = {}
        self.counts = None  # set by fit(): one dict per back-off level, context prefix -> counts per play type
        self.metadata = {}  # free-form info saved with the model, e.g. which seasons it was trained on

    @property
    def n_contexts(self) -> int:
        '''number of possible contexts (product of each marker's bucket count)'''
        return int(np.prod([len(MARKERS[m].labels) for m in self.markers]))

    def context(self, play: dict) -> tuple:
        '''bucket indices of a play's pre-snap situation, one per marker'''
        return tuple(MARKERS[m].bucket(play, self) for m in self.markers)

    def describe(self, play: dict) -> dict:
        '''human-readable context of a play, marker -> bucket label'''
        return {m: MARKERS[m].labels[b] for m, b in zip(self.markers, self.context(play))}

    def fit(self, plays, run_rates: dict = None) -> 'PlayMarkovChain':
        '''
        count play types per context from a corpus of plays

        Args:
            plays: play dicts (or a DataFrame) in the format returned by playByPlay.load_cfb_pbp,
                   ideally passed through prefilter_plays first;
                   plays whose type isn't in self.play_types (e.g. kickoffs) are skipped
            run_rates: offense -> run rate used by the team_tendency marker; defaults to
                       team_run_rates(plays). Pass rates from a previous season to avoid
                       using the season you're predicting.

        Returns:
            self
        '''
        plays = _records(plays)
        self.run_rates = team_run_rates(plays) if run_rates is None else run_rates
        type_idx = {t: i for i, t in enumerate(self.play_types)}
        pair_counts = Counter((self.context(play), type_idx[play['play_type']])
                              for play in plays if play['play_type'] in type_idx)

        # roll each (context, play type) count up into every coarser prefix of the context
        self.counts = [{} for _ in range(len(self.markers) + 1)]
        for (ctx, y), n in pair_counts.items():
            for level, level_counts in enumerate(self.counts):
                key = ctx[:level]
                if key not in level_counts:
                    level_counts[key] = np.zeros(len(self.play_types))
                level_counts[key][y] += n
        return self

    def next_play_type_proba(self, play: dict, play_types: tuple = None) -> np.ndarray:
        '''
        P(play_type | pre-snap situation) for the upcoming play -- this is the vector that
        gets conflated with a play-type classifier's output (see conflation.py)

        Starts from the overall play-type rates and moves toward each finer context in turn:
        p = (counts + k * p_coarser) / (total + k), with k = self.backoff_strength.

        Args:
            play: dict with the pre-snap fields the markers use (down, distance, yards_to_goal,
                  offense, half, period, seconds_left_half, score_diff, and prev_play_type
                  if that marker is used)
            play_types: subset/order of play types to return probabilities for
                        (e.g. the classifier's class list, if it differs from self.play_types)

        Returns:
            probability vector over play_types, renormalized to sum to 1
        '''
        return self._select(self._context_proba(self.context(play)), play_types)

    def predict_proba(self, plays, play_types: tuple = None) -> np.ndarray:
        '''next_play_type_proba for many plays, stacked into an (n_plays x n_play_types) array'''
        cache = {}
        rows = []
        for play in _records(plays):
            ctx = self.context(play)
            if ctx not in cache:
                cache[ctx] = self._select(self._context_proba(ctx), play_types)
            rows.append(cache[ctx])
        return np.array(rows)

    def _context_proba(self, ctx: tuple) -> np.ndarray:
        if self.counts is None:
            raise RuntimeError('call fit() before predicting')
        base = self.counts[0][()] + 1  # Laplace smoothing on the overall rates
        proba = base / base.sum()
        for level in range(1, len(self.markers) + 1):
            n = self.counts[level].get(ctx[:level])
            if n is not None:
                proba = (n + self.backoff_strength * proba) / (n.sum() + self.backoff_strength)
        return proba

    def situation_table(self, play_types: tuple = None) -> pd.DataFrame:
        '''
        predicted probabilities for every possible context, one row per context

        Columns are the markers (bucket labels), n_plays (training plays in exactly that
        context; 0 means the prediction comes entirely from coarser contexts), one probability
        column per play type, and predicted (the most likely play type). The team_tendency
        rows don't depend on which team: every team in a bucket gets the same prediction.
        '''
        if self.counts is None:
            raise RuntimeError('call fit() before predicting')
        play_types = play_types or self.play_types
        finest = self.counts[len(self.markers)]
        rows = []
        for ctx in product(*(range(len(MARKERS[m].labels)) for m in self.markers)):
            proba = self._select(self._context_proba(ctx), play_types)
            n = finest.get(ctx)
            rows.append({**{m: MARKERS[m].labels[b] for m, b in zip(self.markers, ctx)},
                         'n_plays': 0 if n is None else int(n.sum()),
                         **dict(zip(play_types, proba)),
                         'predicted': play_types[int(proba.argmax())]})
        return pd.DataFrame(rows)

    def save(self, path, metadata: dict = None) -> Path:
        '''
        write the trained model to a JSON file that PlayMarkovChain.load can read back

        Saves the markers, play types, back-off strength, team run rates and every count,
        so a loaded model gives exactly the same predictions without the play-by-play data.

        Args:
            path: file to write (parent folders are created)
            metadata: optional extra info to store with it, e.g. {'seasons': [2017]}

        Returns:
            the path written
        '''
        if self.counts is None:
            raise RuntimeError('call fit() before saving')
        state = {
            'format': 'PlayMarkovChain',
            'model_version': MODEL_VERSION,
            'metadata': metadata or {},
            'markers': list(self.markers),
            'play_types': list(self.play_types),
            'backoff_strength': self.backoff_strength,
            'run_rates': self.run_rates,
            'counts': [{'contexts': [list(ctx) for ctx in level],
                        'counts': [n.tolist() for n in level.values()]} for level in self.counts],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding='utf-8')
        return path

    @classmethod
    def load(cls, path) -> 'PlayMarkovChain':
        '''
        read a model written by save(); its metadata is on model.metadata

        Markers registered at runtime (extra entries added to MARKERS) must be registered
        again before loading a model that uses them.
        '''
        state = json.loads(Path(path).read_text(encoding='utf-8'))
        if state.get('format') != 'PlayMarkovChain':
            raise ValueError(f'{path} is not a saved PlayMarkovChain')
        model = cls(tuple(state['markers']), tuple(state['play_types']), state['backoff_strength'])
        model.run_rates = state['run_rates']
        model.counts = [{tuple(ctx): np.array(n, dtype=float) for ctx, n in zip(level['contexts'], level['counts'])}
                        for level in state['counts']]
        model.metadata = state['metadata']
        return model

    def _select(self, proba: np.ndarray, play_types: tuple = None) -> np.ndarray:
        play_types = play_types or self.play_types
        proba = np.array([proba[self.play_types.index(t)] for t in play_types])
        return proba / proba.sum()


if __name__ == '__main__':
    pass
