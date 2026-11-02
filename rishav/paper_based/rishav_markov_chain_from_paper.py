"""
Rishav's first rewrite, written from the paper before seeing last year's code. Markov model of play sequences for the Smart Stadium play classifier (Section 4.2).

Changes from the original version:
  1. Distance-to-go is part of the state, as in the paper's full 1,680-state model.
  2. Hierarchical backoff smoothing replaces the tiny additive constant. When a
     fine-grained state has little data, the estimate leans on coarser states
     (down+type+field -> down+type -> type -> overall frequency) instead of
     going nearly uniform. This fixes the sparsity problem seen in Fig. 9.
  3. The model directly estimates P(next play type | context) rather than a
     dense 1,680 x 1,680 next-state matrix that is then marginalized. Same
     quantity, ~240x fewer parameters, and no 22 MB matrix.
  4. Optional "context" mode: condition on the CURRENT play's pre-snap down,
     distance and field position (known from the scoreboard / play-by-play
     feed) plus the previous play's type. The paper's "previous" mode is kept.
  5. Maps the 7 Markov play types onto the vision model's 5 classes
     (PAT and FG share a class; "Other" has none).
  6. A tunable conflation function, input validation, evaluation metrics,
     and save/load.
"""
from __future__ import annotations

import ast
import json

import numpy as np

DOWNS = (1, 2, 3, 4)
PLAY_TYPES = ('R', 'P', 'X', 'F', 'U', 'K', 'O')  # run, pass, PAT, field goal, punt, kickoff, other
FIELD_BUCKETS = 10                                # 10-yard buckets from the offense's own goal line
DISTANCE_EDGES = (3, 6, 9, 14, 19)                # buckets [1,3] [4,6] [7,9] [10,14] [15,19] [20,99]
N_DISTANCE_BUCKETS = len(DISTANCE_EDGES) + 1

# The vision model's 5 output classes and how the Markov play types map onto them
CLASSIFIER_CLASSES = ('run', 'pass', 'pat_fg', 'punt', 'kickoff')
DEFAULT_CLASS_MAP = {'R': 'run', 'P': 'pass', 'X': 'pat_fg', 'F': 'pat_fg',
                     'U': 'punt', 'K': 'kickoff', 'O': None}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def field_bucket(yardline: float) -> int:
    """Yards from the offense's own goal line (0-100) -> bucket 0..9."""
    if not 0 <= yardline <= 100:
        raise ValueError(f'yardline must be in [0, 100], got {yardline}')
    return min(int(yardline) // 10, FIELD_BUCKETS - 1)


def distance_bucket(yards_to_go: float) -> int:
    """Yards to a first down -> bucket 0..5 (see DISTANCE_EDGES)."""
    if yards_to_go < 1:
        raise ValueError(f'yards_to_go must be >= 1, got {yards_to_go}')
    return int(np.searchsorted(DISTANCE_EDGES, yards_to_go, side='left'))


def make_state(down: int, play_type: str, yardline: float, yards_to_go: float) -> tuple:
    """Build a (down, play_type, field_bucket, distance_bucket) state from raw values."""
    if down not in DOWNS:
        raise ValueError(f'down must be one of {DOWNS}, got {down}')
    if play_type not in PLAY_TYPES:
        raise ValueError(f'play_type must be one of {PLAY_TYPES}, got {play_type!r}')
    return (down, play_type, field_bucket(yardline), distance_bucket(yards_to_go))


def context_of(state: tuple) -> tuple:
    """Pre-snap context of a play: (down, field_bucket, distance_bucket). Drops the play type."""
    return (state[0], state[2], state[3])


def _check_state(state: tuple) -> None:
    if (len(state) != 4 or state[0] not in DOWNS or state[1] not in PLAY_TYPES
            or not 0 <= state[2] < FIELD_BUCKETS or not 0 <= state[3] < N_DISTANCE_BUCKETS):
        raise ValueError(f'invalid state {state!r}; build states with make_state()')


# ---------------------------------------------------------------------------
# Backoff estimator
# ---------------------------------------------------------------------------

class _BackoffCategorical:
    """
    Categorical distribution estimated at several levels of context, most
    specific first. Each level's counts are shrunk toward the level below it:

        p_k = (counts_k + alpha * p_{k-1}) / (n_k + alpha)

    A context seen thousands of times is driven by its own counts; one seen
    twice is mostly the coarser estimate; one never seen is exactly the coarser
    estimate. alpha sets how many observations it takes to trust a level.
    """

    def __init__(self, n_classes: int, n_levels: int, alpha: float):
        self.n_classes = n_classes
        self.alpha = alpha
        self.counts = [dict() for _ in range(n_levels)]

    def add(self, keys: list, y: int) -> None:
        for table, key in zip(self.counts, keys):
            arr = table.get(key)
            if arr is None:
                arr = table[key] = np.zeros(self.n_classes)
            arr[y] += 1

    def proba(self, keys: list) -> np.ndarray:
        p = np.full(self.n_classes, 1.0 / self.n_classes)
        for table, key in zip(reversed(self.counts), reversed(keys)):
            c = table.get(key)
            if c is not None:
                p = (c + self.alpha * p) / (c.sum() + self.alpha)
        return p

    def support(self, keys: list) -> list:
        """Number of training observations behind each level, most specific first."""
        return [int(table[key].sum()) if key in table else 0
                for table, key in zip(self.counts, keys)]


# ---------------------------------------------------------------------------
# Markov chain
# ---------------------------------------------------------------------------

class PlayMarkovChain:
    """
    Predicts the type of the next play.

    mode='previous' (paper's design): condition only on the previous play's
        (down, type, field bucket, distance bucket).
    mode='context': condition on the current play's pre-snap
        (down, field bucket, distance bucket) plus the previous play's type.
        Use this when the live play-by-play feed gives down and distance
        before the snap, which it normally does.

    Input data: a list of games, each a chronological list of states from
    make_state(). Each state is the PRE-SNAP situation plus the type of play
    that was run from it.
    """

    MODES = ('previous', 'context')

    def __init__(self, mode: str = 'context', alpha: float = 5.0, play_types: tuple = PLAY_TYPES):
        if mode not in self.MODES:
            raise ValueError(f'mode must be one of {self.MODES}, got {mode!r}')
        self.mode = mode
        self.alpha = alpha
        self.play_types = tuple(play_types)
        self._type_idx = {t: i for i, t in enumerate(self.play_types)}
        self._model = None

    # -- context keys, most specific first ---------------------------------
    def _keys(self, prev_state: tuple, current_context: tuple | None) -> list:
        d, t, fb, db = prev_state
        if self.mode == 'previous':
            return [(d, t, fb, db), (d, t, fb), (d, t), (t,), ()]
        if current_context is None:
            raise ValueError("mode='context' needs current_context=(down, field_bucket, distance_bucket)")
        cd, cfb, cdb = current_context
        return [(cd, cdb, cfb, t), (cd, cdb, cfb), (cd, cdb), (cd,), ()]

    # -- training ----------------------------------------------------------
    def fit(self, play_sequences: list) -> 'PlayMarkovChain':
        self._model = _BackoffCategorical(len(self.play_types), 5, self.alpha)
        for game in play_sequences:
            for state in game:
                _check_state(state)
            for prev, cur in zip(game, game[1:]):
                self._model.add(self._keys(prev, context_of(cur)), self._type_idx[cur[1]])
        return self

    def _require_fit(self) -> None:
        if self._model is None:
            raise RuntimeError('call fit() before predicting')

    # -- prediction --------------------------------------------------------
    def next_play_type_proba(self, prev_state: tuple, current_context: tuple | None = None) -> np.ndarray:
        """P(next play type) over self.play_types (7 types)."""
        self._require_fit()
        _check_state(prev_state)
        self._model.alpha = self.alpha
        return self._model.proba(self._keys(prev_state, current_context))

    def classifier_proba(self, prev_state: tuple, current_context: tuple | None = None,
                         classes: tuple = CLASSIFIER_CLASSES,
                         class_map: dict = DEFAULT_CLASS_MAP) -> np.ndarray:
        """Same prediction, merged onto the vision model's classes; ready to conflate."""
        return to_classes(self.next_play_type_proba(prev_state, current_context),
                          self.play_types, classes, class_map)

    def support(self, prev_state: tuple, current_context: tuple | None = None) -> list:
        """How much training data sits behind a prediction, per backoff level."""
        self._require_fit()
        return self._model.support(self._keys(prev_state, current_context))

    # -- evaluation --------------------------------------------------------
    def evaluate(self, play_sequences: list, classes: tuple = CLASSIFIER_CLASSES,
                 class_map: dict = DEFAULT_CLASS_MAP) -> dict:
        """Accuracy and log loss of next-play prediction on held-out games (5 classes)."""
        correct, losses = 0, []
        for game in play_sequences:
            for prev, cur in zip(game, game[1:]):
                label = class_map.get(cur[1])
                if label is None:
                    continue
                p = self.classifier_proba(prev, context_of(cur), classes, class_map)
                y = classes.index(label)
                correct += int(np.argmax(p) == y)
                losses.append(-np.log(max(p[y], 1e-12)))
        n = len(losses)
        return {'n': n, 'accuracy': correct / n if n else float('nan'),
                'log_loss': float(np.mean(losses)) if n else float('nan')}

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        self._require_fit()
        payload = {'mode': self.mode, 'alpha': self.alpha, 'play_types': list(self.play_types),
                   'counts': [{repr(k): v.tolist() for k, v in table.items()}
                              for table in self._model.counts]}
        with open(path, 'w') as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> 'PlayMarkovChain':
        with open(path) as f:
            payload = json.load(f)
        chain = cls(payload['mode'], payload['alpha'], tuple(payload['play_types']))
        chain._model = _BackoffCategorical(len(chain.play_types), len(payload['counts']), chain.alpha)
        chain._model.counts = [{ast.literal_eval(k): np.array(v) for k, v in table.items()}
                               for table in payload['counts']]
        return chain


# ---------------------------------------------------------------------------
# Class mapping and conflation
# ---------------------------------------------------------------------------

def to_classes(proba: np.ndarray, play_types: tuple = PLAY_TYPES,
               classes: tuple = CLASSIFIER_CLASSES, class_map: dict = DEFAULT_CLASS_MAP) -> np.ndarray:
    """Merge a distribution over play types onto classifier classes and renormalize."""
    out = np.zeros(len(classes))
    for p, t in zip(proba, play_types):
        label = class_map.get(t)
        if label is not None:
            out[classes.index(label)] += p
    total = out.sum()
    return out / total if total > 0 else np.full(len(classes), 1.0 / len(classes))


def conflate(p_vision: np.ndarray, p_markov: np.ndarray, markov_weight: float = 1.0,
             floor: float = 1e-4) -> np.ndarray:
    """
    Weighted conflation: q ∝ p_vision * p_markov ** markov_weight.

    markov_weight=1 is pure conflation (Hill & Miller). Tune it on validation
    data: <1 trusts video more, >1 trusts game context more. The floor stops
    one source's near-zero from vetoing a class the other is sure of.
    Works on single vectors or (n, k) batches.
    """
    lv = np.log(np.clip(p_vision, floor, 1.0))
    lm = np.log(np.clip(p_markov, floor, 1.0))
    z = lv + markov_weight * lm
    z = z - z.max(axis=-1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=-1, keepdims=True)


if __name__ == '__main__':
    # Smoke test on a tiny hand-built drive
    drive = [make_state(1, 'K', 35, 10), make_state(1, 'R', 25, 10), make_state(2, 'P', 29, 6),
             make_state(3, 'P', 29, 6), make_state(4, 'U', 29, 6), make_state(1, 'K', 35, 10)]
    for mode in PlayMarkovChain.MODES:
        chain = PlayMarkovChain(mode=mode).fit([drive] * 20)
        p = chain.classifier_proba(drive[3], context_of(drive[4]))
        print(f'{mode:>8}: after 3rd-down pass ->', dict(zip(CLASSIFIER_CLASSES, p.round(3))))
