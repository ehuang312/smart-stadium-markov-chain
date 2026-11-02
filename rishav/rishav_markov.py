"""
Rishav's upgrade of last year's Markov chain (archive/last_year/MarkovChain.py).
Separate from the repo's main model (markovChain.py / versions/); it does not
replace it.

Markov chains over football play states for Smart Stadium (paper Section 4.2).

Drop-in replacement for last year's Markov class. With the default
cross_drive='legacy', Markov().createDownMarkov() ... createDownTypeFPDistMarkov()
write the same CSVs (same layout, labels and values) as the original code.

What is new:
  * One parser and one generic builder instead of four copies of the same loop.
    Vectorized: all four chains build in well under a second.
  * The play-by-play is parsed and validated once. Bad downs, yard lines with
    no H/V letter, or non-numeric distances raise a clear error instead of
    silently landing in the wrong row (numpy wraps negative indices).
  * A choice of which cross-drive transitions to keep. The original keeps only
    transitions out of punts and kickoffs, so it never learns from turnovers,
    missed field goals, turnovers on downs, or (in the Down chain) any drive change.
  * PlayPredictor: P(next play type) that blends the four chains with backoff
    smoothing, so rare or unseen states don't produce empty rows. An optional
    'context' mode also conditions on the upcoming play's down, distance and
    field position, which the play-by-play feed gives before the snap.
  * Mapping to the vision model's 5 classes and weighted conflation.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

PLAY_TYPES = ('R', 'P', 'X', 'F', 'U', 'K', 'O')  # run, pass, PAT, FG, punt, kickoff, other
TYPE_ID = {t: i for i, t in enumerate(PLAY_TYPES)}
DIST_LABELS = ('1-3', '4-6', '7-9', '10-14', '15-19', '20-99')
DIST_EDGES = (3, 6, 9, 14, 19)
FP_LABELS = ('H0', 'H1', 'H2', 'H3', 'H4', 'V4', 'V3', 'V2', 'V1', 'V0')  # own 0-9 ... opponent 0-9
SIZES = {'down': 4, 'type': 7, 'fp': 10, 'dist': 6}
CHAINS = {
    'Down': ('down',),
    'DownType': ('down', 'type'),
    'DownTypeFP': ('down', 'type', 'fp'),
    'DownTypeFPDist': ('down', 'type', 'fp', 'dist'),
}
CROSS_DRIVE = ('none', 'kicks', 'all')

CLASSIFIER_CLASSES = ('run', 'pass', 'pat_fg', 'punt', 'kickoff')
DEFAULT_CLASS_MAP = {'R': 'run', 'P': 'pass', 'X': 'pat_fg', 'F': 'pat_fg',
                     'U': 'punt', 'K': 'kickoff', 'O': None}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_plays(source='allplays.csv') -> pd.DataFrame:
    """
    Read allplays.csv (or a DataFrame with playID/type/context columns) into one
    row per play with the state attributes already computed.

    Same filtering as the original: rows with no type, drive-start rows ('#')
    and penalties ('E') are dropped. Unknown play types become 'O'.
    """
    raw = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    raw = raw[raw['type'].notna()]
    raw = raw[~raw['type'].isin(['#', 'E'])].reset_index(drop=True)

    ctx = raw['context'].astype(str).str.split(',', expand=True)
    if ctx.shape[1] < 4:
        raise ValueError("context must look like 'H,1,10,H35' (possessor, down, distance, ball spot)")
    possessor, spot = ctx[0].str.strip(), ctx[3].str.strip()
    letter = spot.str[0]
    has_letter = letter.isin(['H', 'V'])
    midfield = ~has_letter & (spot == '50')
    yards = pd.to_numeric(spot.where(~has_letter, spot.str[1:]), errors='coerce')
    down = pd.to_numeric(ctx[1], errors='coerce')
    dist = pd.to_numeric(ctx[2], errors='coerce')
    drive = pd.to_numeric(raw['playID'].astype(str).str.split(',').str[0], errors='coerce')

    problems = {
        'a down that is not 1-4': ~down.isin([1, 2, 3, 4]),
        'a distance that is not a number': dist.isna(),
        'a ball spot not like H35 / V35': (~has_letter & ~midfield) | yards.isna() | (yards < 0) | (yards > 50),
        'no drive number at the start of playID': drive.isna(),
    }
    for what, mask in problems.items():
        if mask.any():
            r = int(np.flatnonzero(mask.to_numpy())[0])
            raise ValueError(f'{int(mask.sum())} play(s) with {what}; first one: '
                             f'playID={raw.playID.iloc[r]!r}, context={raw.context.iloc[r]!r}')

    own_side = ((letter == possessor) | midfield).to_numpy()
    bucket = np.minimum(yards.to_numpy().astype(int), 49) // 10
    ptype = raw['type'].where(raw['type'].isin(PLAY_TYPES), 'O')
    drive = drive.astype(int)
    return pd.DataFrame({
        'game': (drive.diff() < 0).cumsum().to_numpy(),  # drive numbers restart each game
        'drive': drive.to_numpy(),
        'possessor': possessor.to_numpy(),
        'down': down.astype(int).to_numpy(),
        'dist': dist.astype(int).to_numpy(),
        'dist_bucket': np.searchsorted(DIST_EDGES, dist.to_numpy(), side='left'),
        'fp': np.where(own_side, bucket, 9 - bucket),
        'type': ptype.to_numpy(),
        'type_id': ptype.map(TYPE_ID).to_numpy(),
    })


def transitions(plays: pd.DataFrame, cross_drive: str = 'all'):
    """
    Index pairs (i, j) of consecutive plays that count as a transition.

    'none'  : same drive only (the original Down chain)
    'kicks' : same drive, or out of a punt/kickoff into the next drive (the original other chains)
    'all'   : any consecutive plays in the same game, i.e. also after turnovers,
              missed FGs and turnovers on downs. This matches what the live
              system sees: there is always a previous play.
    """
    if cross_drive not in CROSS_DRIVE:
        raise ValueError(f'cross_drive must be one of {CROSS_DRIVE}')
    d, t = plays['drive'].to_numpy(), plays['type'].to_numpy()
    cur = np.arange(len(plays) - 1)
    nxt = cur + 1
    keep = d[cur] == d[nxt]
    if cross_drive != 'none':
        following = d[cur] + 1 == d[nxt]
        if cross_drive == 'kicks':
            following &= np.isin(t[cur], ['U', 'K'])
        keep |= following
    return cur[keep], nxt[keep]


def state_index(plays: pd.DataFrame, attrs: tuple) -> np.ndarray:
    """Row index of each play in a chain's matrix (same ordering as the original code)."""
    cols = {'down': plays['down'].to_numpy() - 1, 'type': plays['type_id'].to_numpy(),
            'fp': plays['fp'].to_numpy(), 'dist': plays['dist_bucket'].to_numpy()}
    idx = np.zeros(len(plays), dtype=int)
    for a in attrs:
        idx = idx * SIZES[a] + cols[a]
    return idx


def state_labels(attrs: tuple) -> list:
    """Row/column labels in the original CSV format."""
    labels = []
    for combo in product(*(range(SIZES[a]) for a in attrs)):
        v = dict(zip(attrs, combo))
        d, t = v.get('down', 0) + 1, PLAY_TYPES[v.get('type', 0)]
        f, di = FP_LABELS[v.get('fp', 0)], DIST_LABELS[v.get('dist', 0)]
        if attrs == CHAINS['Down']:
            labels.append(str(d))
        elif attrs == CHAINS['DownType']:
            labels.append(f'{d}{t}')
        elif attrs == CHAINS['DownTypeFP']:
            labels.append(f'{d}{t}@{f}')
        elif attrs == CHAINS['DownTypeFPDist']:
            labels.append(f'{d}&{di} {t}@{f}0')
        else:
            labels.append('|'.join(f'{a}={x}' for a, x in v.items()))
    return labels


def transition_matrix(plays: pd.DataFrame, attrs: tuple, cross_drive: str = 'kicks'):
    """
    The original table layout: normalized transition rows, then a Total column,
    then (if play type is in the state) P(next play type) for R..O.
    Returns (table, raw_counts).
    """
    s = state_index(plays, attrs)
    i, j = transitions(plays, cross_drive)
    shape = [SIZES[a] for a in attrs]
    n = int(np.prod(shape))
    counts = np.zeros((n, n))
    np.add.at(counts, (s[i], s[j]), 1)
    total = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    blocks = [probs, (total > 0).astype(float)]
    if 'type' in attrs:
        k = attrs.index('type')
        other_axes = tuple(ax + 1 for ax in range(len(attrs)) if ax != k)
        blocks.append(probs.reshape(n, *shape).sum(axis=other_axes))
    return np.hstack(blocks), counts


# ---------------------------------------------------------------------------
# Compact versions: keep only what actually happened
# ---------------------------------------------------------------------------

def transition_list(plays: pd.DataFrame, attrs: tuple, cross_drive: str = 'all') -> pd.DataFrame:
    """
    The same counts as transition_matrix, but only transitions that actually
    happened: one row per (from state, to state) with its count and probability.
    The finest chain's full matrix has 2.8 million cells and is ~99.9% zeros;
    this list keeps only the few thousand that aren't.
    """
    s = state_index(plays, attrs)
    i, j = transitions(plays, cross_drive)
    labels = np.array(state_labels(attrs))
    pairs = pd.DataFrame({'from_index': s[i], 'to_index': s[j]})
    pairs = pairs.value_counts().rename('count').reset_index().sort_values(['from_index', 'to_index'])
    pairs['probability'] = pairs['count'] / pairs.groupby('from_index')['count'].transform('sum')
    pairs.insert(1, 'from_state', labels[pairs['from_index']])
    pairs.insert(3, 'to_state', labels[pairs['to_index']])
    return pairs.reset_index(drop=True)


def next_type_table(plays: pd.DataFrame, attrs: tuple, cross_drive: str = 'all', alpha: float = 20.0) -> pd.DataFrame:
    """
    The part conflation actually uses, with no zeros: one row per state that
    occurred, its number of plays, and SMOOTHED P(next play type) for R..O.

    Smoothing backs off through the coarser chains (for the full chain:
    DownTypeFPDist -> DownTypeFP -> DownType -> Down -> overall), so no play
    type is ever exactly 0. Same math as PlayPredictor(mode='previous').
    """
    i, j = transitions(plays, cross_drive)
    y = plays['type_id'].to_numpy()[j]
    full = state_index(plays, attrs)[i]
    observed = np.unique(full)
    sizes = [SIZES[a] for a in attrs]
    k = len(PLAY_TYPES)
    probs = np.full((len(observed), k), 1.0 / k)
    for m in range(len(attrs) + 1):                        # coarsest level (nothing) to finest (all attrs)
        div = int(np.prod(sizes[m:]))                      # dropping the last attrs = integer division
        c = np.zeros((int(np.prod(sizes[:m])), k))
        np.add.at(c, (full // div, y), 1)
        oc = c[observed // div]
        probs = (oc + alpha * probs) / (oc.sum(axis=1, keepdims=True) + alpha)
    out = pd.DataFrame(probs, columns=list(PLAY_TYPES), index=np.array(state_labels(attrs))[observed])
    out.insert(0, 'plays', oc.sum(axis=1).astype(int))
    return out


# ---------------------------------------------------------------------------
# Drop-in replacement for the original class
# ---------------------------------------------------------------------------

class Markov:
    """
    Same methods and output files as the original Markov class.

    cross_drive='legacy' reproduces the original exactly (Down chain: same-drive
    transitions only; other chains: also out of punts/kickoffs). Pass 'all' to
    keep every within-game transition in all four chains.
    """

    def __init__(self, path='allplays.csv', cross_drive: str = 'legacy'):
        self.plays = load_plays(path)
        self.cross_drive = cross_drive

    def _policy(self, chain: str) -> str:
        if self.cross_drive == 'legacy':
            return 'none' if chain == 'Down' else 'kicks'
        return self.cross_drive

    def build(self, chain: str, out: str | None = None) -> pd.DataFrame:
        attrs = CHAINS[chain]
        table, _ = transition_matrix(self.plays, attrs, self._policy(chain))
        labels = state_labels(attrs)
        columns = labels + ['Total'] + (list(PLAY_TYPES) if 'type' in attrs else [])
        df = pd.DataFrame(table, columns=columns, index=labels)
        if out:
            df.to_csv(out)
        return df

    def build_compact(self, chain: str, out: str | None = None, alpha: float = 20.0) -> pd.DataFrame:
        """Only states that occurred, with smoothed P(next play type) and no zeros (e.g. 381 rows x 8 columns)."""
        df = next_type_table(self.plays, CHAINS[chain], self._policy(chain), alpha)
        if out:
            df.to_csv(out)
        return df

    def build_transition_list(self, chain: str, out: str | None = None) -> pd.DataFrame:
        """Only transitions that happened, one per row, instead of the full mostly-zero matrix."""
        df = transition_list(self.plays, CHAINS[chain], self._policy(chain))
        if out:
            df.to_csv(out, index=False)
        return df

    def createDownMarkov(self, out='Down.csv'):
        self.down_matrix = self.build('Down', out).to_numpy()

    def createDownTypeMarkov(self, out='DownType.csv'):
        self.down_type_matrix = self.build('DownType', out).to_numpy()

    def createDownTypeFPMarkov(self, out='DownTypeFP.csv'):
        self.down_type_fp_matrix = self.build('DownTypeFP', out).to_numpy()

    def createDownTypeFPDistMarkov(self, out='DownTypeFPDist.csv'):
        self.down_type_fp_dist_matrix = self.build('DownTypeFPDist', out).to_numpy()


# ---------------------------------------------------------------------------
# Prediction with backoff smoothing
# ---------------------------------------------------------------------------

class PlayPredictor:
    """
    Predicts the type of the next play, P(type | context), over PLAY_TYPES.

    mode='previous' (the paper's design) conditions on the previous play's state.
        The backoff levels are exactly the team's four chains:
        DownTypeFPDist -> DownTypeFP -> DownType -> Down -> overall frequency.
    mode='context' conditions on the upcoming play's down, distance and field
        position (from the play-by-play feed, known before the snap) plus the
        previous play's type.

    Each level is shrunk toward the coarser one:
        p_k = (counts_k + alpha * p_{k-1}) / (n_k + alpha)
    so a well-populated state uses its own counts, a rare one leans on the
    coarser chains, and an unseen one falls back to them entirely.
    """

    MODES = ('previous', 'context')

    def __init__(self, mode: str = 'context', alpha: float = 5.0, cross_drive: str = 'all'):
        if mode not in self.MODES:
            raise ValueError(f'mode must be one of {self.MODES}')
        if cross_drive not in CROSS_DRIVE:
            raise ValueError(f'cross_drive must be one of {CROSS_DRIVE}')
        self.mode, self.alpha, self.cross_drive = mode, alpha, cross_drive
        self.counts = None

    def _fields(self, plays: pd.DataFrame, i: np.ndarray, j: np.ndarray):
        """Conditioning variables, most important first, with their sizes."""
        p = plays
        if self.mode == 'previous':
            return [(p['down'].to_numpy()[i] - 1, 4), (p['type_id'].to_numpy()[i], 7),
                    (p['fp'].to_numpy()[i], 10), (p['dist_bucket'].to_numpy()[i], 6)]
        return [(p['down'].to_numpy()[j] - 1, 4), (p['dist_bucket'].to_numpy()[j], 6),
                (p['fp'].to_numpy()[j], 10), (p['type_id'].to_numpy()[i], 7)]

    @staticmethod
    def _level_codes(fields):
        """Level m uses the first m fields; returns [(codes, n_keys)] for m = 0..len."""
        codes, n = np.zeros(len(fields[0][0]), dtype=int), 1
        out = [(codes.copy(), 1)]
        for values, size in fields:
            codes, n = codes * size + values, n * size
            out.append((codes.copy(), n))
        return out

    def fit(self, plays: pd.DataFrame) -> 'PlayPredictor':
        i, j = transitions(plays, self.cross_drive)
        y = plays['type_id'].to_numpy()[j]
        self.counts = []                                   # per level: (situations seen, their counts), no empty rows
        for codes, _ in self._level_codes(self._fields(plays, i, j)):
            keys, row = np.unique(codes, return_inverse=True)
            c = np.zeros((len(keys), len(PLAY_TYPES)))
            np.add.at(c, (row, y), 1)
            self.counts.append((keys, c))
        return self

    def predict_proba(self, plays: pd.DataFrame, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """(len(i), 7) probabilities for play j given the previous play i."""
        if self.counts is None:
            raise RuntimeError('call fit() first')
        p = np.full((len(i), len(PLAY_TYPES)), 1.0 / len(PLAY_TYPES))
        for (codes, _), (keys, table) in zip(self._level_codes(self._fields(plays, i, j)), self.counts):
            pos = np.minimum(np.searchsorted(keys, codes), len(keys) - 1)   # where each situation would be stored
            seen = keys[pos] == codes                                       # False = never seen at this level
            c = np.where(seen[:, None], table[pos], 0.0)                    # unseen situations count as 0 plays
            p = (c + self.alpha * p) / (c.sum(axis=1, keepdims=True) + self.alpha)
        return p

    def classifier_proba(self, plays, i, j) -> np.ndarray:
        """Same prediction on the vision model's 5 classes, ready to conflate."""
        return to_classes(self.predict_proba(plays, i, j))

    def predict_next(self, prev_context: str, prev_type: str, upcoming_context: str | None = None) -> dict:
        """
        Live use with raw feed strings, e.g.
            predict_next('H,3,6,H42', 'P', upcoming_context='H,4,6,H42')
        Returns {class: probability} over CLASSIFIER_CLASSES.
        """
        if self.mode == 'context' and upcoming_context is None:
            raise ValueError("mode='context' needs upcoming_context (the next play's down/distance/spot)")
        rows = pd.DataFrame({'playID': ['1', '1'], 'type': [prev_type, 'O'],
                             'context': [prev_context, upcoming_context or prev_context]})
        p = self.classifier_proba(load_plays(rows), np.array([0]), np.array([1]))[0]
        return dict(zip(CLASSIFIER_CLASSES, p.round(4)))


# ---------------------------------------------------------------------------
# Class mapping and conflation
# ---------------------------------------------------------------------------

def to_classes(proba: np.ndarray, class_map: dict = DEFAULT_CLASS_MAP,
               classes: tuple = CLASSIFIER_CLASSES) -> np.ndarray:
    """Merge (n, 7) play-type probabilities onto the classifier's classes. Empty rows -> uniform."""
    proba = np.atleast_2d(proba)
    m = np.zeros((len(PLAY_TYPES), len(classes)))
    for t, label in class_map.items():
        if label is not None:
            m[TYPE_ID[t], classes.index(label)] = 1
    out = proba @ m
    total = out.sum(axis=1, keepdims=True)
    return np.divide(out, total, out=np.full_like(out, 1.0 / len(classes)), where=total > 0)


def conflate(p_vision: np.ndarray, p_markov: np.ndarray, markov_weight: float = 1.0,
             floor: float = 1e-4) -> np.ndarray:
    """
    Weighted conflation: q ∝ p_vision * p_markov ** markov_weight.
    markov_weight=1 is pure conflation; tune it on validation data. The floor
    stops a zero from one source vetoing a class the other source is sure of.
    """
    z = np.log(np.clip(p_vision, floor, 1)) + markov_weight * np.log(np.clip(p_markov, floor, 1))
    z -= z.max(axis=-1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=-1, keepdims=True)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'allplays.csv'
    m = Markov(path)
    for name in CHAINS:
        getattr(m, f'create{name}Markov')()
        print(f'wrote {name}.csv')
    predictor = PlayPredictor(mode='context').fit(m.plays)
    print('3rd-down pass at own 42 -> 4th and 6:',
          predictor.predict_next('H,3,6,H42', 'P', upcoming_context='H,4,6,H42'))
