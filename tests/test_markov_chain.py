import unittest

import sys
import os
import numpy as np
import pandas as pd
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_dir)
from markovChain import PlayMarkovChain, MARKERS, DEFAULT_MARKERS, PLAY_TYPES, team_run_rates, prefilter_plays

REAL_DATA = os.path.join(root_dir, 'data', 'cfb_pbp', 'play_by_play_2017.parquet')


def play(**fields):
    '''a pre-snap situation with neutral defaults: 1st & 10 at own 25, 1st quarter, tied'''
    situation = {'offense': 'Team A', 'down': 1, 'distance': 10, 'yards_to_goal': 75, 'score_diff': 0,
                 'half': 1, 'period': 1, 'seconds_left_half': 1500, 'play_type': 'R', 'prev_play_type': 'R'}
    situation.update(fields)
    return situation


def corpus():
    '''small training set: runs on early downs, passes on 3rd & long, punts on 4th & long'''
    return ([play(play_type='R')] * 6 + [play(play_type='P')] * 4 +
            [play(down=3, distance=9, play_type='P')] * 8 + [play(down=3, distance=9, play_type='R')] * 2 +
            [play(down=4, distance=9, play_type='U')] * 5 + [play(play_type='K')] * 3)


class TestMarkers(unittest.TestCase):

    def setUp(self):
        self.model = PlayMarkovChain()
        self.model.run_rates = {'Run U': 0.80, 'Pass U': 0.30, 'Edge Low': 0.45, 'Edge High': 0.60}

    def bucket(self, marker, **fields):
        return MARKERS[marker].bucket(play(**fields), self.model)

    def test_down(self):
        '''TEST 1: down buckets, clipped to 1-4'''
        for down, expected in {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3}.items():
            with self.subTest(down=down):
                self.assertEqual(self.bucket('down', down=down), expected)

    def test_distance(self):
        '''TEST 2: distance buckets 1 / 2-3 / 4-6 / 7-10 / 11+'''
        for distance, expected in {1: 0, 2: 1, 3: 1, 4: 2, 6: 2, 7: 3, 10: 3, 11: 4, 25: 4}.items():
            with self.subTest(distance=distance):
                self.assertEqual(self.bucket('distance', distance=distance), expected)

    def test_field_zone(self):
        '''TEST 3: field zone edges at the 20, 35, 50 and own 20'''
        for ytg, expected in {1: 0, 20: 0, 21: 1, 35: 1, 36: 2, 49: 2, 50: 3, 79: 3, 80: 4, 99: 4}.items():
            with self.subTest(yards_to_goal=ytg):
                self.assertEqual(self.bucket('field_zone', yards_to_goal=ytg), expected)
        self.assertEqual(MARKERS['field_zone'].labels[0], 'red zone')

    def test_game_situation(self):
        '''TEST 4: score and clock combine into one game-situation marker'''
        cases = [
            ('1st quarter', dict(), 'normal'),
            ('trailing late', dict(half=2, period=4, seconds_left_half=300, score_diff=-7), 'trailing late'),
            ('leading late', dict(half=2, period=4, seconds_left_half=300, score_diff=3), 'leading late'),
            ('tied late', dict(half=2, period=4, seconds_left_half=300, score_diff=0), 'normal'),
            ('trailing early in 4th', dict(half=2, period=4, seconds_left_half=600, score_diff=-7), 'normal'),
            ('end of 1st half', dict(half=1, period=2, seconds_left_half=90), 'end of 1st half'),
            ('overtime', dict(half=2, period=5, seconds_left_half=0, score_diff=-7), 'normal'),
        ]
        labels = MARKERS['game_situation'].labels
        for name, fields, expected in cases:
            with self.subTest(case=name):
                self.assertEqual(labels[self.bucket('game_situation', **fields)], expected)

    def test_team_tendency(self):
        '''TEST 5: run-rate buckets, with unknown teams treated as balanced'''
        labels = MARKERS['team_tendency'].labels
        cases = {'Run U': 'run-heavy', 'Pass U': 'pass-heavy', 'Edge Low': 'balanced',
                 'Edge High': 'run-heavy', 'Never Seen U': 'balanced'}
        for team, expected in cases.items():
            with self.subTest(team=team):
                self.assertEqual(labels[self.bucket('team_tendency', offense=team)], expected)


class TestPlayMarkovChain(unittest.TestCase):

    def test_bad_markers_and_unfitted_model(self):
        '''TEST 1: unknown markers and predicting before fit() raise errors'''
        with self.assertRaises(ValueError):
            PlayMarkovChain(markers=('down', 'weather'))
        with self.assertRaises(RuntimeError):
            PlayMarkovChain().next_play_type_proba(play())

    def test_n_contexts(self):
        '''TEST 2: the context count is the product of the markers' bucket counts'''
        self.assertEqual(PlayMarkovChain().n_contexts, 4 * 5 * 5 * 3 * 4)
        self.assertEqual(PlayMarkovChain(markers=DEFAULT_MARKERS + ('prev_play_type',)).n_contexts, 1200 * 7)
        self.assertEqual(PlayMarkovChain(markers=()).n_contexts, 1)

    def test_probabilities_are_valid(self):
        '''TEST 3: seen and unseen contexts both give positive probabilities that sum to 1'''
        model = PlayMarkovChain().fit(corpus())
        situations = {'seen': play(), 'unseen': play(down=2, distance=3, yards_to_goal=10, offense='Nobody')}
        for name, situation in situations.items():
            with self.subTest(context=name):
                proba = model.next_play_type_proba(situation)
                self.assertEqual(len(proba), len(PLAY_TYPES))
                self.assertAlmostEqual(proba.sum(), 1.0)
                self.assertTrue((proba > 0).all())

    def test_learns_situations(self):
        '''TEST 4: the most likely play type follows the training data's situations'''
        model = PlayMarkovChain(backoff_strength=1.0).fit(corpus())
        cases = [(play(), 'R'), (play(down=3, distance=9), 'P'), (play(down=4, distance=9), 'U')]
        for situation, expected in cases:
            with self.subTest(down=situation['down']):
                self.assertEqual(PLAY_TYPES[model.next_play_type_proba(situation).argmax()], expected)

    def test_zero_backoff_gives_empirical_rates(self):
        '''TEST 5: with no back-off, a seen context returns its raw play-type rates'''
        model = PlayMarkovChain(backoff_strength=0.0).fit(corpus())
        np.testing.assert_allclose(model.next_play_type_proba(play()), [0.6, 0.4, 0, 0, 0])

    def test_unseen_context_backs_off(self):
        '''TEST 6: an unseen finest context falls back to the next-coarser context'''
        full = PlayMarkovChain().fit(corpus())
        coarser = PlayMarkovChain(markers=DEFAULT_MARKERS[:-1]).fit(corpus())
        late = play(half=2, period=4, seconds_left_half=200, score_diff=-3)  # game situation never seen in corpus
        np.testing.assert_allclose(full.next_play_type_proba(late), coarser.next_play_type_proba(late))

    def test_large_counts_dominate_backoff(self):
        '''TEST 7: with many plays in a context, its own rates outweigh the back-off prior'''
        plays = [play(play_type='R')] * 700 + [play(play_type='P')] * 300 + [play(down=3, play_type='P')] * 1000
        proba = PlayMarkovChain().fit(plays).next_play_type_proba(play())
        self.assertAlmostEqual(proba[PLAY_TYPES.index('R')], 0.7, delta=0.02)

    def test_skips_kickoffs(self):
        '''TEST 8: kickoffs are not counted, since their down/distance are placeholders'''
        model = PlayMarkovChain().fit(corpus())
        self.assertEqual(model.counts[0][()].sum(), len(corpus()) - 3)

    def test_play_types_subset(self):
        '''TEST 9: asking for a subset reorders and renormalizes'''
        model = PlayMarkovChain().fit(corpus())
        full = model.next_play_type_proba(play())
        subset = model.next_play_type_proba(play(), play_types=('P', 'R'))
        expected = np.array([full[PLAY_TYPES.index('P')], full[PLAY_TYPES.index('R')]])
        np.testing.assert_allclose(subset, expected / expected.sum())

    def test_team_run_rates(self):
        '''TEST 10: run rates use only runs and passes; fit() can take them from elsewhere'''
        plays = [play(offense='A', play_type='R')] * 3 + [play(offense='A', play_type='P'),
                 play(offense='A', play_type='U'), play(offense='B', play_type='P')]
        self.assertEqual(team_run_rates(plays), {'A': 0.75, 'B': 0.0})
        self.assertEqual(PlayMarkovChain().fit(plays).run_rates, {'A': 0.75, 'B': 0.0})
        self.assertEqual(PlayMarkovChain().fit(plays, run_rates={'A': 0.5}).run_rates, {'A': 0.5})

    def test_dataframe_and_dicts_agree(self):
        '''TEST 11: fitting/predicting on a DataFrame matches lists of dicts'''
        from_dicts = PlayMarkovChain().fit(corpus())
        from_frame = PlayMarkovChain().fit(pd.DataFrame(corpus()))
        situations = [play(), play(down=3, distance=9), play(down=2, distance=5, yards_to_goal=30)]
        stacked = np.array([from_dicts.next_play_type_proba(s) for s in situations])
        np.testing.assert_allclose(from_frame.predict_proba(pd.DataFrame(situations)), stacked)
        np.testing.assert_allclose(from_dicts.predict_proba(situations), stacked)

    def test_prev_play_type_marker(self):
        '''TEST 12: the optional previous-play marker separates otherwise identical situations'''
        plays = [play(prev_play_type='R', play_type='R')] * 10 + [play(prev_play_type='P', play_type='P')] * 10
        model = PlayMarkovChain(markers=DEFAULT_MARKERS + ('prev_play_type',)).fit(plays)
        after_run = model.next_play_type_proba(play(prev_play_type='R'))
        after_pass = model.next_play_type_proba(play(prev_play_type='P'))
        self.assertGreater(after_run[PLAY_TYPES.index('R')], after_pass[PLAY_TYPES.index('R')])
        self.assertEqual(model.describe(play(prev_play_type='K'))['prev_play_type'], 'K')

    @unittest.skipUnless(os.path.exists(REAL_DATA), 'real 2017 play-by-play not downloaded')
    def test_real_data_beats_overall_rates(self):
        '''TEST 13: on held-out 2017 games the model clearly beats the overall play-type rates'''
        from playByPlay import load_cfb_pbp
        plays = load_cfb_pbp(REAL_DATA)
        test_games = set(plays.game_id.drop_duplicates().sample(frac=0.2, random_state=0))
        is_test = plays.game_id.isin(test_games)
        test = plays[is_test & plays.play_type.isin(PLAY_TYPES)]
        y = test.play_type.map({t: i for i, t in enumerate(PLAY_TYPES)}).values

        def log_loss(model):
            P = model.fit(plays[~is_test]).predict_proba(test)
            return -np.log(P[np.arange(len(y)), y]).mean()

        self.assertLess(log_loss(PlayMarkovChain()), log_loss(PlayMarkovChain(markers=())) - 0.25)


class TestPrefilterPlays(unittest.TestCase):

    def setUp(self):
        self.raw = pd.DataFrame([
            play(down=1, distance=10, yards_to_goal=75),               # 0: fine
            play(down=0, distance=10, yards_to_goal=75),               # 1: down 0 -> dropped
            play(down=1, distance=10, yards_to_goal=0),                # 2: snap on the goal line -> dropped
            play(down=1, distance=0, yards_to_goal=5),                 # 3: goal-to-go, distance blank -> 5
            play(down=4, distance=0, yards_to_goal=18),                # 4: distance unknown -> dropped
            play(down=2, distance=15, yards_to_goal=5),                # 5: distance past the goal -> 5
            play(down=0, distance=0, yards_to_goal=100, play_type='K'),  # 6: kickoff, untouched
        ])
        self.raw['play_id'] = range(len(self.raw))

    def test_drops_and_repairs(self):
        '''TEST 14: impossible plays are dropped or repaired, kickoffs pass through'''
        clean, counts = prefilter_plays(self.raw, report=True)
        self.assertEqual(list(clean.play_id), [0, 3, 5, 6])
        self.assertEqual(list(clean.distance), [10, 5, 5, 0])
        self.assertEqual(counts, {'dropped: down not 1-4': 1, 'dropped: yards_to_goal not 1-99': 1,
                                  'dropped: no distance outside goal-to-go': 1,
                                  'repaired: goal-to-go distance': 1, 'repaired: distance past the goal line': 1})

    def test_leaves_input_and_clean_data_alone(self):
        '''TEST 15: the input isn't modified, and clean plays come back unchanged'''
        before = self.raw.copy()
        prefilter_plays(self.raw)
        pd.testing.assert_frame_equal(self.raw, before)
        clean = prefilter_plays(self.raw)
        pd.testing.assert_frame_equal(prefilter_plays(clean), clean)


if __name__ == '__main__':
    unittest.main()
