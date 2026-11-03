import unittest

import sys
import os
import tempfile
import pandas as pd
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_dir)
from playByPlay import load_cfb_pbp, map_play_types, PLAY_CODES

REAL_DATA = os.path.join(root_dir, 'data', 'cfb_pbp', 'play_by_play_2017.parquet')


def raw_row(**fields):
    '''one cfbfastR-style row with neutral defaults'''
    row = {'game_id': 1, 'game_play_number': 1, 'season': 2017, 'week': 1, 'pos_team': 'Georgia Tech',
           'def_pos_team': 'Clemson', 'play_type': 'Rush', 'play_text': '', 'down': 1.0, 'distance': 10.0,
           'yards_to_goal': 75.0, 'pos_score_diff_start': 0.0, 'half': 1, 'period': 1, 'TimeSecsRem': 1800.0,
           'rush': 0.0, 'pass': 0.0}
    row.update(fields)
    return row


class TestMapPlayTypes(unittest.TestCase):

    def test_mapping(self):
        '''TEST 1: cfbfastR play types collapse to the right codes'''
        cases = [
            ('rush', dict(play_type='Rush', rush=1.0), 'R'),
            ('rushing touchdown', dict(play_type='Rushing Touchdown', rush=1.0), 'R'),
            ('sack', dict(play_type='Sack', **{'pass': 1.0}), 'P'),
            ('interception', dict(play_type='Interception Return', **{'pass': 1.0}), 'P'),
            ('fumble on a run', dict(play_type='Fumble Recovery (Opponent)', rush=1.0), 'R'),
            ('fumble on a pass', dict(play_type='Fumble Recovery (Own)', **{'pass': 1.0}), 'P'),
            ('missed field goal', dict(play_type='Field Goal Missed'), 'F'),
            ('blocked punt', dict(play_type='Blocked Punt'), 'U'),
            ('kickoff return', dict(play_type='Kickoff Return (Offense)'), 'K'),
            ('penalty wiping out a pass', dict(play_type='Penalty', play_text=
                'Smith pass incomplete to Jones, PENALTY GT holding 10 yards, NO PLAY.'), 'P'),
            ('penalty wiping out a sack', dict(play_type='Penalty', play_text=
                'Smith sacked by Lee for a loss of 7 yards, PENALTY CLEM facemask, NO PLAY.'), 'P'),
            ('penalty wiping out a run', dict(play_type='Penalty', play_text=
                'Smith run for 12 yds, PENALTY GT holding 10 yards, NO PLAY.'), 'R'),
            ('penalty on a punt', dict(play_type='Penalty', play_text=
                'Jones punt for 41 yds, PENALTY CLEM roughing the kicker, NO PLAY.'), 'U'),
            ('pre-snap foul', dict(play_type='Penalty', play_text='GEORGIA TECH Penalty, False Start (-5 Yards)'), 'O'),
            ('safety', dict(play_type='Safety'), 'O'),
        ]
        codes = map_play_types(pd.DataFrame([raw_row(**fields) for _, fields, _ in cases]))
        for (name, _, expected), code in zip(cases, codes):
            with self.subTest(case=name):
                self.assertEqual(code, expected)

    def test_codes_are_documented(self):
        '''TEST 2: every code the mapper can produce is listed in PLAY_CODES'''
        rows = [raw_row(play_type=t) for t in ('Rush', 'Punt', 'Kickoff', 'Field Goal Good', 'Penalty', 'Safety')]
        self.assertTrue(set(map_play_types(pd.DataFrame(rows))) <= set(PLAY_CODES))


class TestLoadCfbPbp(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # game 1 is stored out of order and includes rows that aren't plays
        game_1 = [
            raw_row(game_id=1, game_play_number=3, play_type='Pass Reception', **{'pass': 1.0}),
            raw_row(game_id=1, game_play_number=1, play_type='Kickoff'),
            raw_row(game_id=1, game_play_number=4, play_type='Timeout'),
            raw_row(game_id=1, game_play_number=2, play_type='Rush', rush=1.0, distance=0.0),
            raw_row(game_id=1, game_play_number=5, play_type='Rushing Touchdown', rush=1.0, down=0.0),
            raw_row(game_id=1, game_play_number=6, play_type='End Period', down=0.0),
        ]
        game_2 = [
            raw_row(game_id=2, game_play_number=1, play_type='Rush', rush=1.0),
            raw_row(game_id=2, game_play_number=2, play_type='Punt'),
        ]
        self.path_1 = os.path.join(tmp.name, 'game_1.parquet')
        self.path_2 = os.path.join(tmp.name, 'game_2.parquet')
        pd.DataFrame(game_1).to_parquet(self.path_1)
        pd.DataFrame(game_2).to_parquet(self.path_2)

    def test_drops_non_plays(self):
        '''TEST 1: timeouts and end-of-period rows are removed'''
        plays = load_cfb_pbp(self.path_1)
        self.assertEqual(list(plays.raw_play_type), ['Kickoff', 'Rush', 'Pass Reception', 'Rushing Touchdown'])

    def test_game_order_and_prev_play_type(self):
        '''TEST 2: plays are in game order and prev_play_type restarts at 'S' each game'''
        plays = load_cfb_pbp([self.path_1, self.path_2])
        self.assertEqual(list(plays.play_number), [1, 2, 3, 5, 1, 2])
        self.assertEqual(list(plays.play_type), ['K', 'R', 'P', 'R', 'R', 'U'])
        self.assertEqual(list(plays.prev_play_type), ['S', 'K', 'R', 'P', 'S', 'R'])

    def test_keeps_raw_fields(self):
        '''TEST 3: down and distance are left as recorded for prefilter_plays to handle'''
        plays = load_cfb_pbp(self.path_1).set_index('play_number')
        self.assertEqual(plays.loc[2, 'distance'], 0)
        self.assertEqual(plays.loc[5, 'down'], 0)

    def test_output_columns(self):
        '''TEST 4: the loader returns the documented columns'''
        expected = ['game_id', 'play_number', 'season', 'week', 'offense', 'defense', 'period', 'half',
                    'seconds_left_half', 'down', 'distance', 'yards_to_goal', 'score_diff', 'play_type',
                    'prev_play_type', 'raw_play_type']
        self.assertEqual(list(load_cfb_pbp(self.path_1).columns), expected)

    @unittest.skipUnless(os.path.exists(REAL_DATA), 'real 2017 play-by-play not downloaded')
    def test_real_data(self):
        '''TEST 5: the real 2017 season loads cleanly'''
        plays = load_cfb_pbp(REAL_DATA)
        scrimmage = plays[plays.play_type != 'K']
        marker_fields = ['down', 'distance', 'yards_to_goal', 'score_diff', 'half', 'period', 'seconds_left_half']
        with self.subTest(check='known codes'):
            self.assertTrue(set(plays.play_type) <= set(PLAY_CODES))
        with self.subTest(check='no missing marker fields'):
            self.assertEqual(int(scrimmage[marker_fields].isna().sum().sum()), 0)
        with self.subTest(check='one game start per game'):
            self.assertEqual((plays.prev_play_type == 'S').sum(), plays.game_id.nunique())
        with self.subTest(check='field ranges'):
            self.assertTrue(plays.yards_to_goal.between(0, 100).all())
        with self.subTest(check='field ranges after prefilter_plays'):
            from markovChain import prefilter_plays
            clean = prefilter_plays(plays)
            clean_scrimmage = clean[clean.play_type != 'K']
            self.assertTrue(clean.down.between(1, 4).all())
            self.assertTrue((clean_scrimmage.distance >= 1).all())
            self.assertTrue((clean_scrimmage.distance <= clean_scrimmage.yards_to_goal).all())
            self.assertTrue(clean_scrimmage.yards_to_goal.between(1, 99).all())
            self.assertGreater(len(clean), 0.99 * len(plays))


if __name__ == '__main__':
    unittest.main()
