import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


CFB_PBP_URL = ('https://github.com/sportsdataverse/sportsdataverse-data/releases/download/'
               'cfbfastR_cfb_pbp/play_by_play_{season}.parquet')

# play-type codes produced by this loader
PLAY_CODES = {
    'R': 'run',
    'P': 'pass (incl. sacks and interceptions)',
    'F': 'field goal',
    'U': 'punt',
    'K': 'kickoff',
    'O': 'other (pre-snap foul / no play, safety)',
}

# rows that are not plays at all
ADMIN_TYPES = {'End Period', 'End of Half', 'End of Game', 'Timeout', 'Uncategorized', 'Uncategorized Touchdown',
               'Defensive 2pt Conversion'}
FIELD_GOAL_TYPES = {'Field Goal Good', 'Field Goal Missed', 'Blocked Field Goal', 'Blocked Field Goal Touchdown',
                    'Missed Field Goal Return'}
PUNT_TYPES = {'Punt', 'Blocked Punt', 'Blocked Punt Touchdown', 'Blocked Punt (Safety)', 'Punt Return Touchdown',
              'Punt Team Fumble Recovery', 'Punt Team Fumble Recovery Touchdown'}
KICKOFF_TYPES = {'Kickoff', 'Kickoff Return (Offense)', 'Kickoff Return Touchdown', 'Kickoff Team Fumble Recovery'}

SOURCE_COLUMNS = ['game_id', 'game_play_number', 'season', 'week', 'pos_team', 'def_pos_team', 'play_type',
                  'play_text', 'down', 'distance', 'yards_to_goal', 'pos_score_diff_start', 'half', 'period',
                  'TimeSecsRem', 'rush', 'pass']


def download_cfb_pbp(season: int, dest_dir: str = 'data/cfb_pbp') -> Path:
    '''
    download one season of cfbfastR college football play-by-play (parquet, ~70-130 MB)
    from the SportsDataverse GitHub release; skips the download if the file already exists

    Returns:
        path to the downloaded parquet file
    '''
    dest = Path(dest_dir) / f'play_by_play_{season}.parquet'
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CFB_PBP_URL.format(season=season), dest)
    return dest


def map_play_types(raw: pd.DataFrame) -> np.ndarray:
    '''
    collapse cfbfastR's ~40 play_type labels into the codes in PLAY_CODES

    Fumbles, sacks and interceptions are sorted with the rush/pass flags. "Penalty" rows
    are sorted by their text: most are a real run or pass that was wiped out by the flag,
    the rest are pre-snap fouls where no play happened (-> 'O'). PATs are not separate
    rows in this data (they're folded into the touchdown play), so there is no PAT code.

    Args:
        raw: cfbfastR rows with play_type, play_text, rush and pass columns

    Returns:
        array of play-type codes, one per row
    '''
    play_type = raw.play_type
    text = raw.play_text.fillna('').str.lower()

    codes = np.full(len(raw), 'O', dtype=object)
    codes[(raw.rush == 1).values] = 'R'
    codes[(raw['pass'] == 1).values] = 'P'
    codes[play_type.isin(FIELD_GOAL_TYPES).values] = 'F'
    codes[play_type.isin(PUNT_TYPES).values] = 'U'
    codes[play_type.isin(KICKOFF_TYPES).values] = 'K'

    penalty = (play_type == 'Penalty').values
    pass_text = text.str.contains(r'\bpass\b|sacked').values
    codes[penalty & pass_text] = 'P'
    codes[penalty & ~pass_text & text.str.contains(r'\brun\b|\brush\b|\brushed\b').values] = 'R'
    codes[penalty & text.str.contains('punt').values] = 'U'
    codes[penalty & text.str.contains('field goal').values] = 'F'
    return codes


def load_cfb_pbp(paths) -> pd.DataFrame:
    '''
    load cfbfastR play-by-play parquet file(s) into one clean row per play, in game order

    Args:
        paths: a parquet path, or a list of them (e.g. several seasons)

    Returns:
        DataFrame with columns
            game_id, play_number, season, week, offense, defense,
            period, half, seconds_left_half,                  -- clock before the snap
            down, distance, yards_to_goal, score_diff,        -- situation before the snap (offense's view)
            play_type, prev_play_type, raw_play_type
        play_type uses the codes in PLAY_CODES; prev_play_type is the previous play in the
        same game ('S' for the first play). Kickoff rows carry placeholder down/distance values.
        down and distance are left as recorded (a few hundred rows are impossible, e.g. down 0
        or distance 0); pass the result through markovChain.prefilter_plays before fitting.
    '''
    paths = [paths] if isinstance(paths, (str, Path)) else paths
    raw = pd.concat([pd.read_parquet(p, columns=SOURCE_COLUMNS) for p in paths], ignore_index=True)
    raw = raw[~raw.play_type.isin(ADMIN_TYPES)]
    raw = raw.sort_values(['game_id', 'game_play_number']).reset_index(drop=True)

    plays = pd.DataFrame({
        'game_id': raw.game_id,
        'play_number': raw.game_play_number.astype(int),
        'season': raw.season,
        'week': raw.week,
        'offense': raw.pos_team,
        'defense': raw.def_pos_team,
        'period': raw.period,
        'half': raw.half.astype(int),
        'seconds_left_half': raw.TimeSecsRem,
        'down': raw.down,          # raw values: markovChain.prefilter_plays drops or repairs
        'distance': raw.distance,  # impossible downs, distances and ball spots
        'yards_to_goal': raw.yards_to_goal,
        'score_diff': raw.pos_score_diff_start,
        'play_type': map_play_types(raw),
        'raw_play_type': raw.play_type,
    })
    plays.insert(plays.columns.get_loc('raw_play_type'), 'prev_play_type',
                 plays.groupby('game_id').play_type.shift(1).fillna('S'))
    return plays


if __name__ == '__main__':
    plays = load_cfb_pbp(download_cfb_pbp(2017))
    print(f'{len(plays)} plays from {plays.game_id.nunique()} games')
    print(plays.play_type.value_counts())
