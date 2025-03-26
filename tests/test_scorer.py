import pytest
import pandas as pd
import numpy as np
from IPLScorer import IPLFantasyScorer

@pytest.fixture
def scorer():
    return IPLFantasyScorer()

@pytest.fixture
def sample_match_data():
    # Create a sample match data with various scenarios
    data = {
        'match_id': ['M1'] * 20,
        'date': ['2024-03-26'] * 20,
        'venue': ['Test Venue'] * 20,
        'batting_team': ['Team A'] * 20,
        'bowling_team': ['Team B'] * 20,
        'innings': [1] * 20,
        'over': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        'striker': ['Player1'] * 10 + ['Player2'] * 10,
        'bowler': ['Player3'] * 10 + ['Player4'] * 10,
        'fielder': ['Player5'] * 20,
        'runs_of_bat': [1, 4, 6, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'wide': [0] * 20,
        'noballs': [0] * 20,
        'byes': [0] * 20,
        'legbyes': [0] * 20,
        'extras': [0] * 20,
        'wicket_type': [None] * 19 + ['caught']
    }
    return pd.DataFrame(data)

def test_batting_points(scorer, sample_match_data):
    # Test batting points calculation for Player1
    stats = scorer.calculate_player_match_stats(sample_match_data, 'Player1')
    
    # Player1 scored 15 runs (1+4+6+0+0+1+1+1+1+1)
    # 1 four and 1 six
    expected_batting_score = (
        15 * scorer.points_per_run +  # Basic runs
        1 * scorer.boundary_bonus +   # One four
        1 * scorer.six_bonus         # One six
    )
    
    assert stats['batting_score'] == expected_batting_score

def test_bowling_points(scorer, sample_match_data):
    # Test bowling points calculation for Player3
    stats = scorer.calculate_player_match_stats(sample_match_data, 'Player3')
    
    # Player3 bowled 2 overs, conceded 15 runs, got 1 wicket
    # 1 maiden over (5th over)
    expected_bowling_score = (
        1 * scorer.points_per_wicket +  # One wicket
        1 * scorer.maiden_over_bonus +  # One maiden over
        5 * scorer.dot_ball_points      # 5 dot balls
    )
    
    assert stats['bowling_score'] == expected_bowling_score

def test_fielding_points(scorer, sample_match_data):
    # Test fielding points calculation for Player5
    stats = scorer.calculate_player_match_stats(sample_match_data, 'Player5')
    
    # Player5 took 1 catch
    expected_fielding_score = (
        1 * scorer.catch_points +  # One catch
        4  # Base fielding points
    )
    
    assert stats['fielding_score'] == expected_fielding_score

def test_duck_penalty(scorer):
    # Create data for a player getting out for 0
    data = {
        'match_id': ['M1'] * 3,
        'date': ['2024-03-26'] * 3,
        'venue': ['Test Venue'] * 3,
        'batting_team': ['Team A'] * 3,
        'bowling_team': ['Team B'] * 3,
        'innings': [1] * 3,
        'over': [1, 1, 1],
        'striker': ['Player1'] * 3,
        'bowler': ['Player3'] * 3,
        'fielder': ['Player5'] * 3,
        'runs_of_bat': [0, 0, 0],
        'wide': [0] * 3,
        'noballs': [0] * 3,
        'byes': [0] * 3,
        'legbyes': [0] * 3,
        'extras': [0] * 3,
        'wicket_type': [None, None, 'caught']
    }
    match_data = pd.DataFrame(data)
    
    stats = scorer.calculate_player_match_stats(match_data, 'Player1')
    assert stats['batting_score'] == scorer.dismissal_for_duck

def test_century_bonus(scorer):
    # Create data for a player scoring 100+ runs
    data = {
        'match_id': ['M1'] * 25,
        'date': ['2024-03-26'] * 25,
        'venue': ['Test Venue'] * 25,
        'batting_team': ['Team A'] * 25,
        'bowling_team': ['Team B'] * 25,
        'innings': [1] * 25,
        'over': [1] * 25,
        'striker': ['Player1'] * 25,
        'bowler': ['Player3'] * 25,
        'fielder': ['Player5'] * 25,
        'runs_of_bat': [4] * 25,  # 100 runs
        'wide': [0] * 25,
        'noballs': [0] * 25,
        'byes': [0] * 25,
        'legbyes': [0] * 25,
        'extras': [0] * 25,
        'wicket_type': [None] * 25
    }
    match_data = pd.DataFrame(data)
    
    stats = scorer.calculate_player_match_stats(match_data, 'Player1')
    expected_batting_score = (
        100 * scorer.points_per_run +  # Basic runs
        25 * scorer.boundary_bonus +   # 25 fours
        scorer.century_bonus           # Century bonus
    )
    
    assert stats['batting_score'] == expected_batting_score 