import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from feature_utils import calculate_player_features, get_numeric_features, get_categorical_features

@pytest.fixture
def sample_player_data():
    # Create sample player data with various scenarios
    dates = [datetime.now() - timedelta(days=i*7) for i in range(10)]  # 10 matches over 70 days
    data = {
        'match_id': [f'M{i}' for i in range(10)],
        'date': dates,
        'venue': ['Venue1'] * 5 + ['Venue2'] * 5,
        'batting_team': ['Team A'] * 10,
        'bowling_team': ['Team B'] * 5 + ['Team C'] * 5,
        'innings': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'total_score': [50, 30, 70, 40, 60, 35, 45, 55, 65, 40],
        'batting_score': [30, 20, 40, 25, 35, 20, 25, 30, 35, 25],
        'bowling_score': [20, 10, 30, 15, 25, 15, 20, 25, 30, 15],
        'fielding_score': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'batting_innings': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'bowling_innings': [2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        'batting_stats': [{'balls': 30, 'runs': 30}] * 10,
        'bowling_stats': [{'balls': 24, 'runs': 20, 'wickets': 2}] * 10
    }
    return pd.DataFrame(data)

def test_basic_features(sample_player_data):
    features = calculate_player_features(sample_player_data)
    
    # Test basic features
    assert features['num_matches'] == 10
    assert features['avg_score'] == 50.0  # (50+30+70+40+60+35+45+55+65+40)/10
    assert features['score_std'] is not None
    assert features['batting_avg'] == 30.0  # Average batting score
    assert features['bowling_avg'] == 20.0  # Average bowling score

def test_venue_specific_features(sample_player_data):
    features = calculate_player_features(
        sample_player_data,
        venue='Venue1',
        current_date=datetime.now()
    )
    
    # Test venue-specific features
    assert features['venue_avg_score'] is not None
    assert features['venue_std_score'] is not None
    assert features['num_venue_matches'] == 5  # 5 matches at Venue1
    assert features['venue_batting_avg'] is not None
    assert features['venue_bowling_avg'] is not None

def test_opposition_specific_features(sample_player_data):
    features = calculate_player_features(
        sample_player_data,
        opposition='Team B',
        current_date=datetime.now()
    )
    
    # Test opposition-specific features
    assert features['opposition_avg_score'] is not None
    assert features['opposition_std_score'] is not None
    assert features['num_opposition_matches'] == 5  # 5 matches against Team B
    assert features['opposition_batting_avg'] is not None
    assert features['opposition_bowling_avg'] is not None

def test_innings_specific_features(sample_player_data):
    features = calculate_player_features(
        sample_player_data,
        batting_first=True,
        current_date=datetime.now()
    )
    
    # Test innings-specific features
    assert features['avg_1st_innings_bat_score'] is not None
    assert features['avg_2nd_innings_bat_score'] is not None
    assert features['avg_1st_innings_bowl_score'] is not None
    assert features['avg_2nd_innings_bowl_score'] is not None
    assert features['batting_first'] == 1
    assert features['bowling_first'] == 0

def test_rolling_statistics(sample_player_data):
    features = calculate_player_features(
        sample_player_data,
        current_date=datetime.now()
    )
    
    # Test rolling statistics
    assert features['recent_5_avg'] is not None
    assert features['recent_5_std'] is not None
    assert features['recent_10_avg'] is not None
    assert features['recent_10_std'] is not None
    assert features['recent_20_avg'] is not None
    assert features['recent_20_std'] is not None

def test_feature_lists():
    # Test that feature lists are complete and non-empty
    numeric_features = get_numeric_features()
    categorical_features = get_categorical_features()
    
    assert len(numeric_features) > 0
    assert len(categorical_features) > 0
    assert 'avg_score' in numeric_features
    assert 'team' in categorical_features

def test_empty_data():
    # Test handling of empty data
    empty_data = pd.DataFrame()
    features = calculate_player_features(empty_data)
    assert features is None

def test_weighted_average(sample_player_data):
    features = calculate_player_features(
        sample_player_data,
        current_date=datetime.now()
    )
    
    # Test that weighted average is different from simple average
    # due to recency weighting
    assert features['weighted_avg_score'] != features['avg_score']
    assert features['weighted_avg_score'] is not None 