import pandas as pd
import numpy as np
from datetime import datetime

def get_numeric_features():
    """
    Returns a list of numeric feature names used in the model.
    """
    return [
        # Basic features
        'avg_score', 'weighted_avg_score', 'num_matches', 'recent_score',
        
        # Venue-specific features
        'venue_avg_score', 'venue_std_score', 'venue_recent_score', 'num_venue_matches',
        
        # Opposition-specific features
        'opposition_avg_score', 'opposition_std_score', 'num_opposition_matches',
        
        # Innings-specific features
        'avg_1st_innings_bat_score', 'avg_2nd_innings_bat_score',
        'avg_1st_innings_bowl_score', 'avg_2nd_innings_bowl_score',
        
        # Rolling statistics
        'recent_5_avg', 'recent_5_std', 'recent_10_avg', 'recent_10_std',
        'recent_20_avg', 'recent_20_std',
        
        # Role-specific features
        'batting_avg', 'bowling_avg', 'batting_matches', 'bowling_matches',
        'recent_batting_avg', 'recent_bowling_avg',
        'average_balls_faced', 'average_balls_bowled',
        'average_balls_faced_at_venue', 'average_balls_bowled_at_venue',
        'strike_rate_batting', 'strike_rate_bowling', 'economy_rate_bowling',
        
        # Consistency metrics
        'score_std', 'recent_score_std',
        
        # Form indicators
        'form_trend', 'avg_days_between_matches',
        
        # Historical rank-based features
        'avg_rank_last_5', 'top_11_count_last_5',
        
        # Interaction features
        'venue_opposition_avg', 'venue_opposition_std',
        'venue_batting_avg', 'venue_bowling_avg',
        'opposition_batting_avg', 'opposition_bowling_avg',
        'recent_venue_avg', 'recent_venue_std',
        'recent_opposition_avg', 'recent_opposition_std',

        # Total venue stats
        'total_venue_batting_avg', 'total_venue_bowling_avg',

        # season
        'season', 'match_number',

        # Team composition features
        #'team_batting_strength', 'team_bowling_strength', 'team_all_rounder_count',
        #'team_recent_performance', 'team_balance_score',

        # Player correlation features
        #'teammate_synergy_score', 'role_complementarity', 'batting_position_impact'
    ]

def get_categorical_features():
    """Return the list of categorical features used in the model."""
    return ['team', 'opposition', 'venue', 'player']

def zero_variance_features_check(df):
    """
    Check for zero variance features in the dataframe.
    """
    zero_variance_cols = []

    for col in df.columns:
        # For numeric columns, use variance
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].var() == 0:
                zero_variance_cols.append(col)
        # For non-numeric columns, check if number of unique values is 1
        else:
            if df[col].nunique() == 1:
                zero_variance_cols.append(col)
    
    if zero_variance_cols:
        raise ValueError(f"Zero variance features found: {zero_variance_cols}")