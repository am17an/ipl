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
        'recent_opposition_avg', 'recent_opposition_std'
    ]

def get_categorical_features():
    """Return the list of categorical features used in the model."""
    return ['team', 'opposition', 'venue', 'player']

def calculate_player_features(player_data, team=None, opposition=None, venue=None, batting_first=None, current_date=None):
    """
    Calculate features for a player based on their historical data.
    Args:
        player_data: DataFrame containing player's historical match data
        team: Current team (optional)
        opposition: Current opposition (optional)
        venue: Current venue (optional)
        current_date: Current date for recency calculations (optional)
    Returns:
        Dictionary of calculated features
    """
    if len(player_data) == 0:
        return None
    
    if current_date is None:
        current_date = datetime.now()
    
    # Calculate days since last match
    last_match_date = player_data['date'].max()
    days_since_last_match = (current_date - last_match_date).days
    
    # Calculate features based on historical data
    features = {}
    
    # Basic features
    features['avg_score'] = player_data['total_score'].mean()
    features['weighted_avg_score'] = np.average(
        player_data['total_score'],
        weights=np.exp(-0.01 * (current_date - player_data['date']).dt.days)
    )
    features['num_matches'] = len(player_data)
    
    # Calculate recent scores using only matches before current_date
    recent_matches = player_data[player_data['date'] < current_date].tail(5)
    features['recent_score'] = recent_matches['total_score'].mean() if len(recent_matches) > 0 else features['avg_score']
    
    # Venue-specific features
    if venue is not None:
        venue_data = player_data[
            (player_data['venue'] == venue) & 
            (player_data['date'] < current_date)
        ]
        features['venue_avg_score'] = venue_data['total_score'].mean() if len(venue_data) > 0 else features['avg_score']
        features['venue_std_score'] = venue_data['total_score'].std() if len(venue_data) > 1 else player_data['total_score'].std()
        features['venue_recent_score'] = venue_data.tail(5)['total_score'].mean() if len(venue_data) > 0 else features['recent_score']
        features['num_venue_matches'] = len(venue_data)
        
        # Venue-specific batting and bowling averages
        venue_batting = venue_data[venue_data['batting_stats'].apply(lambda x: x.get('balls', 0)) > 0]
        venue_bowling = venue_data[venue_data['bowling_stats'].apply(lambda x: x.get('balls', 0)) > 0]
        features['venue_batting_avg'] = venue_batting['batting_score'].mean() if len(venue_batting) > 0 else features['batting_avg']
        features['venue_bowling_avg'] = venue_bowling['bowling_score'].mean() if len(venue_bowling) > 0 else features['bowling_avg']
        
        # Recent venue performance
        recent_venue = venue_data.tail(5)
        features['recent_venue_avg'] = recent_venue['total_score'].mean() if len(recent_venue) > 0 else features['venue_avg_score']
        features['recent_venue_std'] = recent_venue['total_score'].std() if len(recent_venue) > 1 else features['venue_std_score']
    else:
        features['venue_avg_score'] = features['avg_score']
        features['venue_std_score'] = player_data['total_score'].std()
        features['venue_recent_score'] = features['recent_score']
        features['num_venue_matches'] = 0
        features['venue_batting_avg'] = features['batting_avg']
        features['venue_bowling_avg'] = features['bowling_avg']
        features['recent_venue_avg'] = features['recent_score']
        features['recent_venue_std'] = features['recent_score_std']
    
    # Opposition-specific features
    if opposition is not None:
        opposition_data = player_data[
            (player_data['opposition'] == opposition) & 
            (player_data['date'] < current_date)
        ]
        features['opposition_avg_score'] = opposition_data['total_score'].mean() if len(opposition_data) > 0 else features['avg_score']
        features['opposition_std_score'] = opposition_data['total_score'].std() if len(opposition_data) > 1 else player_data['total_score'].std()
        features['num_opposition_matches'] = len(opposition_data)
        
        # Opposition-specific batting and bowling averages
        opposition_batting = opposition_data[opposition_data['batting_stats'].apply(lambda x: x.get('balls', 0)) > 0]
        opposition_bowling = opposition_data[opposition_data['bowling_stats'].apply(lambda x: x.get('balls', 0)) > 0]
        features['opposition_batting_avg'] = opposition_batting['batting_score'].mean() if len(opposition_batting) > 0 else features['batting_avg']
        features['opposition_bowling_avg'] = opposition_bowling['bowling_score'].mean() if len(opposition_bowling) > 0 else features['bowling_avg']
        
        # Recent opposition performance
        recent_opposition = opposition_data.tail(5)
        features['recent_opposition_avg'] = recent_opposition['total_score'].mean() if len(recent_opposition) > 0 else features['opposition_avg_score']
        features['recent_opposition_std'] = recent_opposition['total_score'].std() if len(recent_opposition) > 1 else features['opposition_std_score']
    else:
        features['opposition_avg_score'] = features['avg_score']
        features['opposition_std_score'] = player_data['total_score'].std()
        features['num_opposition_matches'] = 0
        features['opposition_batting_avg'] = features['batting_avg']
        features['opposition_bowling_avg'] = features['bowling_avg']
        features['recent_opposition_avg'] = features['recent_score']
        features['recent_opposition_std'] = features['recent_score_std']
    
    # Venue-Opposition interaction features
    if venue is not None and opposition is not None:
        venue_opposition_data = player_data[
            (player_data['venue'] == venue) & 
            (player_data['opposition'] == opposition) & 
            (player_data['date'] < current_date)
        ]
        features['venue_opposition_avg'] = venue_opposition_data['total_score'].mean() if len(venue_opposition_data) > 0 else features['avg_score']
        features['venue_opposition_std'] = venue_opposition_data['total_score'].std() if len(venue_opposition_data) > 1 else player_data['total_score'].std()
    else:
        features['venue_opposition_avg'] = features['avg_score']
        features['venue_opposition_std'] = player_data['total_score'].std()
    
    # Innings-specific features
    if batting_first is not None:
        if team == batting_first:
            features['batting_first'] = 1
            features['bowling_first'] = 0
        else:
            features['batting_first'] = 0
            features['bowling_first'] = 1
    
    # Rolling statistics using only matches before current_date
    historical_data = player_data[player_data['date'] < current_date]
    recent_5_matches = historical_data.tail(5)
    recent_10_matches = historical_data.tail(10)
    recent_20_matches = historical_data.tail(20)
    
    features['recent_5_avg'] = recent_5_matches['total_score'].mean() if len(recent_5_matches) > 0 else features['avg_score']
    features['recent_5_std'] = recent_5_matches['total_score'].std() if len(recent_5_matches) > 1 else player_data['total_score'].std()
    features['recent_10_avg'] = recent_10_matches['total_score'].mean() if len(recent_10_matches) > 0 else features['avg_score']
    features['recent_10_std'] = recent_10_matches['total_score'].std() if len(recent_10_matches) > 1 else player_data['total_score'].std()
    features['recent_20_avg'] = recent_20_matches['total_score'].mean() if len(recent_20_matches) > 0 else features['avg_score']
    features['recent_20_std'] = recent_20_matches['total_score'].std() if len(recent_20_matches) > 1 else player_data['total_score'].std()

    # Role-specific features using only historical data
    batting_data = historical_data[historical_data['batting_stats'].apply(lambda x: x.get('balls', 0)).sum() > 0]
    bowling_data = historical_data[historical_data['bowling_stats'].apply(lambda x: x.get('balls', 0)).sum() > 0]
    
    features['batting_avg'] = batting_data['batting_score'].mean() if len(batting_data) > 0 else features['avg_score']
    features['bowling_avg'] = bowling_data['bowling_score'].mean() if len(bowling_data) > 0 else features['avg_score']
    features['batting_matches'] = len(batting_data)
    features['bowling_matches'] = len(bowling_data)
    features['recent_batting_avg'] = batting_data.tail(5)['batting_score'].mean() if len(batting_data) > 0 else features['batting_avg']
    features['recent_bowling_avg'] = bowling_data.tail(5)['bowling_score'].mean() if len(bowling_data) > 0 else features['bowling_avg']

    # Calculate innings-specific batting and bowling averages using only historical data
    first_innings_batting = historical_data[
        (historical_data['batting_innings'] == 1) & 
        (historical_data['batting_score'] > 0)
    ]
    second_innings_batting = historical_data[
        (historical_data['batting_innings'] == 2) & 
        (historical_data['batting_score'] > 0)
    ]
    
    first_innings_bowling = historical_data[
        (historical_data['bowling_innings'] == 1) & 
        (historical_data['bowling_score'] > 0)
    ]
    second_innings_bowling = historical_data[
        (historical_data['bowling_innings'] == 2) & 
        (historical_data['bowling_score'] > 0)
    ]
    
    features['avg_1st_innings_bat_score'] = first_innings_batting['batting_score'].mean() if len(first_innings_batting) > 0 else features['batting_avg']
    features['avg_2nd_innings_bat_score'] = second_innings_batting['batting_score'].mean() if len(second_innings_batting) > 0 else features['batting_avg']
    features['avg_1st_innings_bowl_score'] = first_innings_bowling['bowling_score'].mean() if len(first_innings_bowling) > 0 else features['bowling_avg']
    features['avg_2nd_innings_bowl_score'] = second_innings_bowling['bowling_score'].mean() if len(second_innings_bowling) > 0 else features['bowling_avg']

    # Consistency metrics using only historical data
    features['score_std'] = historical_data['total_score'].std()
    features['recent_score_std'] = recent_5_matches['total_score'].std() if len(recent_5_matches) > 1 else features['score_std']
    
    # Form indicators using only historical data
    last_3_scores = historical_data.tail(3)['total_score'].values
    features['form_trend'] = np.polyfit(range(len(last_3_scores)), last_3_scores, 1)[0] if len(last_3_scores) >= 2 else 0
    
    # Match frequency using only historical data
    if len(historical_data) > 1:
        dates = pd.to_datetime(historical_data['date'])
        features['avg_days_between_matches'] = (dates.max() - dates.min()).days / (len(dates) - 1)
    else:
        features['avg_days_between_matches'] = 0
    
    # Historical rank-based features using only historical data
    if len(recent_20_matches) > 0:
        # Calculate ranks for each past match
        past_match_ranks = []
        for match_id in recent_20_matches['match_id'].unique():
            match_players = recent_20_matches[recent_20_matches['match_id'] == match_id]
            match_ranks = match_players['total_score'].rank(ascending=False, method='min').astype(int)
            past_match_ranks.extend(match_ranks)
        
        # Calculate average rank and top 11 count
        features['avg_rank_last_20'] = np.mean(past_match_ranks)
        features['top_11_count_last_20'] = sum(1 for rank in past_match_ranks if rank <= 11)
    else:
        features['avg_rank_last_20'] = 12  # Default to middle rank
        features['top_11_count_last_20'] = 0
    
    # Current match features
    if team is not None:
        features['team'] = team
    if opposition is not None:
        features['opposition'] = opposition
    if venue is not None:
        features['venue'] = venue
    
    return features 