import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
import glob
import os
import matplotlib.pyplot as plt
import traceback
from sklearn.impute import SimpleImputer
import kagglehub
from kagglehub import KaggleDatasetAdapter
from IPLScorer import IPLFantasyScorer
from feature_utils import get_numeric_features, get_categorical_features, zero_variance_features_check

def create_player_dataset(file_pattern="ipl_{season}_deliveries.csv", cache_file="player_matches.csv"):
    """
    Create a dataset from multiple IPL seasons
    """
    print("Creating new dataset...")
    all_seasons_df = pd.DataFrame()
    seasons = [2021, 2022, 2023, 2024]
    dl_from_kaggle = [2025]
    
    for season in seasons:
        file_path = file_pattern.format(season=season)
        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            try:
                season_df = pd.read_csv(file_path)
                print(f"Loaded {len(season_df)} rows from {file_path}")
                print(f"Columns: {season_df.columns.tolist()}")
                print(f"Sample of dates: {season_df['date'].head()}")
                all_seasons_df = pd.concat([all_seasons_df, season_df])
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        else:
            print(f"File not found: {file_path}")
    

    for season in dl_from_kaggle:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "sahiltailor/ipl-2024-ball-by-ball-dataset",
            f"ipl_{season}_deliveries.csv"
        )

        print(f"Loaded {len(df)} rows from Kaggle")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample of dates: {df['date'].head()}")

        all_seasons_df = pd.concat([all_seasons_df, df])

    
    print(f"\nTotal rows loaded: {len(all_seasons_df)}")
    # Convert date strings to datetime objects
    try:
        all_seasons_df['date'] = pd.to_datetime(all_seasons_df['date'], format='%b %d, %Y', errors='coerce')
        print(f"Date range: {all_seasons_df['date'].min()} to {all_seasons_df['date'].max()}")
        
        # Save to cache file
        print(f"\nSaving dataset to {cache_file}...")
        all_seasons_df.to_csv(cache_file, index=False)
        print("Dataset saved successfully")
    except Exception as e:
        print(f"Error converting dates or saving cache: {str(e)}")
        print("Sample of date values:", all_seasons_df['date'].head())
    
    return all_seasons_df

def create_features_and_train_test_split(deliveries_df, cache_file="features_df.csv"):
    """
    Create features for all matches, keeping the temporal order intact.
    This will be used for rolling window predictions.
    """
    # Check if cached features exist
    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        try:
            features_df = pd.read_csv(cache_file)
            features_df['date'] = pd.to_datetime(features_df['date'])
            print(f"Successfully loaded {len(features_df)} rows from cache")
            print(f"Date range: {features_df['date'].min()} to {features_df['date'].max()}")
            
            required_features = get_numeric_features() + get_categorical_features()
            
            missing_features = [f for f in required_features if f not in features_df.columns]
            if missing_features:
                print(f"Missing features in cache: {missing_features}")
                print("Recomputing features...")
                players_df = create_features_from_scratch(deliveries_df, cache_file)

                zero_variance_features_check(players_df)

                return players_df
            
            return features_df
        except Exception as e:
            print(f"Error loading cached features: {str(e)}")
    
    return create_features_from_scratch(deliveries_df, cache_file)

def create_features_from_scratch(deliveries_df, cache_file):
    """
    Create features from scratch, ensuring all required features are computed.
    """
    print("\nStarting feature creation process from scratch...")
    print(f"Input data shape: {deliveries_df.shape}")
    print(f"Unique match IDs: {deliveries_df['match_id'].nunique()}")
    
    scorer = IPLFantasyScorer()
    player_matches = []
    
    # First pass: Calculate all player stats for each match
    for match_id in deliveries_df['match_id'].unique():
        match_data = deliveries_df[deliveries_df['match_id'] == match_id]
        
        # Get all players who participated in this match
        all_players = set()
        for innings in match_data['innings'].unique():
            innings_data = match_data[match_data['innings'] == innings]
            all_players.update(innings_data['striker'].unique())
            all_players.update(innings_data['bowler'].unique())
            all_players.update(innings_data['fielder'].dropna().unique())
        
        # Process each player's performance
        match_player_stats = []
        for player in all_players:
            try:
                stats = scorer.calculate_player_match_stats(match_data, player)
                if stats:
                    player_matches.append(stats)
            except Exception as e:
                print(f"Error processing player {player} in match {match_id}: {str(e)}")
                continue
    
    # Convert to DataFrame and sort by date
    player_df = pd.DataFrame(player_matches).sort_values('date')
    print(f"\nPlayer matches DataFrame shape: {player_df.shape}")
    print(player_df.head())
    print(f"Unique players: {player_df['player'].nunique()}")
    
    # Create features based on past performance
    features_df = []

    # Create total venue stats
    venue_features_df = create_venue_stats(player_df)

    for player in player_df['player'].unique():
        try:
            player_matches = player_df[player_df['player'] == player].copy()
            
            if len(player_matches) < 3:
                continue
            
            # For each match, create features based on past matches only
            for i in range(1, len(player_matches)):
                try:
                    current_match = player_matches.iloc[i:i+1]
                    current_date = pd.to_datetime(current_match['date'].iloc[0])
                    
                    # Get past matches by comparing datetime objects
                    past_matches = player_matches[pd.to_datetime(player_matches['date']) < current_date]
                    
                    current_venue = current_match['venue'].iloc[0]
                    current_opposition = current_match['opposition'].iloc[0]
                    
                    # Calculate venue-specific features
                    venue_matches = past_matches[past_matches['venue'] == current_venue]
                    venue_avg_score = venue_matches['total_score'].mean() if len(venue_matches) > 0 else past_matches['total_score'].mean()
                    venue_std_score = venue_matches['total_score'].std() if len(venue_matches) > 1 else past_matches['total_score'].std()
                    
                    # Calculate opposition-specific features
                    opposition_matches = past_matches[past_matches['opposition'] == current_opposition]
                    opposition_avg_score = opposition_matches['total_score'].mean() if len(opposition_matches) > 0 else past_matches['total_score'].mean()
                    opposition_std_score = opposition_matches['total_score'].std() if len(opposition_matches) > 1 else past_matches['total_score'].std()

                    total_venue_matches = venue_features_df[(venue_features_df['venue'] == current_venue) & (venue_features_df['date'] < current_date)]
                    # remove 0 values from batting score 
                    total_venue_batting_avg_first_innings = total_venue_matches['batting_score_first_innings'].mean()
                    total_venue_bowling_avg_first_innings = total_venue_matches['bowling_score_first_innings'].mean()
                    total_venue_batting_avg_second_innings = total_venue_matches['batting_score_second_innings'].mean()
                    total_venue_bowling_avg_second_innings = total_venue_matches['bowling_score_second_innings'].mean()

                    # Calculate innings-specific batting and bowling averages
                    first_innings_batting = past_matches[
                        (past_matches['batting_innings'] == 1) 
                    ]
                    second_innings_batting = past_matches[
                        (past_matches['batting_innings'] == 2) 
                    ]
                    
                    first_innings_bowling = past_matches[
                        (past_matches['bowling_innings'] == 1)
                    ]
                    second_innings_bowling = past_matches[
                        (past_matches['bowling_innings'] == 2)
                    ]
                    
                    avg_1st_innings_bat_score = first_innings_batting['batting_score'].mean() if len(first_innings_batting) >= 3 else 0
                    avg_2nd_innings_bat_score = second_innings_batting['batting_score'].mean() if len(second_innings_batting) >= 3 else 0
                    avg_1st_innings_bowl_score = first_innings_bowling['bowling_score'].mean() if len(first_innings_bowling) >= 3 else 0
                    avg_2nd_innings_bowl_score = second_innings_bowling['bowling_score'].mean() if len(second_innings_bowling) >= 3 else 0
                    
                    # Use last 5 matches at venue if available, otherwise use last 5 overall matches
                    venue_matches_recent = venue_matches.tail(min(5, len(venue_matches)))
                    venue_recent_score = venue_matches_recent['total_score'].mean() if len(venue_matches_recent) > 0 else past_matches.tail(5)['total_score'].mean()
                    
                    # Calculate recency-weighted average score
                    days_diff = (current_date - pd.to_datetime(past_matches['date'])).dt.days
                    days_diff = days_diff.apply(lambda x: min(50, x))  # Apply min to each individual value
                    recency_weights = np.exp(-0.01 * days_diff)
                    recency_weights = recency_weights / recency_weights.sum()
                    weighted_avg_score = np.average(past_matches['total_score'], weights=recency_weights)
                    
                    # Calculate rolling statistics
                    recent_5_matches = past_matches.tail(5)
                    recent_10_matches = past_matches.tail(10)
                    recent_20_matches = past_matches.tail(20)
                    
                    # Calculate batting and bowling averages based on balls faced/bowled
                    batting_matches = past_matches[past_matches.apply(lambda x: x['batting_stats'].get('balls', 0) > 0, axis=1)]
                    bowling_matches = past_matches[past_matches.apply(lambda x: x['bowling_stats'].get('balls', 0) > 0, axis=1)]

                    batting_avg = batting_matches['batting_score'].mean() if len(batting_matches) >= 3 else 0
                    bowling_avg = bowling_matches['bowling_score'].mean() if len(bowling_matches) >= 3 else 0
                    
                    # Calculate role-specific recent performance
                    recent_batting_matches = batting_matches.tail(5)
                    recent_bowling_matches = bowling_matches.tail(5)
                    
                    recent_batting_avg = recent_batting_matches['batting_score'].mean() if len(recent_batting_matches) >= 3 else batting_avg
                    recent_bowling_avg = recent_bowling_matches['bowling_score'].mean() if len(recent_bowling_matches) >= 3 else bowling_avg

                    # Calculate average balls faced and bowled
                    average_balls_faced = batting_matches.apply(lambda x: x['batting_stats'].get('balls', 0), axis=1).mean() if len(batting_matches) >= 5 else 0
                    average_balls_bowled = bowling_matches.apply(lambda x: x['bowling_stats'].get('balls', 0), axis=1).mean() if len(bowling_matches) >= 5 else 0

                    # Calculate venue-specific average balls faced and bowled
                    average_balls_faced_at_venue = venue_matches.apply(lambda x: x['batting_stats'].get('balls', 0), axis=1).mean() if len(venue_matches) >= 3 else average_balls_faced
                    average_balls_bowled_at_venue = venue_matches.apply(lambda x: x['bowling_stats'].get('balls', 0), axis=1).mean() if len(venue_matches) >= 3 else average_balls_bowled

                    # Calculate venue-specific batting and bowling averages
                    venue_batting = venue_matches[venue_matches.apply(lambda x: x['batting_stats'].get('balls', 0) > 0, axis=1)]
                    venue_bowling = venue_matches[venue_matches.apply(lambda x: x['bowling_stats'].get('balls', 0) > 0, axis=1)]
                    
                    # Calculate opposition-specific batting and bowling averages
                    opposition_batting = opposition_matches[opposition_matches.apply(lambda x: x['batting_stats'].get('balls', 0) > 0, axis=1)]
                    opposition_bowling = opposition_matches[opposition_matches.apply(lambda x: x['bowling_stats'].get('balls', 0) > 0, axis=1)]
                    
                    # Calculate venue-opposition interaction features
                    if current_venue is not None and current_opposition is not None:
                        venue_opposition_data = past_matches[
                            (past_matches['venue'] == current_venue) & 
                            (past_matches['opposition'] == current_opposition)
                        ]
                    else:
                        venue_opposition_data = pd.DataFrame()

                    # Calculate recent venue and opposition performance
                    recent_venue = venue_matches.tail(5)
                    recent_opposition = opposition_matches.tail(5)

                    # Calculate strike rates and economy rate
                    def calculate_batting_strike_rate(matches):
                        if len(matches) <= 3:
                            return 0
                        total_runs = matches.apply(lambda x: x['batting_stats'].get('runs', 0), axis=1).sum()
                        total_balls = matches.apply(lambda x: x['batting_stats'].get('balls', 0), axis=1).sum()
                        return total_runs / total_balls if total_balls >= 30 else 0

                    def calculate_bowling_strike_rate(matches):
                        if len(matches) <= 3:
                            return 0
                        total_balls = matches.apply(lambda x: x['bowling_stats'].get('balls', 0), axis=1).sum()
                        total_wickets = matches.apply(lambda x: x['bowling_stats'].get('wickets', 0), axis=1).sum()
                        return total_balls / total_wickets if total_wickets > 0 else 100.0

                    def calculate_economy_rate(matches):
                        if len(matches) <= 3:
                            return 0
                        total_runs = matches.apply(lambda x: x['bowling_stats'].get('runs', 0), axis=1).sum()
                        total_balls = matches.apply(lambda x: x['bowling_stats'].get('balls', 0), axis=1).sum()
                        return total_runs / total_balls if total_balls >= 30 else 20

                    strike_rate_batting = calculate_batting_strike_rate(batting_matches)
                    strike_rate_bowling = calculate_bowling_strike_rate(bowling_matches)
                    economy_rate_bowling = calculate_economy_rate(bowling_matches)
                    
                    # Calculate consistency metrics
                    score_std = past_matches['total_score'].std()
                    recent_score_std = recent_5_matches['total_score'].std()
                    
                    # Calculate form indicators
                    last_3_scores = past_matches.tail(3)['total_score'].values
                    form_trend = np.polyfit(range(len(last_3_scores)), last_3_scores, 1)[0] if len(last_3_scores) >= 2 else 0
                    
                    # Calculate match frequency
                    if len(past_matches) > 1:
                        dates = pd.to_datetime(past_matches['date'])
                        avg_days_between_matches = (dates.max() - dates.min()).days / (len(dates) - 1)
                    else:
                        avg_days_between_matches = 0
                    
                    # Calculate historical rank-based features using only past matches
                    past_match_ranks = []
                    if len(recent_5_matches) > 0:
                        # Calculate ranks for each past match
                        for match_id in recent_5_matches['match_id'].unique():
                            # Get all players who played in this historical match
                            historical_match_players = player_df[
                                (player_df['match_id'] == match_id) & 
                                (pd.to_datetime(player_df['date']) < current_date)
                            ]
                            if len(historical_match_players) > 0:
                                # Calculate ranks for all players in this match
                                match_ranks = historical_match_players['total_score'].rank(ascending=False, method='min').astype(int)
                                
                                # Get the player's rank if they played in this match
                                player_match_data = historical_match_players[historical_match_players['player'] == player]
                                if len(player_match_data) > 0:
                                    player_rank = match_ranks[historical_match_players['player'] == player].iloc[0]
                                    past_match_ranks.append(player_rank)
                    
                    # calculate full venue stats 




                    year = pd.to_datetime(current_match['date'].iloc[0]).year
                    # Calculate average rank and top 11 count
                    features = {
                        'player': player,
                        'match_id': current_match['match_id'].iloc[0],
                        'date': current_match['date'].iloc[0],
                        'season': year,
                        'match_number': int(current_match['match_id'].iloc[0]) %  year,
                        'team': current_match['team'].iloc[0],
                        'opposition': current_match['opposition'].iloc[0],
                        'venue': current_venue,
                        'total_score': current_match['total_score'].iloc[0],
                        'batting_score': current_match['batting_score'].iloc[0],
                        'bowling_score': current_match['bowling_score'].iloc[0],
                        # Basic features
                        'avg_score': past_matches['total_score'].mean(),
                        'weighted_avg_score': weighted_avg_score,
                        'num_matches': len(past_matches),
                        'recent_score': recent_5_matches['total_score'].mean(),

                        'batting_innings': current_match['batting_innings'].iloc[0],
                        'bowling_innings': current_match['bowling_innings'].iloc[0],
                        
                        # Venue-specific features
                        'venue_avg_score': venue_avg_score,
                        'venue_std_score': venue_std_score,
                        'venue_recent_score': venue_recent_score,
                        'num_venue_matches': len(venue_matches),
                        'venue_batting_avg': venue_batting['batting_score'].mean() if len(venue_batting) > 0 else batting_avg,
                        'venue_bowling_avg': venue_bowling['bowling_score'].mean() if len(venue_bowling) > 0 else bowling_avg,
                        
                        # Opposition-specific features
                        'opposition_avg_score': opposition_avg_score,
                        'opposition_std_score': opposition_std_score,
                        'num_opposition_matches': len(opposition_matches),
                        'opposition_batting_avg': opposition_batting['batting_score'].mean() if len(opposition_batting) > 0 else batting_avg,
                        'opposition_bowling_avg': opposition_bowling['bowling_score'].mean() if len(opposition_bowling) > 0 else bowling_avg,
                        
                        # Venue-opposition interaction features
                        'venue_opposition_avg': venue_opposition_data['total_score'].mean() if len(venue_opposition_data) > 0 else past_matches['total_score'].mean(),
                        'venue_opposition_std': venue_opposition_data['total_score'].std() if len(venue_opposition_data) > 1 else past_matches['total_score'].std(),
                        
                        # Recent venue and opposition performance
                        'recent_venue_avg': recent_venue['total_score'].mean() if len(recent_venue) > 0 else venue_avg_score,
                        'recent_venue_std': recent_venue['total_score'].std() if len(recent_venue) > 1 else venue_std_score,
                        'recent_opposition_avg': recent_opposition['total_score'].mean() if len(recent_opposition) > 0 else opposition_avg_score,
                        'recent_opposition_std': recent_opposition['total_score'].std() if len(recent_opposition) > 1 else opposition_std_score,
                        
                        # Innings-specific features
                        'avg_1st_innings_bat_score': avg_1st_innings_bat_score,
                        'avg_2nd_innings_bat_score': avg_2nd_innings_bat_score,
                        'avg_1st_innings_bowl_score': avg_1st_innings_bowl_score,
                        'avg_2nd_innings_bowl_score': avg_2nd_innings_bowl_score,
                        
                        # Rolling statistics
                        'recent_5_avg': recent_5_matches['total_score'].mean(),
                        'recent_5_std': recent_5_matches['total_score'].std(),
                        'recent_10_avg': recent_10_matches['total_score'].mean(),
                        'recent_10_std': recent_10_matches['total_score'].std(),
                        'recent_20_avg': recent_20_matches['total_score'].mean(),
                        'recent_20_std': recent_20_matches['total_score'].std(),
                        
                        # Role-specific features
                        'batting_avg': batting_avg,
                        'bowling_avg': bowling_avg,
                        'batting_matches': len(batting_matches),
                        'bowling_matches': len(bowling_matches),
                        'recent_batting_avg': recent_batting_avg,
                        'recent_bowling_avg': recent_bowling_avg,
                        'average_balls_faced': average_balls_faced,
                        'average_balls_bowled': average_balls_bowled,
                        'average_balls_faced_at_venue': average_balls_faced_at_venue,
                        'average_balls_bowled_at_venue': average_balls_bowled_at_venue,
                        'strike_rate_batting': strike_rate_batting,
                        'strike_rate_bowling': strike_rate_bowling,
                        'economy_rate_bowling': economy_rate_bowling,
                        
                        # Consistency metrics
                        'score_std': score_std,
                        'recent_score_std': recent_score_std,
                        
                        # Form indicators
                        'form_trend': form_trend,
                        'avg_days_between_matches': avg_days_between_matches,
                        
                        # Historical rank-based features
                        'avg_rank_last_5': np.mean(past_match_ranks) if past_match_ranks else 12,
                        'top_11_count_last_5': sum(1 for rank in past_match_ranks if rank <= 11) if past_match_ranks else 0,

                        #total venue stats 
                        'total_venue_batting_avg_first_innings': total_venue_batting_avg_first_innings,
                        'total_venue_bowling_avg_first_innings': total_venue_bowling_avg_first_innings,
                        'total_venue_batting_avg_second_innings': total_venue_batting_avg_second_innings,
                        'total_venue_bowling_avg_second_innings': total_venue_bowling_avg_second_innings,
                    }
                    
                    # Add team composition features
                    #team_players = past_matches[past_matches['team'] == current_match['team'].iloc[0]]
                    #team_features = calculate_team_composition_features(current_match, team_players)
                    #features.update(team_features)
                    
                    # Add player correlation features
                    #correlation_features = calculate_player_correlation_features(current_match, team_players)
                    #features.update(correlation_features)
                    
                    features_df.append(features)
                except Exception as e:
                    print(f"Error processing match {i} for player {player}: {str(e)}")
                    traceback.print_exc()
                    continue
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing player {player}: {str(e)}")
            continue
    
    features_df = pd.DataFrame(features_df)
    print(features_df.head())
    print(f"\nFinal features DataFrame shape: {features_df.shape}")
    print(f"Unique players in features: {features_df['player'].nunique()}")
    print(f"Date range in features: {features_df['date'].min()} to {features_df['date'].max()}")
    
    # Verify all required features are present
    required_features = get_numeric_features() + get_categorical_features()

    missing_features = [f for f in required_features if f not in features_df.columns]
    if missing_features:
        print(f"Warning: Missing features after computation: {missing_features}")
        raise ValueError("Some required features are missing after computation")
    
    # Save features to cache file
    print(f"\nSaving features to {cache_file}...")
    features_df.to_csv(cache_file, index=False)
    print("Features saved successfully")
    
    return features_df

def calculate_team_composition_features(player_data, team_players_data):
    """
    Calculate features related to team composition and strength.
    """
    # Handle empty team_players_data case
    if len(team_players_data) == 0:
        return {
            'team_batting_strength': 0.0,
            'team_bowling_strength': 0.0,
            'team_all_rounder_count': 0,
            'team_recent_performance': 0.0,
            'team_balance_score': 1.0  # Neutral default value
        }
    
    # Get current match date
    current_date = pd.to_datetime(player_data['date'].iloc[0])
    
    # Filter team players data to only include matches before current match
    historical_team_players = team_players_data[pd.to_datetime(team_players_data['date']) < current_date]
    
    if len(historical_team_players) == 0:
        return {
            'team_batting_strength': 0.0,
            'team_bowling_strength': 0.0,
            'team_all_rounder_count': 0,
            'team_recent_performance': 0.0,
            'team_balance_score': 1.0
        }
    
    # Calculate team batting strength (average batting score)
    team_batting_strength = historical_team_players['batting_score'].mean()
    
    # Calculate team bowling strength (average bowling score)
    team_bowling_strength = historical_team_players['bowling_score'].mean()
    
    # Count all-rounders (players with both batting and bowling scores above threshold)
    team_all_rounder_count = len(historical_team_players[
        (historical_team_players['batting_score'] > 30) & 
        (historical_team_players['bowling_score'] > 30)
    ])
    
    # Calculate team's recent performance (last 5 matches before current match)
    team_recent_performance = historical_team_players.sort_values('date').tail(5)['total_score'].mean()
    
    # Calculate team balance score (ratio of batting to bowling strength)
    # Handle edge cases where either strength is 0 or very small
    if team_bowling_strength <= 0.1:  # If bowling strength is very small
        team_balance_score = 2.0 if team_batting_strength > 0 else 1.0  # Max value if batting is good, neutral if not
    else:
        team_balance_score = min(2.0, team_batting_strength / team_bowling_strength)  # Cap at 2.0
    
    return {
        'team_batting_strength': team_batting_strength,
        'team_bowling_strength': team_bowling_strength,
        'team_all_rounder_count': team_all_rounder_count,
        'team_recent_performance': team_recent_performance,
        'team_balance_score': team_balance_score
    }

def calculate_player_correlation_features(player_data, team_players_data):
    """
    Calculate features related to player correlations and team composition.
    """
    # Handle empty team_players_data case
    if len(team_players_data) == 0:
        return {
            'teammate_synergy_score': 0.5,  # Neutral default value
            'role_complementarity': 0.5,    # Neutral default value
            'batting_position_impact': 1.0   # Neutral default value
        }
    
    # Get current match date
    current_date = pd.to_datetime(player_data['date'].iloc[0])
    
    # Filter team players data to only include matches before current match
    historical_team_players = team_players_data[pd.to_datetime(team_players_data['date']) < current_date]
    
    if len(historical_team_players) == 0:
        return {
            'teammate_synergy_score': 0.5,
            'role_complementarity': 0.5,
            'batting_position_impact': 1.0
        }
    
    # Determine player's role based on historical performance
    player_historical = historical_team_players[historical_team_players['player'] == player_data['player'].iloc[0]]
    if len(player_historical) > 0:
        is_batting = player_historical['batting_score'].mean() > player_historical['bowling_score'].mean()
    else:
        # If no historical data, use a neutral value
        is_batting = True
    
    # Count players with complementary roles based on historical performance
    complementary_players = 0
    for teammate in historical_team_players['player'].unique():
        if teammate != player_data['player'].iloc[0]:
            teammate_data = historical_team_players[historical_team_players['player'] == teammate]
            teammate_is_batting = teammate_data['batting_score'].mean() > teammate_data['bowling_score'].mean()
            if teammate_is_batting != is_batting:
                complementary_players += 1
    
    # Calculate teammate synergy score based on complementary players
    teammate_synergy_score = complementary_players / len(historical_team_players['player'].unique())
    
    # Calculate role complementarity
    role_complementarity = complementary_players / len(historical_team_players['player'].unique())
    
    # Calculate batting position impact based on historical performance
    if len(player_historical) > 0:
        batting_score = player_historical['batting_score'].mean()
        bowling_score = player_historical['bowling_score'].mean()
        
        if bowling_score <= 0.1:  # If bowling score is very small
            batting_position_impact = 2.0 if batting_score > 0 else 1.0  # Max value if batting is good, neutral if not
        else:
            batting_position_impact = min(2.0, batting_score / bowling_score)  # Cap at 2.0
    else:
        batting_position_impact = 1.0  # Neutral default value
    
    return {
        'teammate_synergy_score': teammate_synergy_score,
        'role_complementarity': role_complementarity,
        'batting_position_impact': batting_position_impact
    }

def create_venue_stats(player_df):
    """
    Create venue stats for each venue in the dataframe
    """
    venue_features_df = pd.DataFrame()

    for match_id in player_df['match_id'].unique():
        match_data = player_df[player_df['match_id'] == match_id]
        venue = match_data['venue'].iloc[0]
        batting_score_first_innings = match_data[match_data['batting_innings'] == 1]['batting_score'].sum()
        batting_score_second_innings = match_data[match_data['batting_innings'] == 2]['batting_score'].sum()
        bowling_score_first_innings = match_data[match_data['bowling_innings'] == 1]['bowling_score'].sum()
        bowling_score_second_innings = match_data[match_data['bowling_innings'] == 2]['bowling_score'].sum()

        # Create a new row as a list of dictionaries
        new_row_data = [{
            'venue': venue,
            'date': match_data['date'].iloc[0],
            'batting_score_first_innings': batting_score_first_innings,
            'batting_score_second_innings': batting_score_second_innings,
            'bowling_score_first_innings': bowling_score_first_innings,
            'bowling_score_second_innings': bowling_score_second_innings
        }]

        # Create DataFrame from the list of dictionaries
        new_row_df = pd.DataFrame(new_row_data)
        
        # Concatenate with existing DataFrame
        venue_features_df = pd.concat([venue_features_df, new_row_df], ignore_index=True)

    venue_features_df.to_csv('venue_features.csv', index=False)
    return venue_features_df


def main():
    # Create dataset
    deliveries_df = create_player_dataset()
    
    # Create features for all matches
    features_df = create_features_and_train_test_split(deliveries_df)
    
    # Sort all data by date
    features_df = features_df.sort_values('date')
    
    if len(features_df) == 0:
        print("No data found")
        return
        
    print(f"\nFound {len(features_df)} total matches")
    print(f"Unique players: {features_df['player'].nunique()}")
    print(f"Date range: {features_df['date'].min()} to {features_df['date'].max()}")
    
    # Get the last matches for evaluation
    evaluation_matches = features_df.tail(30)
    print(f"\nEvaluating model on last {len(evaluation_matches)} matches from {evaluation_matches['date'].iloc[0].strftime('%Y-%m-%d')} to {evaluation_matches['date'].iloc[-1].strftime('%Y-%m-%d')}")
    
    # Store predictions and actuals for batting+fielding and bowling
    all_predictions_rf_batting = []
    all_actuals_batting = []
    
    all_predictions_rf_bowling = []
    all_actuals_bowling = []
    
    # For each of the last 10 matches
    for i, match in evaluation_matches.iterrows():
        # Get all matches before this one for training
        train_data = features_df[features_df['date'] < match['date']]
        
        if len(train_data) < 50:  # Require more training data since we're using all players
            continue
            
        # Get all players in this match
        match_players = features_df[features_df['match_id'] == match['match_id']]['player'].unique()
        
        # Prepare features for all players in this match
        numeric_features = get_numeric_features()
        categorical_features = get_categorical_features()

        # Filter out any numeric features that don't exist
        numeric_features = [col for col in numeric_features if col in features_df.columns]
        
        # Create dummy variables for categorical features
        X_train = pd.get_dummies(train_data[numeric_features + categorical_features], 
                                columns=categorical_features, drop_first=False)
        
        # Get test data for all players in this match
        match_data = features_df[features_df['match_id'] == match['match_id']]
        X_test = pd.get_dummies(match_data[numeric_features + categorical_features], 
                               columns=categorical_features, drop_first=False)
        
        # Handle missing columns more efficiently
        missing_cols = set(X_train.columns) - set(X_test.columns)
        if missing_cols:
            # Create a DataFrame with missing columns filled with zeros
            missing_df = pd.DataFrame(0, index=X_test.index, columns=list(missing_cols))
            # Concatenate with existing test data
            X_test = pd.concat([X_test, missing_df], axis=1)
            
        # Reorder columns in test data to match training data
        X_test = X_test[X_train.columns]
        
        # Handle NaN values in numeric features
        imputer = SimpleImputer(strategy='mean')
        
        # Get numeric feature columns
        numeric_cols = [col for col in X_train.columns if any(feat in col for feat in numeric_features)]
        
        # Fit imputer on training data and transform both sets
        X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])
        
        # Prepare target variables
        y_train_batting = train_data['batting_score']
        y_train_bowling = train_data['bowling_score']
        y_test_batting = match_data['batting_score']
        y_test_bowling = match_data['bowling_score']
        
        # Calculate time-based weights for training data
        days_diff = (match['date'] - train_data['date']).dt.days
        weights = np.exp(-0.005 * days_diff)  # Reduced decay factor to 0.005 for more gradual decay
        weights = weights / weights.sum()
        
        # Train and evaluate models for batting scores
        rf_model, _, _, _, _, _, _, _, rf_pred = train_and_evaluate_models(
            X_train, y_train_batting, X_test, y_test_batting, weights, "batting"
        )
        
        # Store predictions for batting scores
        all_predictions_rf_batting.extend(rf_pred)
        all_actuals_batting.extend(y_test_batting)
        
        # Train and evaluate models for bowling scores
        rf_model, _, _, _, _, _, _, _, rf_pred = train_and_evaluate_models(
            X_train, y_train_bowling, X_test, y_test_bowling, weights, "bowling"
        )
        
        # Store predictions for bowling scores
        all_predictions_rf_bowling.extend(rf_pred)
        all_actuals_bowling.extend(y_test_bowling)
    
    # Create summary tables
    def create_summary_table(actuals, predictions, score_type):
        results = []
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
        
        # Modified MAPE calculation to handle zero values
        actuals_array = np.array(actuals)
        predictions_array = np.array(predictions)
        # Only calculate MAPE for non-zero actuals
        non_zero_mask = actuals_array != 0
        if non_zero_mask.any():
            mape = np.mean(np.abs((actuals_array[non_zero_mask] - predictions_array[non_zero_mask]) / actuals_array[non_zero_mask])) * 100
        else:
            mape = 0  # If all actuals are zero, MAPE is 0
        
        r2 = r2_score(actuals, predictions)
        
        results.append({
            'Model': 'Random Forest',
            'MSE': f"{mse:.2f}",
            'RMSE': f"{rmse:.2f}",
            'MAE': f"{mae:.2f}",
            'MAPE': f"{mape:.2f}%",
            'R²': f"{r2:.4f}"
        })
        
        df = pd.DataFrame(results)
        df = df.set_index('Model')
        return df
    
    # Create summary tables for both batting+fielding and bowling
    batting_predictions = {'Random Forest': all_predictions_rf_batting}
    bowling_predictions = {'Random Forest': all_predictions_rf_bowling}
    
    # Create combined predictions
    combined_predictions = {'Random Forest': [b + f for b, f in zip(all_predictions_rf_batting, all_predictions_rf_bowling)]}
    
    # Create summary tables
    batting_summary = create_summary_table(all_actuals_batting, batting_predictions['Random Forest'], "batting")
    bowling_summary = create_summary_table(all_actuals_bowling, bowling_predictions['Random Forest'], "bowling")
    combined_summary = create_summary_table(
        [b + f for b, f in zip(all_actuals_batting, all_actuals_bowling)],
        combined_predictions['Random Forest'],
        "combined"
    )
    
    # Print summary tables
    print("\nModel Performance Summary:\n")
    print("Batting Performance:")
    print(batting_summary)
    print("\nBowling Performance:")
    print(bowling_summary)
    print("\nCombined Performance (Batting + Bowling):")
    print(combined_summary)
    
    # Print sample size information
    print(f"\nSample Sizes:")
    print(f"Total matches evaluated: {len(evaluation_matches)}")
    print(f"Total player predictions: {len(all_actuals_batting + all_actuals_bowling)}")
    print(f"Batting predictions: {len(all_actuals_batting)}")
    print(f"Bowling predictions: {len(all_actuals_bowling)}")

if __name__ == "__main__":
    main()