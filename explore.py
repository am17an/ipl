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
from feature_utils import get_numeric_features, get_categorical_features

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
                return create_features_from_scratch(deliveries_df, cache_file)
            
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
                    
                    # Calculate innings-specific batting and bowling averages
                    first_innings_batting = past_matches[
                        (past_matches['batting_innings'] == 1) & 
                        (past_matches['batting_score'] > 0)
                    ]
                    second_innings_batting = past_matches[
                        (past_matches['batting_innings'] == 2) & 
                        (past_matches['batting_score'] > 0)
                    ]
                    
                    first_innings_bowling = past_matches[
                        (past_matches['bowling_innings'] == 1) & 
                        (past_matches['bowling_score'] > 0)
                    ]
                    second_innings_bowling = past_matches[
                        (past_matches['bowling_innings'] == 2) & 
                        (past_matches['bowling_score'] > 0)
                    ]
                    
                    avg_1st_innings_bat_score = first_innings_batting['batting_score'].mean() if len(first_innings_batting) > 0 else past_matches['batting_score'].mean()
                    avg_2nd_innings_bat_score = second_innings_batting['batting_score'].mean() if len(second_innings_batting) > 0 else past_matches['batting_score'].mean()
                    avg_1st_innings_bowl_score = first_innings_bowling['bowling_score'].mean() if len(first_innings_bowling) > 0 else past_matches['bowling_score'].mean()
                    avg_2nd_innings_bowl_score = second_innings_bowling['bowling_score'].mean() if len(second_innings_bowling) > 0 else past_matches['bowling_score'].mean()
                    
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

                    batting_avg = batting_matches['batting_score'].mean() if len(batting_matches) > 0 else past_matches['batting_score'].mean()
                    bowling_avg = bowling_matches['bowling_score'].mean() if len(bowling_matches) > 0 else past_matches['bowling_score'].mean()
                    
                    # Calculate role-specific recent performance
                    recent_batting_matches = batting_matches.tail(5)
                    recent_bowling_matches = bowling_matches.tail(5)
                    
                    recent_batting_avg = recent_batting_matches['batting_score'].mean() if len(recent_batting_matches) > 0 else batting_avg
                    recent_bowling_avg = recent_bowling_matches['bowling_score'].mean() if len(recent_bowling_matches) > 0 else bowling_avg

                    # Calculate average balls faced and bowled
                    average_balls_faced = batting_matches.apply(lambda x: x['batting_stats'].get('balls', 0), axis=1).mean() if len(batting_matches) > 0 else 0
                    average_balls_bowled = bowling_matches.apply(lambda x: x['bowling_stats'].get('balls', 0), axis=1).mean() if len(bowling_matches) > 0 else 0

                    # Calculate venue-specific average balls faced and bowled
                    average_balls_faced_at_venue = venue_matches.apply(lambda x: x['batting_stats'].get('balls', 0), axis=1).mean() if len(venue_matches) > 0 else average_balls_faced
                    average_balls_bowled_at_venue = venue_matches.apply(lambda x: x['bowling_stats'].get('balls', 0), axis=1).mean() if len(venue_matches) > 0 else average_balls_bowled

                    # Calculate strike rates and economy rate
                    def calculate_batting_strike_rate(matches):
                        if len(matches) == 0:
                            return 0
                        total_runs = matches.apply(lambda x: x['batting_stats'].get('runs', 0), axis=1).sum()
                        total_balls = matches.apply(lambda x: x['batting_stats'].get('balls', 0), axis=1).sum()
                        return total_runs / total_balls if total_balls > 0 else 0

                    def calculate_bowling_strike_rate(matches):
                        if len(matches) == 0:
                            return 0
                        total_balls = matches.apply(lambda x: x['bowling_stats'].get('balls', 0), axis=1).sum()
                        total_wickets = matches.apply(lambda x: x['bowling_stats'].get('wickets', 0), axis=1).sum()
                        return total_balls / total_wickets if total_wickets > 0 else 100.0

                    def calculate_economy_rate(matches):
                        if len(matches) == 0:
                            return 0
                        total_runs = matches.apply(lambda x: x['bowling_stats'].get('runs', 0), axis=1).sum()
                        total_balls = matches.apply(lambda x: x['bowling_stats'].get('balls', 0), axis=1).sum()
                        return total_runs / total_balls if total_balls > 0 else 20

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
                                match_ranks = historical_match_players['total_score'].rank(ascending=False, method='min').astype(int)
                                player_rank = match_ranks[historical_match_players['player'] == player].iloc[0] if player in historical_match_players['player'].values else 12
                                past_match_ranks.append(player_rank)
                    
                    features = {
                        'player': player,
                        'match_id': current_match['match_id'].iloc[0],
                        'date': current_match['date'].iloc[0],
                        'season': current_match['season'].iloc[0] if 'season' in current_match.columns else None,
                        'team': current_match['team'].iloc[0],
                        'opposition': current_match['opposition'].iloc[0],
                        'venue': current_venue,
                        'total_score': current_match['total_score'].iloc[0],  # This is the target variable
                        'batting_score': current_match['batting_score'].iloc[0],  # This is the target variable
                        'bowling_score': current_match['bowling_score'].iloc[0],  # This is the target variable
                        # Basic features
                        'avg_score': past_matches['total_score'].mean(),
                        'weighted_avg_score': weighted_avg_score,
                        'num_matches': len(past_matches),
                        'recent_score': recent_5_matches['total_score'].mean(),
                        
                        # Venue-specific features
                        'venue_avg_score': venue_avg_score,
                        'venue_std_score': venue_std_score,
                        'venue_recent_score': venue_recent_score,
                        'num_venue_matches': len(venue_matches),
                        
                        # Opposition-specific features
                        'opposition_avg_score': opposition_avg_score,
                        'opposition_std_score': opposition_std_score,
                        'num_opposition_matches': len(opposition_matches),
                        
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
                        'top_11_count_last_5': sum(1 for rank in past_match_ranks if rank <= 11) if past_match_ranks else 0
                    }
                    
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

def train_and_evaluate_models(X_train, y_train, X_test, y_test, weights, prediction_type):
    """
    Train and evaluate Random Forest model with optimized parameters
    prediction_type can be 'batting' or 'bowling'
    """
    # Calculate bounds for clamping based on training data
    def calculate_bounds(y, iqr_multiplier):
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        return lower_bound, upper_bound
    
    # Use different IQR multipliers for batting and bowling
    if prediction_type == "batting":
        # Use a larger multiplier for batting to allow for more extreme scores
        lower_bound, upper_bound = calculate_bounds(y_train, iqr_multiplier=2.5)
    else:
        # Use a smaller multiplier for bowling since scores are more consistent
        lower_bound, upper_bound = calculate_bounds(y_train, iqr_multiplier=1.5)
    
    # Clamp training and test data
    y_train_clamped = y_train.clip(lower=lower_bound, upper=upper_bound)
    y_test_clamped = y_test.clip(lower=lower_bound, upper=upper_bound)
    
    # Train Random Forest model with optimized parameters
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train_clamped, sample_weight=weights)
    
    # Get predictions and clamp them
    rf_test_pred = rf_model.predict(X_test)
    rf_test_pred = pd.Series(rf_test_pred).clip(lower=lower_bound, upper=upper_bound)
    
    return rf_model, rf_test_pred, rf_test_pred, rf_test_pred, rf_test_pred, rf_test_pred, rf_test_pred, rf_test_pred, rf_test_pred

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
        numeric_features = [
            'avg_score', 'weighted_avg_score', 'num_matches', 'recent_score', 
            'venue_avg_score', 'venue_std_score', 'venue_recent_score', 'num_venue_matches',
            'opposition_avg_score', 'opposition_std_score', 'num_opposition_matches',
            'avg_1st_innings_bat_score', 'avg_2nd_innings_bat_score', 'avg_1st_innings_bowl_score', 'avg_2nd_innings_bowl_score',
            'recent_5_avg', 'recent_5_std', 'recent_10_avg', 'recent_10_std',
            'recent_20_avg', 'recent_20_std', 'batting_avg', 'bowling_avg',
            'batting_matches', 'bowling_matches', 'recent_batting_avg', 'recent_bowling_avg',
            'score_std', 'recent_score_std', 'form_trend', 'avg_days_between_matches'
        ]
        categorical_features = ['team', 'opposition', 'venue', 'player']
        
        # Filter out any numeric features that don't exist
        numeric_features = [col for col in numeric_features if col in features_df.columns]
        
        # Create dummy variables for categorical features
        X_train = pd.get_dummies(train_data[numeric_features + categorical_features], 
                                columns=categorical_features, drop_first=True)
        
        # Get test data for all players in this match
        match_data = features_df[features_df['match_id'] == match['match_id']]
        X_test = pd.get_dummies(match_data[numeric_features + categorical_features], 
                               columns=categorical_features, drop_first=True)
        
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