import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
import pickle
import os
from datetime import datetime
from feature_utils import get_numeric_features, get_categorical_features
import shap
import matplotlib.pyplot as plt
import traceback
import yaml
import argparse

class MatchPredictor:
    def __init__(self, model_path='models', player_mapping_path='player_mapping.csv'):
        """
        Initialize the predictor with trained models and player mapping
        """
        self.player_mapping = pd.read_csv(player_mapping_path)
        self.model_path = model_path
        
        # Load the trained models
        self.batting_xgb = pickle.load(open(os.path.join(model_path, 'batting_xgb_model.pkl'), 'rb'))
        self.batting_rf = pickle.load(open(os.path.join(model_path, 'batting_rf_model.pkl'), 'rb'))
        self.batting_mlp = pickle.load(open(os.path.join(model_path, 'batting_mlp_model.pkl'), 'rb'))
        self.batting_ridge = pickle.load(open(os.path.join(model_path, 'batting_ridge_model.pkl'), 'rb'))
        
        self.bowling_xgb = pickle.load(open(os.path.join(model_path, 'bowling_xgb_model.pkl'), 'rb'))
        self.bowling_rf = pickle.load(open(os.path.join(model_path, 'bowling_rf_model.pkl'), 'rb'))
        self.bowling_mlp = pickle.load(open(os.path.join(model_path, 'bowling_mlp_model.pkl'), 'rb'))
        self.bowling_ridge = pickle.load(open(os.path.join(model_path, 'bowling_ridge_model.pkl'), 'rb'))
        
        self.rank_xgb = pickle.load(open(os.path.join(model_path, 'rank_xgb_model.pkl'), 'rb'))
        self.rank_rf = pickle.load(open(os.path.join(model_path, 'rank_rf_model.pkl'), 'rb'))
        self.rank_mlp = pickle.load(open(os.path.join(model_path, 'rank_mlp_model.pkl'), 'rb'))
        self.rank_ridge = pickle.load(open(os.path.join(model_path, 'rank_ridge_model.pkl'), 'rb'))
        
        # Load the captain model
        self.captain_rf = pickle.load(open(os.path.join(model_path, 'captain_rf_model.pkl'), 'rb'))
        
        # Load the scaler and imputer
        self.scaler = pickle.load(open(os.path.join(model_path, 'scaler.pkl'), 'rb'))
        self.imputer = pickle.load(open(os.path.join(model_path, 'imputer.pkl'), 'rb'))
        
        # Load feature names
        self.feature_names = pickle.load(open(os.path.join(model_path, 'feature_names.pkl'), 'rb'))
        
        # Load the features DataFrame for historical data
        self.features_df = pd.read_csv('features_df.csv')
        self.features_df['date'] = pd.to_datetime(self.features_df['date'])
        
        # Get feature lists from feature_utils
        self.numeric_features = get_numeric_features()
        self.categorical_features = get_categorical_features()
        
        # Load venue information
        self.venues = self.features_df['venue'].unique().tolist()
        
        # Load ensemble weights from training
        self.ensemble_weights = {
            'batting': {'xgb': 0.25, 'rf': 0.25, 'mlp': 0.25, 'ridge': 0.25},
            'bowling': {'xgb': 0.25, 'rf': 0.25, 'mlp': 0.25, 'ridge': 0.25},
            'rank': {'xgb': 0.5, 'rf': 0.5, 'mlp': 0.0, 'ridge': 0.0}
        }
    
    def get_player_features(self, player, match_id, features_df, match_data):
        """
        Get features for a player in a specific match using only historical data.
        """
        # Get all historical matches for this player
        player_history = features_df[features_df['player'] == player].copy()
        
        if len(player_history) == 0:
            raise ValueError(f"No historical data found for player {player}")
        
        # Get the most recent features for this player
        features = player_history.iloc[-1].to_dict()

        current_match = match_data[match_data['match_id'] == match_id].iloc[0]
        venue_features = features_df[(features_df['venue'] == current_match['venue']) & (features_df['date'] < current_match['date'])]

        if len(venue_features) > 0:
            venue_features = venue_features.iloc[-1].to_dict()
            features.update({
                'total_venue_batting_avg_first_innings': venue_features['total_venue_batting_avg_first_innings'],
                'total_venue_bowling_avg_first_innings': venue_features['total_venue_bowling_avg_first_innings'],
                'total_venue_batting_avg_second_innings': venue_features['total_venue_batting_avg_second_innings'],
                'total_venue_bowling_avg_second_innings': venue_features['total_venue_bowling_avg_second_innings']
            })


        # Determine which team the player belongs to
        player_team = None
        player_mapping = self.player_mapping[self.player_mapping['features_player'] == player]
        if len(player_mapping) == 0:
            raise ValueError(f"No mapping found for player {player}")
            
        roster_name = player_mapping['roster_name'].iloc[0]
        if roster_name in current_match['playing_11_team1']:
            player_team = current_match['team1']
        elif roster_name in current_match['playing_11_team2']:
            player_team = current_match['team2']
        else:
            raise ValueError(f"Player {roster_name} not found in either team")

        # Assign innings based on player's team and who's batting first
        batting_innings = 1 if player_team == current_match['batting_first'] else 2
        bowling_innings = 2 if player_team == current_match['batting_first'] else 1

        features.update({
            'team': player_team,
            'opposition': current_match['team2'] if player_team == current_match['team1'] else current_match['team1'],
            'venue': current_match['venue'],
            'batting_innings': batting_innings,
            'bowling_innings': bowling_innings
        })
        
        return features
    
    def predict_match(self, team1, team2, batting_first, playing_11_team1, playing_11_team2, venue, match_date=None):
        """
        Predict scores for all players in both teams using ensemble of models
        Args:
            team1: Name of first team
            team2: Name of second team
            batting_first: Name of team batting first
            playing_11_team1: List of 11 players for first team
            playing_11_team2: List of 11 players for second team
            venue: Name of the venue where the match is being played
            match_date: Date of the match (default: current date)
        """
        # Use provided match date or current date
        if match_date is None:
            match_date = pd.Timestamp.now()
        else:
            match_date = pd.to_datetime(match_date)

        if batting_first == team1:
            batting_innings = 1
            bowling_innings = 2
        else:
            batting_innings = 2
            bowling_innings = 1

        match_data = pd.DataFrame([{
            'match_id': 'current_match',
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'batting_first': batting_first,
            'batting_innings': batting_innings,
            'bowling_innings': bowling_innings,
            'date': match_date,
            'playing_11_team1': playing_11_team1,
            'playing_11_team2': playing_11_team2
        }])
        
        # Determine batting and bowling teams based on who's batting first
        if batting_first == team1:
            batting_team = team1
            bowling_team = team2
            batting_players = playing_11_team1
            bowling_players = playing_11_team2
        else:
            batting_team = team2
            bowling_team = team1
            batting_players = playing_11_team2
            bowling_players = playing_11_team1
        
        predictions = []
        
        # Process batting team (first innings)
        for player in batting_players:
            # Get player's feature name from mapping
            player_mapping = self.player_mapping[self.player_mapping['roster_name'] == player]
            if len(player_mapping) == 0:
                print(f"Warning: No mapping found for player {player}")
                continue
                
            feature_name = player_mapping['features_player'].iloc[0]
            
            # Get features for the player
            features = self.get_player_features(feature_name, 'current_match', self.features_df, match_data)
            if features is None:
                print(f"Warning: No historical data found for player {player}")
                continue
            
            # Create feature vector
            feature_vector = pd.DataFrame([features])
            
            # Create dummy variables for categorical features (excluding 'player')
            categorical_features = [col for col in self.categorical_features if col != 'player']
            X = pd.get_dummies(feature_vector[self.numeric_features + categorical_features], 
                             columns=categorical_features, drop_first=False)
            
            # Handle missing columns more efficiently
            missing_cols = set(self.feature_names) - set(X.columns)
            if missing_cols:
                # Create a DataFrame with missing columns filled with zeros
                missing_df = pd.DataFrame(0, index=X.index, columns=list(missing_cols))
                # Concatenate with existing data
                X = pd.concat([X, missing_df], axis=1)
            
            # Reorder columns to match training data
            X = X[self.feature_names]
            
            # Scale numeric features
            numeric_cols = [col for col in X.columns if any(feat in col for feat in self.numeric_features)]
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
            
            # Make predictions based on role
            batting_score = 0
            bowling_score = 0

            #print all the columns which contains NaNs
            # Get columns containing NaN values
            cols_with_nan = X.columns[X.isna().any()].tolist()

            # Print the column names with NaN values
            # Optional: Print the count of NaN values in each column
            for col in cols_with_nan:
                nan_count = X[col].isna().sum()
                print(f"Column '{col}' has {nan_count} NaN values")


            X.to_csv(f'debug/{player}.csv')
                    
            role = player_mapping['role'].iloc[0]
            if role in ['Batter', 'Wicketkeeper/Batter', 'All-rounder']:
                # Get predictions from all models
                xgb_batting = self.batting_xgb.predict(X)[0]
                rf_batting = self.batting_rf.predict(X)[0]
                mlp_batting = self.batting_mlp.predict(X)[0]
                ridge_batting = self.batting_ridge.predict(X)[0]
                
                # Combine predictions using ensemble weights
                batting_score = (
                    xgb_batting * self.ensemble_weights['batting']['xgb'] +
                    rf_batting * self.ensemble_weights['batting']['rf'] +
                    mlp_batting * self.ensemble_weights['batting']['mlp'] +
                    ridge_batting * self.ensemble_weights['batting']['ridge']
                )
            
            if role in ['Bowler', 'All-rounder']:
                # Get predictions from all models
                xgb_bowling = self.bowling_xgb.predict(X)[0]
                rf_bowling = self.bowling_rf.predict(X)[0]
                mlp_bowling = self.bowling_mlp.predict(X)[0]
                ridge_bowling = self.bowling_ridge.predict(X)[0]
                
                # Combine predictions using ensemble weights
                bowling_score = (
                    xgb_bowling * self.ensemble_weights['bowling']['xgb'] +
                    rf_bowling * self.ensemble_weights['bowling']['rf'] +
                    mlp_bowling * self.ensemble_weights['bowling']['mlp'] +
                    ridge_bowling * self.ensemble_weights['bowling']['ridge']
                )
            
            # Get rank predictions from all models
            xgb_rank = self.rank_xgb.predict(X)[0]
            rf_rank = self.rank_rf.predict(X)[0]
            mlp_rank = self.rank_mlp.predict(X)[0]
            ridge_rank = self.rank_ridge.predict(X)[0]
            
            # Combine rank predictions using ensemble weights
            predicted_rank = (
                xgb_rank * self.ensemble_weights['rank']['xgb'] +
                rf_rank * self.ensemble_weights['rank']['rf'] +
                mlp_rank * self.ensemble_weights['rank']['mlp'] +
                ridge_rank * self.ensemble_weights['rank']['ridge']
            )
            
            # Get captaincy score from the trained model
            captaincy_score = self.captain_rf.predict(X)[0]
            
            predictions.append({
                'player': player,
                'team': batting_team,
                'role': role,
                'batting_score': batting_score,
                'bowling_score': bowling_score,
                'total_score': batting_score + bowling_score,
                'predicted_rank': predicted_rank,
                'captaincy_score': captaincy_score
            })
        
        # Process bowling team
        for player in bowling_players:
            # Get player's feature name from mapping
            player_mapping = self.player_mapping[self.player_mapping['roster_name'] == player]
            if len(player_mapping) == 0:
                print(f"Warning: No mapping found for player {player}")
                continue
                
            feature_name = player_mapping['features_player'].iloc[0]
            
            # Get features for the player
            features = self.get_player_features(feature_name, 'current_match', self.features_df, match_data)
            if features is None:
                print(f"Warning: No historical data found for player {player}")
                continue
            
            # Create feature vector
            feature_vector = pd.DataFrame([features])
            
            # Create dummy variables for categorical features (excluding 'player')
            categorical_features = [col for col in self.categorical_features if col != 'player']
            X = pd.get_dummies(feature_vector[self.numeric_features + categorical_features], 
                             columns=categorical_features, drop_first=False)
            
            # Handle missing columns more efficiently
            missing_cols = set(self.feature_names) - set(X.columns)
            if missing_cols:
                # Create a DataFrame with missing columns filled with zeros
                missing_df = pd.DataFrame(0, index=X.index, columns=list(missing_cols))
                # Concatenate with existing data
                X = pd.concat([X, missing_df], axis=1)

            #print nan columns
            print("NAN columns: ", X.columns[X.isna().any()].tolist())

            
            # Reorder columns to match training data
            X = X[self.feature_names]
            
            # Scale numeric features
            numeric_cols = [col for col in X.columns if any(feat in col for feat in self.numeric_features)]
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
            
            # Make predictions based on role
            batting_score = 0
            bowling_score = 0
            
            role = player_mapping['role'].iloc[0]
            if role in ['Batter', 'Wicketkeeper/Batter', 'All-rounder']:
                # Get predictions from all models
                xgb_batting = self.batting_xgb.predict(X)[0]
                rf_batting = self.batting_rf.predict(X)[0]
                mlp_batting = self.batting_mlp.predict(X)[0]
                ridge_batting = self.batting_ridge.predict(X)[0]
                
                # Combine predictions using ensemble weights
                batting_score = (
                    xgb_batting * self.ensemble_weights['batting']['xgb'] +
                    rf_batting * self.ensemble_weights['batting']['rf'] +
                    mlp_batting * self.ensemble_weights['batting']['mlp'] +
                    ridge_batting * self.ensemble_weights['batting']['ridge']
                )
            
            if role in ['Bowler', 'All-rounder']:
                # Get predictions from all models
                xgb_bowling = self.bowling_xgb.predict(X)[0]
                rf_bowling = self.bowling_rf.predict(X)[0]
                mlp_bowling = self.bowling_mlp.predict(X)[0]
                ridge_bowling = self.bowling_ridge.predict(X)[0]
                
                # Combine predictions using ensemble weights
                bowling_score = (
                    xgb_bowling * self.ensemble_weights['bowling']['xgb'] +
                    rf_bowling * self.ensemble_weights['bowling']['rf'] +
                    mlp_bowling * self.ensemble_weights['bowling']['mlp'] +
                    ridge_bowling * self.ensemble_weights['bowling']['ridge']
                )
            
            # Get rank predictions from all models
            xgb_rank = self.rank_xgb.predict(X)[0]
            rf_rank = self.rank_rf.predict(X)[0]
            mlp_rank = self.rank_mlp.predict(X)[0]
            ridge_rank = self.rank_ridge.predict(X)[0]
            
            # Combine rank predictions using ensemble weights
            predicted_rank = (
                xgb_rank * self.ensemble_weights['rank']['xgb'] +
                rf_rank * self.ensemble_weights['rank']['rf'] +
                mlp_rank * self.ensemble_weights['rank']['mlp'] +
                ridge_rank * self.ensemble_weights['rank']['ridge']
            )
            
            # Get captaincy score from the trained model
            captaincy_score = self.captain_rf.predict(X)[0]
            
            predictions.append({
                'player': player,
                'team': bowling_team,
                'role': role,
                'batting_score': batting_score,
                'bowling_score': bowling_score,
                'total_score': batting_score + bowling_score,
                'predicted_rank': predicted_rank,
                'captaincy_score': captaincy_score
            })
        
        # Convert predictions to DataFrame and sort by total score
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('total_score', ascending=False)
        
        # Add actual rank based on total score
        predictions_df['actual_rank'] = predictions_df['total_score'].rank(ascending=False, method='min').astype(int)
        
        # Add impact score (total_score / predicted_rank)
        predictions_df['impact_score'] = predictions_df['total_score'] / predictions_df['predicted_rank']
        
        # Add total expected value (combines impact score and total score)
        predictions_df['total_expected_value'] = predictions_df['impact_score'] * predictions_df['total_score']
        
        return predictions_df

def diagnose_player_prediction(predictor, player_name, match_id):
    """
    Diagnose a player's prediction by analyzing feature contributions.
    """
    # Get the player's data for this match
    player_data = predictor.features_df[
        (predictor.features_df['player'] == player_name) & 
        (predictor.features_df['match_id'] == match_id)
    ]
    
    if len(player_data) == 0:
        print(f"No data found for player {player_name} in match {match_id}")
        return
    
    # Get match context
    match_context = {
        'player': player_name,
        'team': player_data['team'].iloc[0],
        'opposition': player_data['opposition'].iloc[0],
        'venue': player_data['venue'].iloc[0],
        'date': player_data['date'].iloc[0]
    }
    
    # Get predictions
    predictions = predictor.predict_match(
        team1=match_context['team'],
        team2=match_context['opposition'],
        batting_first=match_context['team'],  # Use team as default batting first
        playing_11_team1=[player_name],  # Only include the player we're diagnosing
        playing_11_team2=[match_context['opposition']],  # Dummy opposition player
        venue=match_context['venue']
    )
    
    # Get player's prediction
    player_pred = predictions[predictions['player'] == player_name].iloc[0]
    
    print("\nMatch Context:")
    print(f"Player: {match_context['player']}")
    print(f"Team: {match_context['team']}")
    print(f"Opposition: {match_context['opposition']}")
    print(f"Venue: {match_context['venue']}")
    print(f"Date: {match_context['date']}")
    
    print("\nPredicted Scores:")
    print(f"Batting Score: {player_pred['batting_score']:.2f}")
    print(f"Bowling Score: {player_pred['bowling_score']:.2f}")
    print(f"Total Score: {player_pred['total_score']:.2f}")
    print(f"Predicted Rank: {player_pred['predicted_rank']}")
    
    # Prepare features for SHAP analysis
    # Get numeric and categorical features
    numeric_features = [col for col in predictor.numeric_features if col in player_data.columns]
    categorical_features = [col for col in predictor.categorical_features if col in player_data.columns]
    
    # Create feature vector
    feature_vector = player_data[numeric_features + categorical_features].copy()
    
    # Create dummy variables for categorical features
    X = pd.get_dummies(feature_vector, columns=categorical_features, drop_first=False)
    
    # Handle missing columns
    missing_cols = set(predictor.feature_names) - set(X.columns)
    if missing_cols:
        # Create a DataFrame with missing columns filled with zeros
        missing_df = pd.DataFrame(0, index=X.index, columns=list(missing_cols))
        # Concatenate with existing data
        X = pd.concat([X, missing_df], axis=1)
    
    # Reorder columns to match training data
    X = X[predictor.feature_names]
    
    # Scale numeric features
    numeric_cols = [col for col in X.columns if any(feat in col for feat in predictor.numeric_features)]
    X[numeric_cols] = predictor.scaler.transform(X[numeric_cols])
    
    # Calculate SHAP values for batting
    explainer_batting = shap.TreeExplainer(predictor.batting_xgb)
    shap_values_batting = explainer_batting.shap_values(X)
    
    # Calculate SHAP values for bowling
    explainer_bowling = shap.TreeExplainer(predictor.bowling_xgb)
    shap_values_bowling = explainer_bowling.shap_values(X)
    
    # Create DataFrames for top features
    batting_features = pd.DataFrame({
        'feature': predictor.feature_names,
        'contribution': np.abs(shap_values_batting[0])
    }).sort_values('contribution', ascending=False).head(10)
    
    bowling_features = pd.DataFrame({
        'feature': predictor.feature_names,
        'contribution': np.abs(shap_values_bowling[0])
    }).sort_values('contribution', ascending=False).head(10)
    
    print("\nTop 10 Features Contributing to Batting Score:")
    print(batting_features)
    
    print("\nTop 10 Features Contributing to Bowling Score:")
    print(bowling_features)
    
    # Create SHAP plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    shap.summary_plot(shap_values_batting, X, plot_type="bar", show=False)
    plt.title("Feature Contributions to Batting Score")
    
    plt.subplot(1, 2, 2)
    shap.summary_plot(shap_values_bowling, X, plot_type="bar", show=False)
    plt.title("Feature Contributions to Bowling Score")
    
    plt.tight_layout()
    plt.savefig(f'diagnosis_{player_name}_{match_id}.png')
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict IPL match outcomes using trained models')
    parser.add_argument('--config', type=str, default='matches/sample_match.yaml',
                      help='Path to the YAML configuration file')
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MatchPredictor()
    
    try:
        # Load match configuration from YAML
        with open(args.config, 'r') as file:
            match_config = yaml.safe_load(file)
        
        # Extract values from config
        team1 = match_config['teams']['team1']
        team2 = match_config['teams']['team2']
        batting_first = match_config['teams']['batting_first']
        venue = match_config['teams']['venue']
        playing_11_team1 = match_config['playing_11']['team1']
        playing_11_team2 = match_config['playing_11']['team2']
        match_date = match_config.get('match_date')  # Get match date from config
        
        # Make predictions using the loaded configuration
        predictions = predictor.predict_match(
            team1=team1,
            team2=team2,
            batting_first=batting_first,
            playing_11_team1=playing_11_team1,
            playing_11_team2=playing_11_team2,
            venue=venue,
            match_date=match_date
        )
        
        # Print overall predictions sorted by total score
        print(f"\nOverall Predictions for {team1} vs {team2} (Sorted by Total Score):")
        print("--------------------------------")
        print(predictions[['player', 'team', 'role', 'batting_score', 'bowling_score', 'total_score', 'predicted_rank', 'captaincy_score']].to_string(index=False))
        
        # Print predictions by role
        roles = ['Batter', 'Bowler', 'All-rounder', 'Wicketkeeper/Batter']
        for role in roles:
            role_predictions = predictions[predictions['role'] == role]
            if len(role_predictions) > 0:
                print(f"\n{role}s (Sorted by Total Score):")
                print("--------------------------------")
                role_sorted = role_predictions.sort_values('total_score', ascending=False)
                print(role_sorted[['player', 'team', 'batting_score', 'bowling_score', 'total_score', 'predicted_rank', 'captaincy_score']].to_string(index=False))
        
        # Print team totals
        print("\nTeam Totals:")
        print(predictions.groupby('team')['total_score'].sum())
        
        # Print top 3 captain recommendations by role
        print("\nTop Captain Recommendations by Role:")
        for role in roles:
            role_predictions = predictions[predictions['role'] == role]
            if len(role_predictions) > 0:
                print(f"\n{role}s (Top 3 by Captaincy Score):")
                print("--------------------------------")
                role_captains = role_predictions.nlargest(3, 'captaincy_score')
                for idx, row in role_captains.iterrows():
                    print(f"{row['player']} ({row['team']}) - Total Score: {row['total_score']:.2f}, Predicted Rank: {row['predicted_rank']:.1f}, Captaincy Score: {row['captaincy_score']:.2f}")
        
        # Print top 3 value picks by role (based on impact_score)
        print("\nTop Value Picks by Role (Based on Impact Score):")
        for role in roles:
            role_predictions = predictions[predictions['role'] == role]
            if len(role_predictions) > 0:
                print(f"\n{role}s (Top 3 by Impact Score):")
                print("--------------------------------")
                role_value = role_predictions.nlargest(3, 'impact_score')
                for idx, row in role_value.iterrows():
                    print(f"{row['player']} ({row['team']}) - Total Score: {row['total_score']:.2f}, Impact Score: {row['impact_score']:.2f}, Predicted Rank: {row['predicted_rank']:.1f}")
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 
