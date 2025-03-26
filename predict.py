import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import os
from datetime import datetime
from feature_utils import get_numeric_features, get_categorical_features, calculate_player_features

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
        
        self.bowling_xgb = pickle.load(open(os.path.join(model_path, 'bowling_xgb_model.pkl'), 'rb'))
        self.bowling_rf = pickle.load(open(os.path.join(model_path, 'bowling_rf_model.pkl'), 'rb'))
        self.bowling_mlp = pickle.load(open(os.path.join(model_path, 'bowling_mlp_model.pkl'), 'rb'))
        
        self.rank_xgb = pickle.load(open(os.path.join(model_path, 'rank_xgb_model.pkl'), 'rb'))
        self.rank_rf = pickle.load(open(os.path.join(model_path, 'rank_rf_model.pkl'), 'rb'))
        self.rank_mlp = pickle.load(open(os.path.join(model_path, 'rank_mlp_model.pkl'), 'rb'))
        
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
            'batting': {'xgb': 0.33, 'rf': 0.33, 'mlp': 0.34},
            'bowling': {'xgb': 0.33, 'rf': 0.33, 'mlp': 0.34},
            'rank': {'xgb': 0.33, 'rf': 0.33, 'mlp': 0.34}
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
        
        # Get current match information
        current_match = match_data[match_data['match_id'] == match_id].iloc[0]
        
        # Update only the current match information
        features.update({
            'team': current_match['team'],
            'opposition': current_match['opposition'],
            'venue': current_match['venue'],
            'batting_first': 1 if current_match['team'] == current_match['batting_first'] else 0
        })
        
        return features
    
    def predict_match(self, team1, team2, batting_first, playing_11_team1, playing_11_team2, venue):
        """
        Predict scores for all players in both teams using ensemble of models
        Args:
            team1: Name of first team
            team2: Name of second team
            batting_first: Name of team batting first
            playing_11_team1: List of 11 players for first team
            playing_11_team2: List of 11 players for second team
            venue: Name of the venue where the match is being played
        """
        # Create a temporary match data DataFrame for the current match
        match_data = pd.DataFrame([{
            'match_id': 'current_match',
            'team': team1,
            'opposition': team2,
            'venue': venue,
            'batting_first': batting_first
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
                             columns=categorical_features, drop_first=True)
            
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
            
            role = player_mapping['role'].iloc[0]
            if role in ['Batter', 'Wicketkeeper/Batter', 'All-rounder']:
                # Get predictions from all models
                xgb_batting = self.batting_xgb.predict(X)[0]
                rf_batting = self.batting_rf.predict(X)[0]
                mlp_batting = self.batting_mlp.predict(X)[0]
                
                # Combine predictions using ensemble weights
                batting_score = (
                    xgb_batting * self.ensemble_weights['batting']['xgb'] +
                    rf_batting * self.ensemble_weights['batting']['rf'] +
                    mlp_batting * self.ensemble_weights['batting']['mlp']
                )
            
            if role in ['Bowler', 'All-rounder']:
                # Get predictions from all models
                xgb_bowling = self.bowling_xgb.predict(X)[0]
                rf_bowling = self.bowling_rf.predict(X)[0]
                mlp_bowling = self.bowling_mlp.predict(X)[0]
                
                # Combine predictions using ensemble weights
                bowling_score = (
                    xgb_bowling * self.ensemble_weights['bowling']['xgb'] +
                    rf_bowling * self.ensemble_weights['bowling']['rf'] +
                    mlp_bowling * self.ensemble_weights['bowling']['mlp']
                )
            
            # Get rank predictions from all models
            xgb_rank = self.rank_xgb.predict(X)[0]
            rf_rank = self.rank_rf.predict(X)[0]
            mlp_rank = self.rank_mlp.predict(X)[0]
            
            # Combine rank predictions using ensemble weights
            predicted_rank = (
                xgb_rank * self.ensemble_weights['rank']['xgb'] +
                rf_rank * self.ensemble_weights['rank']['rf'] +
                mlp_rank * self.ensemble_weights['rank']['mlp']
            )
            
            predictions.append({
                'player': player,
                'team': batting_team,
                'role': role,
                'batting_score': batting_score,
                'bowling_score': bowling_score,
                'total_score': batting_score + bowling_score,
                'predicted_rank': predicted_rank
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
                             columns=categorical_features, drop_first=True)
            
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
            
            role = player_mapping['role'].iloc[0]
            if role in ['Batter', 'Wicketkeeper/Batter', 'All-rounder']:
                # Get predictions from all models
                xgb_batting = self.batting_xgb.predict(X)[0]
                rf_batting = self.batting_rf.predict(X)[0]
                mlp_batting = self.batting_mlp.predict(X)[0]
                
                # Combine predictions using ensemble weights
                batting_score = (
                    xgb_batting * self.ensemble_weights['batting']['xgb'] +
                    rf_batting * self.ensemble_weights['batting']['rf'] +
                    mlp_batting * self.ensemble_weights['batting']['mlp']
                )
            
            if role in ['Bowler', 'All-rounder']:
                # Get predictions from all models
                xgb_bowling = self.bowling_xgb.predict(X)[0]
                rf_bowling = self.bowling_rf.predict(X)[0]
                mlp_bowling = self.bowling_mlp.predict(X)[0]
                
                # Combine predictions using ensemble weights
                bowling_score = (
                    xgb_bowling * self.ensemble_weights['bowling']['xgb'] +
                    rf_bowling * self.ensemble_weights['bowling']['rf'] +
                    mlp_bowling * self.ensemble_weights['bowling']['mlp']
                )
            
            # Get rank predictions from all models
            xgb_rank = self.rank_xgb.predict(X)[0]
            rf_rank = self.rank_rf.predict(X)[0]
            mlp_rank = self.rank_mlp.predict(X)[0]
            
            # Combine rank predictions using ensemble weights
            predicted_rank = (
                xgb_rank * self.ensemble_weights['rank']['xgb'] +
                rf_rank * self.ensemble_weights['rank']['rf'] +
                mlp_rank * self.ensemble_weights['rank']['mlp']
            )
            
            predictions.append({
                'player': player,
                'team': bowling_team,
                'role': role,
                'batting_score': batting_score,
                'bowling_score': bowling_score,
                'total_score': batting_score + bowling_score,
                'predicted_rank': predicted_rank
            })
        
        # Convert predictions to DataFrame and sort by total score
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('total_score', ascending=False)
        
        # Add actual rank based on total score
        predictions_df['actual_rank'] = predictions_df['total_score'].rank(ascending=False, method='min').astype(int)
        
        return predictions_df

def main():
    # Example usage
    predictor = MatchPredictor()
    
    # Example teams and players (replace with actual teams and players)
    team1 = "GT"
    team2 = "PBSK"
    batting_first = "PBSK"  # DC is batting first
    venue = "Narendra Modi Stadium, Ahmedabad"
    
    playing_11_team1 = [
        "Shubman Gill",
        "Jos Buttler",
        "Sai Sudharsan",
        "Shahrukh Khan",
        "Rahul Tewatia",
        "R Sai Kishore",
        "Arshad Khan",
        "Rashid Khan",
        "Kagiso Rabada",
        "Mohammed Siraj",
        "Prasidh Krishna"
    ]
    
    playing_11_team2 = [
        "Prabhsimran Singh",
        "Priyansh Arya",
        "Shreyas Iyer",
        "Shashank Singh",
        "Marcus Stoinis",
        "Glenn Maxwell",
        "Suryansh Shedge",
        "Azmatullah Omarzai",
        "Marco Jansen",
        "Arshdeep Singh",
        "Yuzvendra Chahal"
    ]
    
    # Get predictions
    predictions = predictor.predict_match(team1, team2, batting_first, playing_11_team1, playing_11_team2, venue)
    
    # Sort by predicted rank
    predictions = predictions.sort_values('predicted_rank')
    
    # Print predictions with captain scores
    print("\nPredicted Player Scores and Ranks (Sorted by Predicted Rank):")
    print(predictions[['player', 'team', 'role', 'batting_score', 'bowling_score', 'total_score', 'predicted_rank']].to_string(index=False))
    
    # Print team totals
    print("\nTeam Totals:")
    print(predictions.groupby('team')['total_score'].sum())
    
    # Print top 3 captain recommendations
    print("\nTop 3 Captain Recommendations:")
    captain_recommendations = predictions.nlargest(3, 'total_score')
    for idx, row in captain_recommendations.iterrows():
        print(f"{row['player']} ({row['team']}) - Total Score: {row['total_score']:.2f}")

if __name__ == "__main__":
    main() 