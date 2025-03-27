import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pickle
import os
from explore import create_player_dataset, create_features_and_train_test_split
from feature_utils import get_numeric_features, get_categorical_features
import xgboost as xgb

def train_and_evaluate_models(X_train, y_train, X_test, y_test, weights, prediction_type):
    """
    Train and evaluate XGBoost, Random Forest, and MLP models with optimized parameters
    prediction_type can be 'batting' or 'bowling'
    """
    # Calculate bounds for clamping based on training data
    def calculate_bounds(y, iqr_multiplier):
        q1 = np.percentile(y, 5)
        q3 = np.percentile(y, 95)
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
    y_train_clamped = y_train #$y_train.clip(lower=lower_bound, upper=upper_bound)
    y_test_clamped = y_test #y_test.clip(lower=lower_bound, upper=upper_bound)
    
    # Create validation set for early stopping
    val_size = int(0.2 * len(X_train))
    X_train_final = X_train[:-val_size]
    y_train_final = y_train_clamped[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train_clamped[-val_size:]
    
    # Task-specific parameters
    if prediction_type == "batting":
        params = {
            'n_estimators': 2500,
            'max_depth': 4,
            'learning_rate': 0.003,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.3,
            'reg_lambda': 1.5,
            'gamma': 0.1
        }
    elif prediction_type == "bowling":
        params = {
            'n_estimators': 2000,
            'max_depth': 3,
            'learning_rate': 0.005,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 7,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'gamma': 0.2
        }
    else:  # rank prediction
        params = {
            'n_estimators': 2500,
            'max_depth': 3,
            'learning_rate': 0.01,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'gamma': 0.05
        }
    
    # Add common parameters
    params.update({
        'bootstrap': True,
        'random_state': 12412,
        'n_jobs': -1,
        'early_stopping_rounds':100
    })
    
    # Train XGBoost model with task-specific parameters
    xgb_model = xgb.XGBRegressor(**params)
    
    # Train with early stopping and no weights initially
    xgb_model.fit(
        X_train_final, 
        y_train_final,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Train Random Forest model with optimized parameters
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train_clamped, sample_weight=weights)
    
    # Train MLP model with task-specific architecture
    mlp_hidden = (50, 50)  # Larger network for batting
    mlp_alpha = 0.001  # L2 regularization
    
    mlp_model = MLPRegressor(
        hidden_layer_sizes=mlp_hidden,
        activation='relu',
        solver='adam',
        alpha=mlp_alpha,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )
    mlp_model.fit(X_train, y_train_clamped)
    
    # Get predictions and clamp them
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_train_pred = pd.Series(xgb_train_pred).clip(lower=lower_bound, upper=upper_bound)
    xgb_test_pred = xgb_model.predict(X_test)
    xgb_test_pred = pd.Series(xgb_test_pred).clip(lower=lower_bound, upper=upper_bound)
    
    rf_train_pred = rf_model.predict(X_train)
    rf_train_pred = pd.Series(rf_train_pred).clip(lower=lower_bound, upper=upper_bound)
    rf_test_pred = rf_model.predict(X_test)
    rf_test_pred = pd.Series(rf_test_pred).clip(lower=lower_bound, upper=upper_bound)
    
    mlp_train_pred = mlp_model.predict(X_train)
    mlp_train_pred = pd.Series(mlp_train_pred).clip(lower=lower_bound, upper=upper_bound)
    mlp_test_pred = mlp_model.predict(X_test)
    mlp_test_pred = pd.Series(mlp_test_pred).clip(lower=lower_bound, upper=upper_bound)
    
    # Optimize ensemble weights using validation set
    # Use fair weighting scheme instead of R²-based optimization
    best_weights = {'xgb': 0.33, 'rf': 0.33, 'mlp': 0.33}  # Equal weights for all models
    
    # Calculate ensemble predictions using fair weights
    ensemble_train_pred = (
        xgb_train_pred * best_weights['xgb'] + 
        rf_train_pred * best_weights['rf'] +
        mlp_train_pred * best_weights['mlp']
    )
    ensemble_test_pred = (
        xgb_test_pred * best_weights['xgb'] + 
        rf_test_pred * best_weights['rf'] +
        mlp_test_pred * best_weights['mlp']
    )
    
    # Calculate and print training and validation scores
    xgb_train_r2 = r2_score(y_train_clamped, xgb_train_pred)
    xgb_val_r2 = r2_score(y_test_clamped, xgb_test_pred)
    rf_train_r2 = r2_score(y_train_clamped, rf_train_pred)
    rf_val_r2 = r2_score(y_test_clamped, rf_test_pred)
    mlp_train_r2 = r2_score(y_train_clamped, mlp_train_pred)
    mlp_val_r2 = r2_score(y_test_clamped, mlp_test_pred)
    ensemble_train_r2 = r2_score(y_train_clamped, ensemble_train_pred)
    ensemble_val_r2 = r2_score(y_test_clamped, ensemble_test_pred)
    
    print(f"\n{prediction_type.capitalize()} Model Performance:")
    print(f"XGBoost - Training R²: {xgb_train_r2:.4f}, Validation R²: {xgb_val_r2:.4f}")
    print(f"Random Forest - Training R²: {rf_train_r2:.4f}, Validation R²: {rf_val_r2:.4f}")
    print(f"MLP - Training R²: {mlp_train_r2:.4f}, Validation R²: {mlp_val_r2:.4f}")
    print(f"Ensemble - Training R²: {ensemble_train_r2:.4f}, Validation R²: {ensemble_val_r2:.4f}")
    print(f"Optimal weights - XGBoost: {best_weights['xgb']:.2f}, Random Forest: {best_weights['rf']:.2f}, MLP: {best_weights['mlp']:.2f}")
    
    return xgb_model, rf_model, mlp_model, xgb_test_pred, rf_test_pred, mlp_test_pred, ensemble_test_pred

def calculate_captain_score(features_df, batting_pred, bowling_pred):
    """
    Calculate a captain score for each player based on multiple factors.
    Higher score indicates better captain choice.
    """
    captain_features = pd.DataFrame()
    
    # Base score prediction
    captain_features['predicted_total'] = batting_pred + bowling_pred
    
    # Consistency score (using recent matches)
    recent_std = features_df.groupby('player')['total_score'].rolling(
        window=5, min_periods=1  # Changed from 2 to 1
    ).std().reset_index(0, drop=True)
    captain_features['consistency_score'] = 1 / (recent_std + 1)
    
    # All-rounder bonus
    captain_features['all_rounder_score'] = np.where(
        (features_df['batting_avg'] > 15) & (features_df['bowling_avg'] < 30),
        1.2,  # 20% bonus for all-rounders
        1.0
    )
    
    # Form score (recent performance)
    captain_features['form_score'] = features_df.groupby('player')['total_score'].rolling(
        window=3, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    # Fill NaN values with appropriate defaults
    captain_features['consistency_score'] = captain_features['consistency_score'].fillna(1.0)
    captain_features['form_score'] = captain_features['form_score'].fillna(
        features_df.groupby('player')['total_score'].transform('mean')
    )
    
    # Calculate final captain score
    captain_score = (
        captain_features['predicted_total'] * 0.4 +     # Base prediction
        captain_features['consistency_score'] * 0.3 +   # Consistency
        captain_features['all_rounder_score'] * 0.1 +  # All-rounder bonus
        captain_features['form_score'] * 0.2           # Recent form
    )
    
    # Ensure no NaN values in final score
    captain_score = captain_score.fillna(
        captain_features['predicted_total']  # Fallback to just predicted total if all else fails
    )
    
    return captain_score

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
    
    # Get the last 10 matches for validation
    unique_matches = features_df['match_id'].unique()
    last_10_match_ids = unique_matches[-1:]  # Get last 10 unique match IDs
    validation_matches = features_df[features_df['match_id'].isin(last_10_match_ids)]
    train_data = features_df[~features_df['match_id'].isin(last_10_match_ids)]
    
    print(f"\nTraining on {len(train_data)} matches from {train_data['date'].min().strftime('%Y-%m-%d')} to {train_data['date'].max().strftime('%Y-%m-%d')}")
    print(f"Validating on {len(validation_matches)} matches from {validation_matches['date'].min().strftime('%Y-%m-%d')} to {validation_matches['date'].max().strftime('%Y-%m-%d')}")
    
    # Calculate match ranks for training data
    for match_id in train_data['match_id'].unique():
        match_players = train_data[train_data['match_id'] == match_id]
        train_data.loc[train_data['match_id'] == match_id, 'match_rank'] = match_players['total_score'].rank(ascending=False, method='min').astype(int)
    
    # Calculate match ranks for validation data
    for match_id in validation_matches['match_id'].unique():
        match_players = validation_matches[validation_matches['match_id'] == match_id]
        validation_matches.loc[validation_matches['match_id'] == match_id, 'match_rank'] = match_players['total_score'].rank(ascending=False, method='min').astype(int)
    
    # Prepare features for training and validation
    numeric_features = get_numeric_features()
    categorical_features = get_categorical_features()
    
    # Filter out any numeric features that don't exist
    numeric_features = [col for col in numeric_features if col in features_df.columns]
    
    # Create dummy variables for categorical features
    X_train = pd.get_dummies(train_data[numeric_features + categorical_features], 
                            columns=categorical_features, drop_first=True)
    
    # Get validation data
    X_val = pd.get_dummies(validation_matches[numeric_features + categorical_features], 
                          columns=categorical_features, drop_first=True)
    
    # Handle missing columns more efficiently
    missing_cols = set(X_train.columns) - set(X_val.columns)
    if missing_cols:
        # Create a DataFrame with missing columns filled with zeros
        missing_df = pd.DataFrame(0, index=X_val.index, columns=list(missing_cols))
        # Concatenate with existing validation data
        X_val = pd.concat([X_val, missing_df], axis=1)
        
    # Reorder columns in validation data to match training data
    X_val = X_val[X_train.columns]
    
    # Handle NaN values in numeric features
    imputer = SimpleImputer(strategy='mean')

    # Get numeric feature columns
    numeric_cols = [col for col in X_train.columns if any(feat in col for feat in numeric_features)]
    
    # Fit imputer on training data and transform both sets
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    
    # Prepare target variables
    y_train_batting = train_data['batting_score']
    y_train_bowling = train_data['bowling_score']
    y_val_batting = validation_matches['batting_score']
    y_val_bowling = validation_matches['bowling_score']
    
    # Calculate time-based weights for training data
    current_date = train_data['date'].max()
    days_diff = (current_date - train_data['date']).dt.days
    weights = np.exp(-0.005 * days_diff)
    weights = weights / weights.sum()
    
    # Train and evaluate batting models
    batting_xgb, batting_rf, batting_mlp, batting_xgb_pred, batting_rf_pred, batting_mlp_pred, batting_ensemble_pred = train_and_evaluate_models(
        X_train, y_train_batting, X_val, y_val_batting, weights, "batting"
    )
    
    # Train and evaluate bowling models
    bowling_xgb, bowling_rf, bowling_mlp, bowling_xgb_pred, bowling_rf_pred, bowling_mlp_pred, bowling_ensemble_pred = train_and_evaluate_models(
        X_train, y_train_bowling, X_val, y_val_bowling, weights, "bowling"
    )
    
    # Train and evaluate rank prediction models
    rank_xgb, rank_rf, rank_mlp, rank_xgb_pred, rank_rf_pred, rank_mlp_pred, rank_ensemble_pred = train_and_evaluate_models(
        X_train, train_data['match_rank'], X_val, validation_matches['match_rank'], weights, "rank"
    )
    
    # Calculate captain scores for training data
    train_captain_scores = calculate_captain_score(
        train_data,
        y_train_batting,
        y_train_bowling
    )
    
    # Train captain prediction model
    captain_rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_split=6,
        min_samples_leaf=4,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    captain_rf.fit(X_train, train_captain_scores, sample_weight=weights)
    
    # Print feature importance for each model
    def print_feature_importance(model, feature_names, model_name):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 20 Most Important Features for {model_name} Model:")
        print(importance_df.head(20))
    
    print_feature_importance(batting_xgb, X_train.columns, "Batting XGBoost")
    print_feature_importance(batting_rf, X_train.columns, "Batting Random Forest")
    print_feature_importance(bowling_xgb, X_train.columns, "Bowling XGBoost")
    print_feature_importance(bowling_rf, X_train.columns, "Bowling Random Forest")
    print_feature_importance(rank_xgb, X_train.columns, "Rank XGBoost")
    print_feature_importance(rank_rf, X_train.columns, "Rank Random Forest")
    
    # Create summary tables
    def create_summary_table(actuals, xgb_pred, rf_pred, mlp_pred, ensemble_pred, score_type):
        results = []
        
        # Calculate metrics for XGBoost
        xgb_mse = mean_squared_error(actuals, xgb_pred)
        xgb_rmse = np.sqrt(xgb_mse)
        xgb_mae = np.mean(np.abs(np.array(actuals) - np.array(xgb_pred)))
        xgb_r2 = r2_score(actuals, xgb_pred)
        
        # Calculate metrics for Random Forest
        rf_mse = mean_squared_error(actuals, rf_pred)
        rf_rmse = np.sqrt(rf_mse)
        rf_mae = np.mean(np.abs(np.array(actuals) - np.array(rf_pred)))
        rf_r2 = r2_score(actuals, rf_pred)
        
        # Calculate metrics for MLP
        mlp_mse = mean_squared_error(actuals, mlp_pred)
        mlp_rmse = np.sqrt(mlp_mse)
        mlp_mae = np.mean(np.abs(np.array(actuals) - np.array(mlp_pred)))
        mlp_r2 = r2_score(actuals, mlp_pred)
        
        # Calculate metrics for Ensemble
        ensemble_mse = mean_squared_error(actuals, ensemble_pred)
        ensemble_rmse = np.sqrt(ensemble_mse)
        ensemble_mae = np.mean(np.abs(np.array(actuals) - np.array(ensemble_pred)))
        ensemble_r2 = r2_score(actuals, ensemble_pred)
        
        # Modified MAPE calculation to handle zero values
        actuals_array = np.array(actuals)
        xgb_pred_array = np.array(xgb_pred)
        rf_pred_array = np.array(rf_pred)
        mlp_pred_array = np.array(mlp_pred)
        ensemble_pred_array = np.array(ensemble_pred)
        non_zero_mask = actuals_array != 0
        
        if non_zero_mask.any():
            xgb_mape = np.mean(np.abs((actuals_array[non_zero_mask] - xgb_pred_array[non_zero_mask]) / actuals_array[non_zero_mask])) * 100
            rf_mape = np.mean(np.abs((actuals_array[non_zero_mask] - rf_pred_array[non_zero_mask]) / actuals_array[non_zero_mask])) * 100
            mlp_mape = np.mean(np.abs((actuals_array[non_zero_mask] - mlp_pred_array[non_zero_mask]) / actuals_array[non_zero_mask])) * 100
            ensemble_mape = np.mean(np.abs((actuals_array[non_zero_mask] - ensemble_pred_array[non_zero_mask]) / actuals_array[non_zero_mask])) * 100
        else:
            xgb_mape = 0
            rf_mape = 0
            mlp_mape = 0
            ensemble_mape = 0
        
        results.append({
            'Model': 'XGBoost',
            'MSE': f"{xgb_mse:.2f}",
            'RMSE': f"{xgb_rmse:.2f}",
            'MAE': f"{xgb_mae:.2f}",
            'MAPE': f"{xgb_mape:.2f}%",
            'R²': f"{xgb_r2:.4f}"
        })
        
        results.append({
            'Model': 'Random Forest',
            'MSE': f"{rf_mse:.2f}",
            'RMSE': f"{rf_rmse:.2f}",
            'MAE': f"{rf_mae:.2f}",
            'MAPE': f"{rf_mape:.2f}%",
            'R²': f"{rf_r2:.4f}"
        })
        
        results.append({
            'Model': 'MLP',
            'MSE': f"{mlp_mse:.2f}",
            'RMSE': f"{mlp_rmse:.2f}",
            'MAE': f"{mlp_mae:.2f}",
            'MAPE': f"{mlp_mape:.2f}%",
            'R²': f"{mlp_r2:.4f}"
        })
        
        results.append({
            'Model': 'Ensemble',
            'MSE': f"{ensemble_mse:.2f}",
            'RMSE': f"{ensemble_rmse:.2f}",
            'MAE': f"{ensemble_mae:.2f}",
            'MAPE': f"{ensemble_mape:.2f}%",
            'R²': f"{ensemble_r2:.4f}"
        })
        
        df = pd.DataFrame(results)
        df = df.set_index('Model')
        return df
    
    # Create summary tables for both batting and bowling
    batting_summary = create_summary_table(y_val_batting, batting_xgb_pred, batting_rf_pred, batting_mlp_pred, batting_ensemble_pred, "batting")
    bowling_summary = create_summary_table(y_val_bowling, bowling_xgb_pred, bowling_rf_pred, bowling_mlp_pred, bowling_ensemble_pred, "bowling")
    rank_summary = create_summary_table(validation_matches['match_rank'], rank_xgb_pred, rank_rf_pred, rank_mlp_pred, rank_ensemble_pred, "rank")
    
    # Print summary tables
    print("\nModel Performance Summary:\n")
    print("Batting Performance:")
    print(batting_summary)
    print("\nBowling Performance:")
    print(bowling_summary)
    print("\nRank Prediction Performance:")
    print(rank_summary)
    
    # Print sample size information
    print(f"\nSample Sizes:")
    print(f"Training matches: {len(train_data)}")
    print(f"Validation matches: {len(validation_matches)}")
    print(f"Total batting predictions: {len(y_val_batting)}")
    print(f"Total bowling predictions: {len(y_val_bowling)}")
    print(f"Total rank predictions: {len(validation_matches['match_rank'])}")
    
    # Save the models
    print("\nSaving models...")
    os.makedirs('models', exist_ok=True)
    
    # Save XGBoost models
    with open('models/batting_xgb_model.pkl', 'wb') as f:
        pickle.dump(batting_xgb, f)
    with open('models/bowling_xgb_model.pkl', 'wb') as f:
        pickle.dump(bowling_xgb, f)
    with open('models/rank_xgb_model.pkl', 'wb') as f:
        pickle.dump(rank_xgb, f)
    
    # Save Random Forest models
    with open('models/batting_rf_model.pkl', 'wb') as f:
        pickle.dump(batting_rf, f)
    with open('models/bowling_rf_model.pkl', 'wb') as f:
        pickle.dump(bowling_rf, f)
    with open('models/rank_rf_model.pkl', 'wb') as f:
        pickle.dump(rank_rf, f)
    
    # Save MLP models
    with open('models/batting_mlp_model.pkl', 'wb') as f:
        pickle.dump(batting_mlp, f)
    with open('models/bowling_mlp_model.pkl', 'wb') as f:
        pickle.dump(bowling_mlp, f)
    with open('models/rank_mlp_model.pkl', 'wb') as f:
        pickle.dump(rank_mlp, f)
    
    # Save captain model
    with open('models/captain_rf_model.pkl', 'wb') as f:
        pickle.dump(captain_rf, f)
    
    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(X_train.columns.tolist(), f)
    
    # Save scaler and imputer
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    
    print("Models and preprocessing objects saved successfully!")

if __name__ == "__main__":
    main()
