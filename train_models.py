import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
from explore import create_player_dataset, create_features_and_train_test_split
from feature_utils import get_numeric_features, get_categorical_features, calculate_player_features
import xgboost as xgb

def train_and_evaluate_models(X_train, y_train, X_test, y_test, weights, prediction_type):
    """
    Train and evaluate both XGBoost and Random Forest models with optimized parameters
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
    y_train_clamped = y_train.clip(lower=lower_bound, upper=upper_bound)
    y_test_clamped = y_test.clip(lower=lower_bound, upper=upper_bound)
    
    # Train XGBoost model with optimized parameters
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.001,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        boostrap= True,
        random_state=12412,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_clamped, 
                 sample_weight=weights)
    
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
    
    # Get predictions and clamp them
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_train_pred = pd.Series(xgb_train_pred).clip(lower=lower_bound, upper=upper_bound)
    xgb_test_pred = xgb_model.predict(X_test)
    xgb_test_pred = pd.Series(xgb_test_pred).clip(lower=lower_bound, upper=upper_bound)
    
    rf_train_pred = rf_model.predict(X_train)
    rf_train_pred = pd.Series(rf_train_pred).clip(lower=lower_bound, upper=upper_bound)
    rf_test_pred = rf_model.predict(X_test)
    rf_test_pred = pd.Series(rf_test_pred).clip(lower=lower_bound, upper=upper_bound)
    
    # Calculate and print training and validation scores
    xgb_train_r2 = r2_score(y_train_clamped, xgb_train_pred)
    xgb_val_r2 = r2_score(y_test_clamped, xgb_test_pred)
    rf_train_r2 = r2_score(y_train_clamped, rf_train_pred)
    rf_val_r2 = r2_score(y_test_clamped, rf_test_pred)
    
    print(f"\n{prediction_type.capitalize()} Model Performance:")
    print(f"XGBoost - Training R²: {xgb_train_r2:.4f}, Validation R²: {xgb_val_r2:.4f}")
    print(f"Random Forest - Training R²: {rf_train_r2:.4f}, Validation R²: {rf_val_r2:.4f}")
    
    return xgb_model, rf_model, xgb_test_pred, rf_test_pred, xgb_train_pred, rf_train_pred, xgb_test_pred, rf_test_pred, xgb_test_pred

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
    categorical_features = ['team', 'opposition', 'venue', 'player']
    
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
    batting_xgb, batting_rf, batting_xgb_pred, batting_rf_pred, _, _, _, _, _ = train_and_evaluate_models(
        X_train, y_train_batting, X_val, y_val_batting, weights, "batting"
    )
    
    # Train and evaluate bowling models
    bowling_xgb, bowling_rf, bowling_xgb_pred, bowling_rf_pred, _, _, _, _, _ = train_and_evaluate_models(
        X_train, y_train_bowling, X_val, y_val_bowling, weights, "bowling"
    )
    
    # Train and evaluate rank prediction models
    rank_xgb, rank_rf, rank_xgb_pred, rank_rf_pred, _, _, _, _, _ = train_and_evaluate_models(
        X_train, train_data['match_rank'], X_val, validation_matches['match_rank'], weights, "rank"
    )
    
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
    def create_summary_table(actuals, xgb_pred, rf_pred, score_type):
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
        
        # Modified MAPE calculation to handle zero values
        actuals_array = np.array(actuals)
        xgb_pred_array = np.array(xgb_pred)
        rf_pred_array = np.array(rf_pred)
        non_zero_mask = actuals_array != 0
        
        if non_zero_mask.any():
            xgb_mape = np.mean(np.abs((actuals_array[non_zero_mask] - xgb_pred_array[non_zero_mask]) / actuals_array[non_zero_mask])) * 100
            rf_mape = np.mean(np.abs((actuals_array[non_zero_mask] - rf_pred_array[non_zero_mask]) / actuals_array[non_zero_mask])) * 100
        else:
            xgb_mape = 0
            rf_mape = 0
        
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
        
        df = pd.DataFrame(results)
        df = df.set_index('Model')
        return df
    
    # Create summary tables for both batting and bowling
    batting_summary = create_summary_table(y_val_batting, batting_xgb_pred, batting_rf_pred, "batting")
    bowling_summary = create_summary_table(y_val_bowling, bowling_xgb_pred, bowling_rf_pred, "bowling")
    rank_summary = create_summary_table(validation_matches['match_rank'], rank_xgb_pred, rank_rf_pred, "rank")
    
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