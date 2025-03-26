import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def clean_player_name(name):
    """Clean player name by removing country and special characters."""
    # Remove country in parentheses
    if '(' in name:
        name = name.split('(')[0].strip()
    # Remove special characters and extra spaces
    name = name.replace('  ', ' ').strip()
    return name

def create_player_mapping():
    # Read roster data
    roster_df = pd.read_csv('roster.csv', skiprows=1)  # Skip the first row with empty columns
    
    # Clean roster player names
    roster_df['clean_name'] = roster_df['Name'].apply(clean_player_name)
    
    # Read features data to get unique players
    features_df = pd.read_csv('features_df.csv')
    unique_players = features_df['player'].unique()
    
    # Create mapping dictionary
    player_mapping = {}
    
    # For each player in features, find best match in roster
    for player in unique_players:
        # Clean the player name from features
        clean_player = clean_player_name(player)
        
        # Find best match in roster
        best_match = process.extractOne(clean_player, roster_df['clean_name'])
        
        if best_match[1] >= 90:  # High confidence match
            roster_player = roster_df[roster_df['clean_name'] == best_match[0]].iloc[0]
            player_mapping[player] = {
                'roster_name': roster_player['Name'],
                'team': roster_player['Team'],
                'role': roster_player['Role'],
                'geography': roster_player['Geography'],
                'confidence': best_match[1]
            }
        else:
            player_mapping[player] = {
                'roster_name': None,
                'team': None,
                'role': None,
                'geography': None,
                'confidence': best_match[1]
            }
    
    # Convert mapping to DataFrame
    mapping_df = pd.DataFrame.from_dict(player_mapping, orient='index')
    mapping_df.index.name = 'features_player'
    
    # Save mapping to CSV
    mapping_df.to_csv('player_mapping.csv')
    
    # Print summary
    total_players = len(player_mapping)
    mapped_players = sum(1 for v in player_mapping.values() if v['roster_name'] is not None)
    print(f"\nPlayer Mapping Summary:")
    print(f"Total players in features: {total_players}")
    print(f"Successfully mapped players: {mapped_players}")
    print(f"Unmapped players: {total_players - mapped_players}")
    
    # Print unmapped players
    print("\nUnmapped Players:")
    for player, info in player_mapping.items():
        if info['roster_name'] is None:
            print(f"{player} (confidence: {info['confidence']})")

if __name__ == "__main__":
    create_player_mapping() 