#!/usr/bin/env python3
import argparse
import pandas as pd
from tabulate import tabulate
from explore import create_player_dataset
from IPLScorer import IPLFantasyScorer

def calculate_match_scores(deliveries_df, match_id):
    """
    Calculate detailed fantasy scores for all players in a specific match.
    
    Args:
        deliveries_df (pd.DataFrame): DataFrame containing ball-by-ball data
        match_id (int): ID of the match to analyze
        
    Returns:
        tuple: (player_scores, match_info)
    """
    # Filter data for the specific match
    match_data = deliveries_df[deliveries_df['match_id'] == match_id].copy()
    
    if len(match_data) == 0:
        raise ValueError(f"No data found for match ID {match_id}")
    
    # Create scorer object
    scorer = IPLFantasyScorer()
    
    # Get all players who participated in this match
    all_players = set()
    for innings in match_data['innings'].unique():
        innings_data = match_data[match_data['innings'] == innings]
        all_players.update(innings_data['striker'].unique())
        all_players.update(innings_data['bowler'].unique())
        all_players.update(innings_data['fielder'].dropna().unique())
    
    # Get match info
    match_info = {
        'venue': match_data['venue'].iloc[0],
        'match_no': match_data['match_no'].iloc[0],
        'date': match_data['date'].iloc[0],
        'teams': sorted(match_data['batting_team'].unique())
    }
    
    # Process each player's performance
    player_scores = []
    for player in all_players:
        try:
            stats = scorer.calculate_player_match_stats(match_data, player)
            if stats:
                player_scores.append({
                    'Player': stats['player'],
                    'Team': stats['team'],
                    'Total Score': stats['total_score'],
                    'Batting Score': stats['batting_score'],
                    'Bowling Score': stats['bowling_score'],
                    'Fielding Score': stats['fielding_score'],
                    'Batting Stats': f"R:{stats['batting_stats']['runs']} B:{stats['batting_stats']['balls']} 4s:{stats['batting_stats']['fours']} 6s:{stats['batting_stats']['sixes']}",
                    'Bowling Stats': f"O:{stats['bowling_stats']['overs']} R:{stats['bowling_stats']['runs']} W:{stats['bowling_stats']['wickets']} M:{stats['bowling_stats']['maiden_overs']}",
                    'Fielding Stats': f"C:{stats['fielding_stats']['catches']} S:{stats['fielding_stats']['stumpings']} RO:{stats['fielding_stats']['runouts']}"
                })
        except Exception as e:
            print(f"Error processing player {player}: {str(e)}")
            continue
    
    return player_scores, match_info

def display_match_scores(match_id, data_path=None):
    """Display fantasy scores for all players in a given match."""
    try:
        # Load data
        deliveries_df = create_player_dataset(data_path) if data_path else create_player_dataset()
        
        # Calculate scores
        player_scores, match_info = calculate_match_scores(deliveries_df, match_id)
        
        if not player_scores:
            print(f"No players found for match ID {match_id}")
            return
        
        # Convert to DataFrame for easier manipulation
        scores_df = pd.DataFrame(player_scores)
        
        # Print match details
        print("\nMatch Details:")
        print("-------------")
        print(f"Match: {match_info['match_no']}")
        print(f"Venue: {match_info['venue']}")
        print(f"Date: {match_info['date']}")
        print(f"Teams: {' vs '.join(match_info['teams'])}")
        
        # Display team-wise scores
        print("\nTeam-wise Scores:")
        print("----------------")
        for team in scores_df['Team'].unique():
            print(f"\n{team}:")
            team_scores = scores_df[scores_df['Team'] == team].copy()
            
            # Sort by total score
            team_scores = team_scores.sort_values('Total Score', ascending=False)
            
            # Select and rename columns for display
            display_cols = [
                'Player', 'Total Score', 'Batting Score', 'Bowling Score', 
                'Fielding Score', 'Batting Stats', 'Bowling Stats', 'Fielding Stats'
            ]
            
            print(tabulate(
                team_scores[display_cols],
                headers='keys',
                tablefmt='grid',
                floatfmt='.1f'
            ))
            
            # Print team total
            team_total = team_scores['Total Score'].sum()
            print(f"\nTeam Total: {team_total:.1f}")
        
        # Print match summary
        print("\nMatch Summary:")
        print("-------------")
        team_totals = scores_df.groupby('Team')['Total Score'].sum().round(1)
        for team, total in team_totals.items():
            print(f"{team}: {total:.1f}")
        
        # Print top performers
        print("\nTop Performers:")
        print("--------------")
        top_batsmen = scores_df.nlargest(3, 'Batting Score')[['Player', 'Team', 'Batting Score', 'Batting Stats']]
        top_bowlers = scores_df.nlargest(3, 'Bowling Score')[['Player', 'Team', 'Bowling Score', 'Bowling Stats']]
        top_fielders = scores_df.nlargest(3, 'Fielding Score')[['Player', 'Team', 'Fielding Score', 'Fielding Stats']]
        
        print("\nTop 3 Batsmen:")
        print(tabulate(top_batsmen, headers='keys', tablefmt='grid', floatfmt='.1f'))
        
        print("\nTop 3 Bowlers:")
        print(tabulate(top_bowlers, headers='keys', tablefmt='grid', floatfmt='.1f'))
        
        print("\nTop 3 Fielders:")
        print(tabulate(top_fielders, headers='keys', tablefmt='grid', floatfmt='.1f'))
        
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Calculate and display fantasy scores for an IPL match')
    parser.add_argument('match_id', type=int, help='ID of the match to analyze')
    parser.add_argument('--data-path', type=str, help='Optional: Path to the ball-by-ball data CSV file')
    
    args = parser.parse_args()
    display_match_scores(args.match_id, args.data_path)

if __name__ == "__main__":
    main() 