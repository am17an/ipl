import pandas as pd
import numpy as np
from collections import defaultdict

class IPLFantasyScorer:
    def __init__(self):
        # Batting points
        self.points_per_run = 1
        self.boundary_bonus = 4  # Additional point for each boundary (4s)
        self.six_bonus = 6  # Additional points for each six (6s)
        self.half_century_bonus = 8  # Additional points for scoring 50+ runs
        self.century_bonus = 16  # Additional points for scoring 100+ runs
        self.quarter_century_bonus = 4  # Additional points for scoring 25-49 runs
        self.seventy_five_bonus = 12  # Additional points for scoring 75+ runs
        self.dismissal_for_duck = -2  # Negative points for getting out for 0 runs
        
        # Bowling points
        self.points_per_wicket = 25  # Each wicket (excluding run out)
        self.lbw_or_bowled_bonus = 8  # Additional points for LBW/bowled wickets
        self.three_wicket_bonus = 4  # Additional points for 3 wickets in match
        self.four_wicket_bonus = 8  # Additional points for 4 wickets in match
        self.five_wicket_bonus = 16  # Additional points for 5 wickets in match
        self.maiden_over_bonus = 12  # Additional points for a maiden over
        self.dot_ball_points = 1  # Points for each dot ball
        
        # Fielding points
        self.catch_points = 8  # Each catch
        self.stumping_points = 12  # Each stumping
        self.run_out_direct_hit = 12  # Direct hit run out
        self.run_out_indirect = 6  # Indirect hit run out
        self.three_catch_bonus = 4
        
        # Economy rate points (applicable to bowlers who bowl minimum 2 overs)
        self.economy_rate_ranges = [
            (0, 5), (5.01, 6), (6.01, 7), (10, 11), (11.01, 12), (12, float('inf'))
        ]
        self.economy_rate_points = [6, 4, 2, -2, -4, -6]
        
        # Strike rate points (applicable to batsmen who face minimum 10 balls)
        self.strike_rate_ranges = [
            (0, 49.99), (50.00, 59.99), (60, 70), (130, 150), (150.01, 170), (170.01, float('inf'))
        ]
        self.strike_rate_points = [-6, -4, -2, 2, 4, 6]

        self.run_bonus_ranges = [(25, 49), (50, 74), (75, 99), (100, float('inf'))]
        self.run_bonus_points = [4, 8, 12, 16]

        self.wicket_bonus = [3, 4, 5]

    def calculate_player_match_stats(self, match_data, player):
        """
        Calculate detailed statistics and fantasy scores for a player in a match.
        
        Args:
            match_data (pd.DataFrame): DataFrame containing ball-by-ball data for the match
            player (str): Name of the player
        Returns:
            dict: Dictionary containing player statistics and scores
        """
        # Filter data for this player
        player_data = match_data[
            (match_data['striker'] == player) | 
            (match_data['bowler'] == player) | 
            (match_data['fielder'] == player)
        ].copy()
        
        if len(player_data) == 0:
            return None
            
        # Determine which team the player belongs to and track innings participation
        batting_data = player_data[player_data['striker'] == player]
        bowling_data = player_data[player_data['bowler'] == player]

        opposition = None
        batting_innings = None
        bowling_innings = None

        if not batting_data.empty:
            team = batting_data['batting_team'].iloc[0]
            opposition = batting_data['bowling_team'].iloc[0]
            batting_innings = batting_data['innings'].iloc[0]
        else:
            # If they didn't bat, check if they bowled
            bowling_data = player_data[player_data['bowler'] == player]
            if not bowling_data.empty:
                team = bowling_data['bowling_team'].iloc[0]
                opposition = bowling_data['batting_team'].iloc[0]
                bowling_innings = bowling_data['innings'].iloc[0]
            else:
                # If they didn't bowl, check if they fielded
                fielding_data = player_data[player_data['fielder'] == player]
                if not fielding_data.empty:
                    team = fielding_data['batting_team'].iloc[0]
                    opposition = fielding_data['bowling_team'].iloc[0]
                else:
                    return None  # Player didn't participate in any capacity

        # If we found team from batting but not bowling innings, get bowling innings
        if batting_innings is not None and bowling_innings is None and not bowling_data.empty:
            bowling_innings = bowling_data['innings'].iloc[0]
        # If we found team from bowling but not batting innings, get batting innings
        elif bowling_innings is not None and batting_innings is None and not batting_data.empty:
            batting_innings = batting_data['innings'].iloc[0]

        if bowling_innings is None:
            if batting_innings == 1:
                bowling_innings = 2
            else:
                bowling_innings = 1


        if batting_innings is None:
            if bowling_innings == 1:
                batting_innings = 2
            else:
                batting_innings = 1

        # Calculate batting stats
        batting_data = player_data[player_data['striker'] == player]
        batting_stats = {
            'runs': batting_data['runs_of_bat'].sum(),
            'balls': len(batting_data[batting_data['wide'] == 0]),  # Exclude wides from balls faced
            'fours': len(batting_data[batting_data['runs_of_bat'] == 4]),
            'sixes': len(batting_data[batting_data['runs_of_bat'] == 6])
        }
        
        # Calculate bowling stats
        bowling_data = player_data[player_data['bowler'] == player].copy()
        bowler_balls = bowling_data.copy()
        
        # Calculate legal balls and overs
        legal_balls = len(bowler_balls[bowler_balls['wide'] == 0])
        overs_bowled = legal_balls // 6 + (legal_balls % 6) / 10.0
        
        # Calculate wickets
        wickets = len(bowler_balls[
            (bowler_balls['wicket_type'].notna()) & 
            (bowler_balls['wicket_type'] != 'runout')
        ])
        
        # Calculate runs conceded
        runs_conceded = (
            bowler_balls['runs_of_bat'].sum() + 
            bowler_balls['wide'].sum() + 
            bowler_balls['noballs'].sum()
        )
        
        # Calculate dot balls
        dot_balls = len(bowler_balls[
            (bowler_balls['runs_of_bat'] == 0) & 
            (bowler_balls['wide'] == 0) & 
            (bowler_balls['noballs'] == 0) &
            (bowler_balls['byes'] == 0) &
            (bowler_balls['legbyes'] == 0) &
            (bowler_balls['extras'] == 0)
        ])

        # Calculate LBW/bowled wickets
        lbw_or_bowled = len(bowler_balls[
            (bowler_balls['wicket_type'] == 'lbw') | 
            (bowler_balls['wicket_type'] == 'bowled')
        ])
        
        # Calculate maiden overs
        bowler_balls.loc[:, 'over_number'] = bowler_balls['over'].astype(int)
        runs_per_over = bowler_balls.groupby('over_number')[['runs_of_bat', 'extras', 'noballs', 'wide', 'byes', 'legbyes']].sum()
        runs_per_over['total_runs'] = runs_per_over.sum(axis=1)
        maiden_overs = sum(runs_per_over['total_runs'] == 0)
        
        bowling_stats = {
            'overs': overs_bowled,
            'balls': legal_balls,
            'runs': runs_conceded,
            'wickets': wickets,
            'maiden_overs': maiden_overs,
            'lbw_or_bowled': lbw_or_bowled,
            'dot_balls': dot_balls
        }
        
        # Calculate fielding stats
        fielding_data = player_data[player_data['fielder'] == player]
        fielding_stats = {
            'catches': len(fielding_data[fielding_data['wicket_type'] == 'caught']),
            'stumpings': len(fielding_data[fielding_data['wicket_type'] == 'stumped']),
            'runouts': len(fielding_data[fielding_data['wicket_type'] == 'runout'])
        }
        
        # Calculate component scores
        batting_score = 0
        if batting_stats['balls'] > 0:
            # Basic batting points
            batting_score += batting_stats['runs'] * self.points_per_run
            batting_score += batting_stats['fours'] * self.boundary_bonus
            batting_score += batting_stats['sixes'] * self.six_bonus
            
            # Duck penalty
            if batting_stats['runs'] == 0:
                batting_score += self.dismissal_for_duck
            
            # Strike rate bonus
            if batting_stats['balls'] >= 10:
                strike_rate = (batting_stats['runs'] / batting_stats['balls']) * 100
                for i, (min_sr, max_sr) in enumerate(self.strike_rate_ranges):
                    if min_sr <= strike_rate <= max_sr:
                        batting_score += self.strike_rate_points[i]
                        break

            # Run bonus
            for i, (min_runs, max_runs) in enumerate(self.run_bonus_ranges):
                if min_runs <= batting_stats['runs'] <= max_runs:
                    batting_score += self.run_bonus_points[i]
                    break
        
        bowling_score = 0
        if bowling_stats['overs'] > 0:
            # Basic bowling points
            bowling_score += bowling_stats['wickets'] * self.points_per_wicket
            bowling_score += bowling_stats['maiden_overs'] * self.maiden_over_bonus
            bowling_score += bowling_stats['dot_balls'] * self.dot_ball_points
            bowling_score += bowling_stats['lbw_or_bowled'] * self.lbw_or_bowled_bonus

            # Economy rate bonus/penalty
            economy_rate = bowling_stats['runs'] / bowling_stats['overs']
            for i, (min_economy, max_economy) in enumerate(self.economy_rate_ranges):
                if min_economy <= economy_rate <= max_economy:
                    bowling_score += self.economy_rate_points[i]
                    break

            # Wicket bonus
            if wickets >= 5:
                bowling_score += self.five_wicket_bonus
            elif wickets >= 4:
                bowling_score += self.four_wicket_bonus
            elif wickets >= 3:
                bowling_score += self.three_wicket_bonus

        fielding_score = (
            fielding_stats['catches'] * self.catch_points +
            fielding_stats['stumpings'] * self.stumping_points +
            fielding_stats['runouts'] * self.run_out_direct_hit
        )
        
        return {
            'player': player,
            'match_id': match_data['match_id'].iloc[0],
            'team': team,
            'date': match_data['date'].iloc[0],
            'venue': match_data['venue'].iloc[0],
            'opposition': opposition,
            'batting_innings': batting_innings,
            'bowling_innings': bowling_innings,
            'total_score': batting_score + bowling_score + fielding_score,
            'batting_score': batting_score,
            'bowling_score': bowling_score,
            'fielding_score': fielding_score + 4,
            'batting_stats': batting_stats,
            'bowling_stats': bowling_stats,
            'fielding_stats': fielding_stats
        }