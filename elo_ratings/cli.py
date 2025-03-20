"""
Command-line interface for the ELO ratings system.

Provides functions for interactively updating and querying ELO ratings.
"""

import sys
import argparse
import logging
import pandas as pd
from .data_loader import load_mens_data, load_womens_data, make_team_name_dict
from .elo import (
    load_ratings, save_ratings, update_single_game, 
    fuzzy_team_search, predict_spread_single
)
from .utils import to_american_odds, apply_longshot, prob_and_odds
from .analysis import show_top_teams
from .config import load_config, get_gender_config

logger = logging.getLogger(__name__)

def create_parser():
    """
    Create command-line argument parser.
    
    Returns
    -------
    argparse.ArgumentParser
        Command-line argument parser
    """
    parser = argparse.ArgumentParser(description="NCAA Basketball ELO Rating System")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run ELO computation
    run_parser = subparsers.add_parser("run", help="Run full ELO calculation")
    run_parser.add_argument("--gender", choices=["men", "women", "both"], 
                           default="both", help="Gender to run (default: both)")
    run_parser.add_argument("--start-year", type=int, default=1985,
                           help="First season to process (default: 1985)")
    run_parser.add_argument("--end-year", type=int, default=2025,
                           help="Last season to process (default: 2025)")
    run_parser.add_argument("--data-dir", default="data",
                           help="Directory containing data files (default: data)")
    
    # Update single game
    update_parser = subparsers.add_parser("update", help="Update ELO for a single game")
    update_parser.add_argument("--team-a", required=True, 
                              help="Team A ID or name")
    update_parser.add_argument("--team-b", required=True,
                              help="Team B ID or name")
    update_parser.add_argument("--score-a", type=int, required=True,
                              help="Team A score")
    update_parser.add_argument("--score-b", type=int, required=True,
                              help="Team B score")
    update_parser.add_argument("--location", choices=["N", "A", "B"], default="N",
                              help="Game location: N=neutral, A=team A home, B=team B home (default: N)")
    update_parser.add_argument("--gender", choices=["men", "women"], required=True,
                              help="Gender (men or women)")
    update_parser.add_argument("--data-dir", default="data",
                              help="Directory containing data files (default: data)")
    update_parser.add_argument("--ratings-file", 
                              help="File to load/save ratings (default: elo_ratings_{gender}_2025.json)")
    
    # Predict spread
    spread_parser = subparsers.add_parser("spread", help="Predict spread between two teams")
    spread_parser.add_argument("--team-a", required=True,
                              help="Team A ID or name")
    spread_parser.add_argument("--team-b", required=True,
                              help="Team B ID or name")
    spread_parser.add_argument("--location", choices=["N", "A", "B"], default="N",
                              help="Game location: N=neutral, A=team A home, B=team B home (default: N)")
    spread_parser.add_argument("--gender", choices=["men", "women"], required=True,
                              help="Gender (men or women)")
    spread_parser.add_argument("--data-dir", default="data",
                              help="Directory containing data files (default: data)")
    spread_parser.add_argument("--ratings-file",
                              help="File to load ratings from (default: elo_ratings_{gender}_2025.json)")
    
    # Predict probability and odds
    odds_parser = subparsers.add_parser("odds", help="Predict probability and odds")
    odds_parser.add_argument("--team-a", required=True,
                            help="Team A ID or name")
    odds_parser.add_argument("--team-b", required=True,
                            help="Team B ID or name")
    odds_parser.add_argument("--longshot", action="store_true",
                            help="Apply longshot bias")
    odds_parser.add_argument("--gender", choices=["men", "women"], required=True,
                            help="Gender (men or women)")
    odds_parser.add_argument("--data-dir", default="data",
                            help="Directory containing data files (default: data)")
    odds_parser.add_argument("--ratings-file",
                            help="File to load ratings from (default: elo_ratings_{gender}_2025.json)")
    
    # Show top teams
    top_parser = subparsers.add_parser("top", help="Show top teams")
    top_parser.add_argument("--gender", choices=["men", "women", "both"], 
                           default="both", help="Gender to show (default: both)")
    top_parser.add_argument("--count", type=int, default=10,
                           help="Number of teams to show (default: 10)")
    top_parser.add_argument("--data-dir", default="data",
                           help="Directory containing data files (default: data)")
    top_parser.add_argument("--ratings-file",
                           help="File to load ratings from (default: elo_ratings_{gender}_2025.json)")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("--gender", choices=["men", "women"], required=True,
                                   help="Gender (men or women)")
    interactive_parser.add_argument("--data-dir", default="data",
                                   help="Directory containing data files (default: data)")
    
    return parser

def interactive_mode(gender, data_dir):
    """
    Run interactive mode.
    
    Parameters
    ----------
    gender : str
        'men' or 'women'
    data_dir : str
        Directory containing data files
    """
    print(f"=== Interactive Mode ({gender.upper()}) ===\n")
    
    # Load configuration
    config = load_config()
    gender_config = get_gender_config(gender, config)
    
    # Load data
    if gender == "men":
        _, _, teams_df, _, _ = load_mens_data(data_dir)
        ratings_file = f"elo_ratings_men_2025.json"
        is_men = True
    else:  # women
        _, _, teams_df, _, _ = load_womens_data(data_dir)
        ratings_file = f"elo_ratings_women_2025.json"
        is_men = False
    
    # Get parameters from configuration
    home_court = gender_config['home_court']
    slope = gender_config['slope']
    
    team_name_dict = make_team_name_dict(teams_df)
    
    # Load ratings
    rating_dict = load_ratings(ratings_file)
    if not rating_dict:
        print(f"Warning: No ratings found in {ratings_file}. Starting with default ratings.")
    
    # Store previous ratings for revert functionality
    previous_rating_dict = None
    last_update_info = None
    
    while True:
        print("\nOptions:")
        print("1. Predict spread between two teams")
        print("2. Predict win probability and odds")
        print("3. Update ratings with game result")
        print("4. Revert last update")
        print("5. View top teams")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ")
        
        if choice == '1':
            # Predict spread
            team_a = input("Enter Team A name or ID: ")
            team_b = input("Enter Team B name or ID: ")
            
            location_choice = input("Location (N=neutral, A=Team A home, B=Team B home) [N]: ").upper()
            location = location_choice if location_choice in ['N', 'A', 'B'] else 'N'
            
            # Try to find teams
            team_a_matches = fuzzy_team_search(team_a, teams_df, exact=False)
            team_b_matches = fuzzy_team_search(team_b, teams_df, exact=False)
            
            if len(team_a_matches) == 0:
                print(f"No teams found matching '{team_a}'")
                continue
            
            if len(team_b_matches) == 0:
                print(f"No teams found matching '{team_b}'")
                continue
            
            # Handle multiple matches
            if len(team_a_matches) > 1:
                print(f"Multiple teams found matching '{team_a}':")
                for i, row in enumerate(team_a_matches.iterrows()):
                    _, r = row
                    print(f"{i+1}. {r['TeamName']} (ID: {r['TeamID']})")
                
                choice_a = input("Enter number of team to use: ")
                try:
                    idx = int(choice_a) - 1
                    if idx < 0 or idx >= len(team_a_matches):
                        print("Invalid choice, using first match")
                        idx = 0
                except ValueError:
                    print("Invalid input, using first match")
                    idx = 0
                
                team_a_id = team_a_matches.iloc[idx]['TeamID']
                team_a_name = team_a_matches.iloc[idx]['TeamName']
            else:
                team_a_id = team_a_matches.iloc[0]['TeamID']
                team_a_name = team_a_matches.iloc[0]['TeamName']
            
            # Same for team B
            if len(team_b_matches) > 1:
                print(f"Multiple teams found matching '{team_b}':")
                for i, row in enumerate(team_b_matches.iterrows()):
                    _, r = row
                    print(f"{i+1}. {r['TeamName']} (ID: {r['TeamID']})")
                
                choice_b = input("Enter number of team to use: ")
                try:
                    idx = int(choice_b) - 1
                    if idx < 0 or idx >= len(team_b_matches):
                        print("Invalid choice, using first match")
                        idx = 0
                except ValueError:
                    print("Invalid input, using first match")
                    idx = 0
                
                team_b_id = team_b_matches.iloc[idx]['TeamID']
                team_b_name = team_b_matches.iloc[idx]['TeamName']
            else:
                team_b_id = team_b_matches.iloc[0]['TeamID']
                team_b_name = team_b_matches.iloc[0]['TeamName']
            
            # Predict spread
            spread = predict_spread_single(
                team_a_id, team_b_id, location, rating_dict,
                men_slope=slope if is_men else None,
                women_slope=slope if not is_men else None,
                home_court_men=home_court if is_men else None,
                home_court_women=home_court if not is_men else None
            )
            
            # Format output
            location_desc = "neutral court"
            if location == 'A':
                location_desc = f"{team_a_name}'s home court"
            elif location == 'B':
                location_desc = f"{team_b_name}'s home court"
            
            print("\n==== Spread Prediction ====")
            print(f"Team A: {team_a_name} (Rating: {rating_dict.get(team_a_id, gender_config['initial_rating']):.1f})")
            print(f"Team B: {team_b_name} (Rating: {rating_dict.get(team_b_id, gender_config['initial_rating']):.1f})")
            print(f"Location: {location_desc}")
            
            if spread > 0:
                print(f"\n{team_a_name} is favored by {abs(spread):.1f} points over {team_b_name}")
            elif spread < 0:
                print(f"\n{team_b_name} is favored by {abs(spread):.1f} points over {team_a_name}")
            else:
                print("\nThe game is a pick'em (even spread)")
            
            print(f"\nThis prediction uses a slope factor of {slope:.4f} and {home_court} ELO points for home court advantage.")
            
        elif choice == '2':
            # Predict win probability and odds
            team_a = input("Enter Team A name or ID: ")
            team_b = input("Enter Team B name or ID: ")
            
            longshot = input("Apply longshot bias adjustment? (y/n) [n]: ").lower() == 'y'
            
            # Try to find teams
            team_a_matches = fuzzy_team_search(team_a, teams_df, exact=False)
            team_b_matches = fuzzy_team_search(team_b, teams_df, exact=False)
            
            if len(team_a_matches) == 0:
                print(f"No teams found matching '{team_a}'")
                continue
            
            if len(team_b_matches) == 0:
                print(f"No teams found matching '{team_b}'")
                continue
            
            # Handle multiple matches
            if len(team_a_matches) > 1:
                print(f"Multiple teams found matching '{team_a}':")
                for i, row in enumerate(team_a_matches.iterrows()):
                    _, r = row
                    print(f"{i+1}. {r['TeamName']} (ID: {r['TeamID']})")
                
                choice_a = input("Enter number of team to use: ")
                try:
                    idx = int(choice_a) - 1
                    if idx < 0 or idx >= len(team_a_matches):
                        print("Invalid choice, using first match")
                        idx = 0
                except ValueError:
                    print("Invalid input, using first match")
                    idx = 0
                
                team_a_id = team_a_matches.iloc[idx]['TeamID']
                team_a_name = team_a_matches.iloc[idx]['TeamName']
            else:
                team_a_id = team_a_matches.iloc[0]['TeamID']
                team_a_name = team_a_matches.iloc[0]['TeamName']
            
            # Same for team B
            if len(team_b_matches) > 1:
                print(f"Multiple teams found matching '{team_b}':")
                for i, row in enumerate(team_b_matches.iterrows()):
                    _, r = row
                    print(f"{i+1}. {r['TeamName']} (ID: {r['TeamID']})")
                
                choice_b = input("Enter number of team to use: ")
                try:
                    idx = int(choice_b) - 1
                    if idx < 0 or idx >= len(team_b_matches):
                        print("Invalid choice, using first match")
                        idx = 0
                except ValueError:
                    print("Invalid input, using first match")
                    idx = 0
                
                team_b_id = team_b_matches.iloc[idx]['TeamID']
                team_b_name = team_b_matches.iloc[idx]['TeamName']
            else:
                team_b_id = team_b_matches.iloc[0]['TeamID']
                team_b_name = team_b_matches.iloc[0]['TeamName']
            
            # Get ratings
            r_a = rating_dict.get(team_a_id, gender_config['initial_rating'])
            r_b = rating_dict.get(team_b_id, gender_config['initial_rating'])
            
            # Calculate probabilities
            # Formula: 10^(r_a/400) / (10^(r_a/400) + 10^(r_b/400))
            expected_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
            expected_b = 1.0 - expected_a
            
            # Apply longshot bias if requested
            if longshot:
                adj_prob_a, adj_prob_b = apply_longshot(expected_a, expected_b)
            else:
                adj_prob_a, adj_prob_b = None, None
            
            # Calculate odds
            odds_a = to_american_odds(adj_prob_a if longshot else expected_a)
            odds_b = to_american_odds(adj_prob_b if longshot else expected_b)
            
            print("\n==== Win Probability & Odds ====")
            print(f"Team A: {team_a_name} (Rating: {r_a:.1f})")
            print(f"Team B: {team_b_name} (Rating: {r_b:.1f})")
            
            print("\nRaw Probabilities:")
            print(f"{team_a_name}: {expected_a*100:.1f}%")
            print(f"{team_b_name}: {expected_b*100:.1f}%")
            
            if longshot:
                print("\nAdjusted Probabilities (with longshot bias):")
                print(f"{team_a_name}: {adj_prob_a*100:.1f}%")
                print(f"{team_b_name}: {adj_prob_b*100:.1f}%")
            
            print("\nAmerican Odds:")
            print(f"{team_a_name}: {odds_a}")
            print(f"{team_b_name}: {odds_b}")
            
        elif choice == '3':
            # Update ratings
            winner = input("Enter winning team name or ID: ")
            loser = input("Enter losing team name or ID: ")
            
            winner_score_input = input("Enter winner's score: ")
            loser_score_input = input("Enter loser's score: ")
            
            try:
                winner_score = int(winner_score_input)
                loser_score = int(loser_score_input)
                if winner_score <= loser_score:
                    print("Error: Winner's score must be higher than loser's score")
                    continue
            except ValueError:
                print("Error: Scores must be numbers")
                continue
            
            location_choice = input("Location (N=neutral, A=winner's home, B=loser's home) [N]: ").upper()
            location = location_choice if location_choice in ['N', 'A', 'B'] else 'N'
            
            # Try to find teams
            winner_matches = fuzzy_team_search(winner, teams_df, exact=False)
            loser_matches = fuzzy_team_search(loser, teams_df, exact=False)
            
            if len(winner_matches) == 0:
                print(f"No teams found matching '{winner}'")
                continue
            
            if len(loser_matches) == 0:
                print(f"No teams found matching '{loser}'")
                continue
            
            # Handle multiple matches
            if len(winner_matches) > 1:
                print(f"Multiple teams found matching '{winner}':")
                for i, row in enumerate(winner_matches.iterrows()):
                    _, r = row
                    print(f"{i+1}. {r['TeamName']} (ID: {r['TeamID']})")
                
                choice_w = input("Enter number of team to use: ")
                try:
                    idx = int(choice_w) - 1
                    if idx < 0 or idx >= len(winner_matches):
                        print("Invalid choice, using first match")
                        idx = 0
                except ValueError:
                    print("Invalid input, using first match")
                    idx = 0
                
                winner_id = winner_matches.iloc[idx]['TeamID']
                winner_name = winner_matches.iloc[idx]['TeamName']
            else:
                winner_id = winner_matches.iloc[0]['TeamID']
                winner_name = winner_matches.iloc[0]['TeamName']
            
            # Same for loser team
            if len(loser_matches) > 1:
                print(f"Multiple teams found matching '{loser}':")
                for i, row in enumerate(loser_matches.iterrows()):
                    _, r = row
                    print(f"{i+1}. {r['TeamName']} (ID: {r['TeamID']})")
                
                choice_l = input("Enter number of team to use: ")
                try:
                    idx = int(choice_l) - 1
                    if idx < 0 or idx >= len(loser_matches):
                        print("Invalid choice, using first match")
                        idx = 0
                except ValueError:
                    print("Invalid input, using first match")
                    idx = 0
                
                loser_id = loser_matches.iloc[idx]['TeamID']
                loser_name = loser_matches.iloc[idx]['TeamName']
            else:
                loser_id = loser_matches.iloc[0]['TeamID']
                loser_name = loser_matches.iloc[0]['TeamName']
            
            # Get old ratings
            old_rating_winner = rating_dict.get(winner_id, gender_config['initial_rating'])
            old_rating_loser = rating_dict.get(loser_id, gender_config['initial_rating'])
            
            # Format location for display
            if location == 'N':
                location_desc = "neutral court"
            elif location == 'A':
                location_desc = f"{winner_name}'s home court"
            else:  # location == 'B'
                location_desc = f"{loser_name}'s home court"
            
            # Show confirmation
            print("\n==== Update Confirmation ====")
            print(f"Winner: {winner_name} ({old_rating_winner:.1f}) - {winner_score} points")
            print(f"Loser: {loser_name} ({old_rating_loser:.1f}) - {loser_score} points")
            print(f"Location: {location_desc}")
            
            confirm = input("\nConfirm update? (y/n): ").lower()
            if confirm != 'y':
                print("Update cancelled")
                continue
            
            # Store previous state for revert
            previous_rating_dict = rating_dict.copy()
            
            last_update_info = {
                'winner': {'id': winner_id, 'name': winner_name},
                'loser': {'id': loser_id, 'name': loser_name},
                'score': f"{winner_score}-{loser_score}",
                'location': location_desc
            }
            
            # Set parameters based on gender config
            k_factor = gender_config['k_factor']
            alpha_value = gender_config['alpha']
            
            # Update the ratings
            new_ratings = update_single_game(
                winner_id, loser_id, winner_score, loser_score, location, 
                rating_dict, 
                k=k_factor,
                alpha=alpha_value,
                home_court=home_court,
                mov_formula=gender_config['mov_formula']
            )
            
            # Update the rating dict with new values
            rating_dict.update(new_ratings)
            
            # Save to file
            save_ratings(rating_dict, ratings_file)
            
            # Show results
            new_rating_winner = rating_dict[winner_id]
            new_rating_loser = rating_dict[loser_id]
            
            print("\n==== Update Complete ====")
            print(f"Winner ({winner_name}):")
            print(f"  Old Rating: {old_rating_winner:.1f}")
            print(f"  New Rating: {new_rating_winner:.1f}")
            print(f"  Change: +{new_rating_winner - old_rating_winner:.1f}")
            
            print(f"\nLoser ({loser_name}):")
            print(f"  Old Rating: {old_rating_loser:.1f}")
            print(f"  New Rating: {new_rating_loser:.1f}")
            print(f"  Change: {new_rating_loser - old_rating_loser:.1f}")
            
            print(f"\nRatings saved to {ratings_file}")
            
        elif choice == '4':
            # Revert last update
            if previous_rating_dict is None or last_update_info is None:
                print("Nothing to revert.")
                continue
            
            print("\n==== Revert Last Update ====")
            print(f"Winner: {last_update_info['winner']['name']}")
            print(f"Loser: {last_update_info['loser']['name']}")
            print(f"Score: {last_update_info['score']}")
            print(f"Location: {last_update_info['location']}")
            
            confirm = input("\nConfirm revert? (y/n): ").lower()
            if confirm != 'y':
                print("Revert cancelled")
                continue
            
            # Restore previous ratings
            rating_dict = previous_rating_dict.copy()
            
            # Save to file
            save_ratings(rating_dict, ratings_file)
            
            print(f"\nReverted to previous ratings and saved to {ratings_file}")
            
            # Clear revert data
            previous_rating_dict = None
            last_update_info = None
            
        elif choice == '5':
            # View top teams
            try:
                count = int(input("Number of teams to show [10]: ") or "10")
            except ValueError:
                print("Invalid input, using default of 10")
                count = 10
            
            # Sort teams by rating
            sorted_ratings = sorted(rating_dict.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n==== Top {min(count, len(sorted_ratings))} Teams ====")
            
            for i, (team_id, rating) in enumerate(sorted_ratings[:count]):
                team_name = team_name_dict.get(team_id, f"Team {team_id}")
                print(f"{i+1}. {team_name}: {rating:.1f}")
            
        elif choice == '6':
            # Exit
            print("\nExiting interactive mode.")
            break
            
        else:
            print("Invalid choice, please try again.")
            
        input("\nPress Enter to continue...")

def main():
    """Run the ELO ratings CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.command == "interactive":
            interactive_mode(args.gender, args.data_dir)
        
        # Add implementations for other commands as needed
        # (run, update, spread, odds, top)
    
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 