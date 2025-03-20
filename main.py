"""
Main entry point for the NCAA Basketball ELO Rating System.

This script serves as the primary interface for running the ELO calculations,
managing ratings, and generating predictions.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import json

from elo_ratings.data_loader import load_mens_data, load_womens_data, make_team_name_dict
from elo_ratings.elo import (
    load_ratings, save_ratings, update_single_game, 
    fuzzy_team_search, predict_spread_single, get_final_ratings_dict
)
from elo_ratings.utils import setup_logging, to_american_odds, apply_longshot, prob_and_odds
from elo_ratings.analysis import (
    run_mens_elo, run_womens_elo, show_top_teams,
    calculate_margin_slopes, plot_team_rating_history,
    plot_elo_diff_vs_margin
)
from elo_ratings.cli import interactive_mode

# Set up default logger
logger = logging.getLogger('elo_ratings')

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NCAA Basketball ELO Rating System")
    
    parser.add_argument("command", choices=[
        "run", "update", "spread", "odds", "top", "interactive"
    ], help="Command to run")
    
    # Run ELO computation
    parser.add_argument("--gender", choices=["men", "women", "both"], 
                       default="both", help="Gender to run (default: both)")
    parser.add_argument("--start-year", type=int, default=1985,
                       help="First season to process (default: 1985)")
    parser.add_argument("--end-year", type=int, default=2025,
                       help="Last season to process (default: 2025)")
    
    # Game update arguments
    parser.add_argument("--team-a", help="Team A ID or name")
    parser.add_argument("--team-b", help="Team B ID or name")
    parser.add_argument("--score-a", type=int, help="Team A score")
    parser.add_argument("--score-b", type=int, help="Team B score")
    parser.add_argument("--location", choices=["N", "A", "B"], default="N",
                       help="Game location: N=neutral, A=team A home, B=team B home (default: N)")
    
    # Other options
    parser.add_argument("--longshot", action="store_true",
                       help="Apply longshot bias")
    parser.add_argument("--count", type=int, default=10,
                       help="Number of teams to show (default: 10)")
    parser.add_argument("--data-dir", default="data",
                       help="Directory containing data files (default: data)")
    parser.add_argument("--ratings-file", 
                       help="File to load/save ratings (default: elo_ratings_{gender}_2025.json)")
    parser.add_argument("--log-file", 
                       help="Log file to write to")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()

def run_elo_calculation(gender, start_year, end_year, data_dir):
    """
    Run full ELO calculation for specified gender(s).
    
    Parameters
    ----------
    gender : str
        'men', 'women', or 'both'
    start_year : int
        First season to process
    end_year : int
        Last season to process
    data_dir : str
        Directory containing data files
        
    Returns
    -------
    dict
        Results of the calculation
    """
    results = {}
    
    if gender in ["men", "both"]:
        logger.info(f"Loading men's data from {data_dir}")
        _, _, teams_m, data_m, teams_m_list = load_mens_data(data_dir)
        
        logger.info(f"Running men's ELO calculation ({start_year}-{end_year})")
        elo_df_men, brier_men = run_mens_elo(teams_m_list, data_m)
        
        logger.info(f"Men's ELO calculation complete: Brier score: {brier_men:.6f}")
        
        # Save final ratings
        ratings_men = get_final_ratings_dict(elo_df_men, season=end_year)
        save_ratings(ratings_men, f"elo_ratings_men_{end_year}.json")
        
        results["men"] = {
            "elo_df": elo_df_men,
            "brier": brier_men,
            "ratings": ratings_men
        }
    
    if gender in ["women", "both"]:
        logger.info(f"Loading women's data from {data_dir}")
        _, _, teams_w, data_w, teams_w_list = load_womens_data(data_dir)
        
        logger.info(f"Running women's ELO calculation ({start_year}-{end_year})")
        elo_df_women, brier_women = run_womens_elo(teams_w_list, data_w)
        
        logger.info(f"Women's ELO calculation complete: Brier score: {brier_women:.6f}")
        
        # Save final ratings
        ratings_women = get_final_ratings_dict(elo_df_women, season=end_year)
        save_ratings(ratings_women, f"elo_ratings_women_{end_year}.json")
        
        results["women"] = {
            "elo_df": elo_df_women,
            "brier": brier_women,
            "ratings": ratings_women
        }
    
    if gender == "both" and "men" in results and "women" in results:
        # Show top teams for both genders
        top_men, top_women = show_top_teams(
            results["men"]["elo_df"], 
            results["women"]["elo_df"],
            teams_m, teams_w
        )
        
        logger.info("Top men's teams:")
        logger.info(top_men)
        
        logger.info("Top women's teams:")
        logger.info(top_women)
    
    return results

def update_game(args):
    """
    Update ELO ratings for a single game.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    dict
        Updated ratings
    """
    gender = args.gender
    data_dir = args.data_dir
    
    # Determine ratings file based on gender
    ratings_file = args.ratings_file
    if not ratings_file:
        ratings_file = f"elo_ratings_{gender}_2025.json"
    
    # Load team data and ratings
    if gender == "men":
        _, _, teams_df, _, _ = load_mens_data(data_dir)
        k_factor = 140
        alpha_value = 40
        home_court = 75
    else:  # women
        _, _, teams_df, _, _ = load_womens_data(data_dir)
        k_factor = 100
        alpha_value = 20
        home_court = 50
    
    team_name_dict = make_team_name_dict(teams_df)
    
    # Load current ratings
    rating_dict = load_ratings(ratings_file)
    if not rating_dict:
        logger.warning(f"No ratings found in {ratings_file}. Starting with default ratings.")
    
    # Determine team IDs
    team_a_id = args.team_a
    team_b_id = args.team_b
    
    # Try to parse team A ID, or search by name
    try:
        team_a_id = int(team_a_id)
    except ValueError:
        matches = fuzzy_team_search(team_a_id, teams_df, exact=False)
        if len(matches) == 0:
            logger.error(f"No teams found matching '{team_a_id}'")
            return None
        elif len(matches) > 1:
            logger.error(f"Multiple teams found matching '{team_a_id}'. Please be more specific.")
            for _, row in matches.iterrows():
                logger.info(f"ID: {row['TeamID']} | Name: {row['TeamName']}")
            return None
        team_a_id = matches.iloc[0]['TeamID']
    
    # Try to parse team B ID, or search by name
    try:
        team_b_id = int(team_b_id)
    except ValueError:
        matches = fuzzy_team_search(team_b_id, teams_df, exact=False)
        if len(matches) == 0:
            logger.error(f"No teams found matching '{team_b_id}'")
            return None
        elif len(matches) > 1:
            logger.error(f"Multiple teams found matching '{team_b_id}'. Please be more specific.")
            for _, row in matches.iterrows():
                logger.info(f"ID: {row['TeamID']} | Name: {row['TeamName']}")
            return None
        team_b_id = matches.iloc[0]['TeamID']
    
    # Determine winner and loser
    if args.score_a > args.score_b:
        win_team_id = team_a_id
        lose_team_id = team_b_id
        win_score = args.score_a
        lose_score = args.score_b
        
        # Convert location code from team A perspective to winner perspective
        if args.location == "A":
            wloc = "H"  # Team A (winner) is home
        elif args.location == "B":
            wloc = "A"  # Team B (loser) is home
        else:
            wloc = "N"  # Neutral
    else:
        win_team_id = team_b_id
        lose_team_id = team_a_id
        win_score = args.score_b
        lose_score = args.score_a
        
        # Convert location code from team A perspective to winner perspective
        if args.location == "A":
            wloc = "A"  # Team A (loser) is home
        elif args.location == "B":
            wloc = "H"  # Team B (winner) is home
        else:
            wloc = "N"  # Neutral
    
    # Get team names
    win_name = team_name_dict.get(win_team_id, f"Team {win_team_id}")
    lose_name = team_name_dict.get(lose_team_id, f"Team {lose_team_id}")
    
    # Get old ratings
    old_rating_win = rating_dict.get(win_team_id, 1500.0)
    old_rating_lose = rating_dict.get(lose_team_id, 1500.0)
    
    # Update ratings
    rating_dict, brier = update_single_game(
        win_team_id=win_team_id,
        lose_team_id=lose_team_id,
        win_score=win_score,
        lose_score=lose_score,
        location=wloc,
        rating_dict=rating_dict,
        k=k_factor,
        alpha=alpha_value,
        home_court=home_court,
        mov_formula="linear"
    )
    
    # Get new ratings
    new_rating_win = rating_dict[win_team_id]
    new_rating_lose = rating_dict[lose_team_id]
    
    logger.info(f"Updated ratings:")
    logger.info(f"{win_name} (winner): {old_rating_win:.1f} → {new_rating_win:.1f} "
                f"({new_rating_win - old_rating_win:+.1f})")
    logger.info(f"{lose_name} (loser): {old_rating_lose:.1f} → {new_rating_lose:.1f} "
                f"({new_rating_lose - old_rating_lose:+.1f})")
    
    # Save updated ratings
    save_ratings(rating_dict, ratings_file)
    logger.info(f"Saved updated ratings to {ratings_file}")
    
    return rating_dict

def predict_spread(args):
    """
    Predict spread between two teams.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    tuple
        (spread, team_a_name, team_b_name)
    """
    gender = args.gender
    data_dir = args.data_dir
    
    # Determine ratings file based on gender
    ratings_file = args.ratings_file
    if not ratings_file:
        ratings_file = f"elo_ratings_{gender}_2025.json"
    
    # Load team data and ratings
    if gender == "men":
        _, _, teams_df, _, _ = load_mens_data(data_dir)
        home_court = 75
        slope = 0.033#0.0158
        is_men = True
    else:  # women
        _, _, teams_df, _, _ = load_womens_data(data_dir)
        home_court = 50
        slope = 0.0434#0.0175
        is_men = False
    
    team_name_dict = make_team_name_dict(teams_df)
    
    # Load ratings
    rating_dict = load_ratings(ratings_file)
    if not rating_dict:
        logger.warning(f"No ratings found in {ratings_file}. Using default ratings.")
    
    # Determine team IDs
    team_a_id = args.team_a
    team_b_id = args.team_b
    
    # Try to parse team A ID, or search by name
    try:
        team_a_id = int(team_a_id)
    except ValueError:
        matches = fuzzy_team_search(team_a_id, teams_df, exact=False)
        if len(matches) == 0:
            logger.error(f"No teams found matching '{team_a_id}'")
            return None
        elif len(matches) > 1:
            logger.error(f"Multiple teams found matching '{team_a_id}'. Please be more specific.")
            for _, row in matches.iterrows():
                logger.info(f"ID: {row['TeamID']} | Name: {row['TeamName']}")
            return None
        team_a_id = matches.iloc[0]['TeamID']
    
    # Try to parse team B ID, or search by name
    try:
        team_b_id = int(team_b_id)
    except ValueError:
        matches = fuzzy_team_search(team_b_id, teams_df, exact=False)
        if len(matches) == 0:
            logger.error(f"No teams found matching '{team_b_id}'")
            return None
        elif len(matches) > 1:
            logger.error(f"Multiple teams found matching '{team_b_id}'. Please be more specific.")
            for _, row in matches.iterrows():
                logger.info(f"ID: {row['TeamID']} | Name: {row['TeamName']}")
            return None
        team_b_id = matches.iloc[0]['TeamID']
    
    # Convert location code
    if args.location == "A":
        home_loc = "H"  # Team A is home
    elif args.location == "B":
        home_loc = "A"  # Team B is home
    else:
        home_loc = "N"  # Neutral
    
    # Predict spread
    spread, name_a, name_b = predict_spread_single(
        teamA_id=team_a_id,
        teamB_id=team_b_id,
        is_men=is_men,
        home_loc=home_loc,
        rating_dict_men=rating_dict if is_men else None,
        rating_dict_women=rating_dict if not is_men else None,
        men_slope=slope,
        women_slope=slope,
        home_court_men=home_court,
        home_court_women=home_court,
        team_name_dict_men=team_name_dict if is_men else None,
        team_name_dict_women=team_name_dict if not is_men else None
    )
    
    # Display result
    if spread > 0:
        logger.info(f"{name_a} is favored by {spread:.1f} points over {name_b}")
    elif spread < 0:
        logger.info(f"{name_b} is favored by {abs(spread):.1f} points over {name_a}")
    else:
        logger.info(f"Pick'em: {name_a} vs {name_b}")
    
    return spread, name_a, name_b

def predict_odds(args):
    """
    Predict probability and American odds.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    dict
        Prediction results
    """
    gender = args.gender
    data_dir = args.data_dir
    apply_ls = args.longshot
    
    # Determine ratings file based on gender
    ratings_file = args.ratings_file
    if not ratings_file:
        ratings_file = f"elo_ratings_{gender}_2025.json"
    
    # Load team data and ratings
    if gender == "men":
        _, _, teams_df, _, _ = load_mens_data(data_dir)
    else:  # women
        _, _, teams_df, _, _ = load_womens_data(data_dir)
    
    team_name_dict = make_team_name_dict(teams_df)
    
    # Load ratings
    rating_dict = load_ratings(ratings_file)
    if not rating_dict:
        logger.warning(f"No ratings found in {ratings_file}. Using default ratings.")
    
    # Determine team IDs
    team_a_id = args.team_a
    team_b_id = args.team_b
    
    # Try to parse team A ID, or search by name
    try:
        team_a_id = int(team_a_id)
    except ValueError:
        matches = fuzzy_team_search(team_a_id, teams_df, exact=False)
        if len(matches) == 0:
            logger.error(f"No teams found matching '{team_a_id}'")
            return None
        elif len(matches) > 1:
            logger.error(f"Multiple teams found matching '{team_a_id}'. Please be more specific.")
            for _, row in matches.iterrows():
                logger.info(f"ID: {row['TeamID']} | Name: {row['TeamName']}")
            return None
        team_a_id = matches.iloc[0]['TeamID']
    
    # Try to parse team B ID, or search by name
    try:
        team_b_id = int(team_b_id)
    except ValueError:
        matches = fuzzy_team_search(team_b_id, teams_df, exact=False)
        if len(matches) == 0:
            logger.error(f"No teams found matching '{team_b_id}'")
            return None
        elif len(matches) > 1:
            logger.error(f"Multiple teams found matching '{team_b_id}'. Please be more specific.")
            for _, row in matches.iterrows():
                logger.info(f"ID: {row['TeamID']} | Name: {row['TeamName']}")
            return None
        team_b_id = matches.iloc[0]['TeamID']
    
    # Get team ratings
    r_a = rating_dict.get(team_a_id, 1500.0)
    r_b = rating_dict.get(team_b_id, 1500.0)
    
    # Get team names
    name_a = team_name_dict.get(team_a_id, f"Team {team_a_id}")
    name_b = team_name_dict.get(team_b_id, f"Team {team_b_id}")
    
    # Calculate probability and odds
    result = prob_and_odds(r_a, r_b, apply_ls=apply_ls)
    
    # Display result
    logger.info(f"{name_a} vs {name_b} Prediction:")
    logger.info(f"Raw probability: {name_a}: {result['team_a']['raw_prob']:.1%}, "
                f"{name_b}: {result['team_b']['raw_prob']:.1%}")
    
    if apply_ls:
        logger.info(f"Adjusted probability (with longshot bias): "
                    f"{name_a}: {result['team_a']['adj_prob']:.1%}, "
                    f"{name_b}: {result['team_b']['adj_prob']:.1%}")
    
    odds_a = result['team_a']['odds']
    odds_b = result['team_b']['odds']
    
    if abs(odds_a) == float('inf'):
        odds_a_str = "∞" if odds_a > 0 else "-∞"
    else:
        odds_a_str = f"{int(odds_a)}" if odds_a < 0 else f"+{int(odds_a)}"
    
    if abs(odds_b) == float('inf'):
        odds_b_str = "∞" if odds_b > 0 else "-∞"
    else:
        odds_b_str = f"{int(odds_b)}" if odds_b < 0 else f"+{int(odds_b)}"
    
    logger.info(f"American odds: {name_a}: {odds_a_str}, {name_b}: {odds_b_str}")
    
    return result

def show_top_n_teams(args):
    """
    Show top N teams.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    dict
        Top teams for each gender
    """
    gender = args.gender
    data_dir = args.data_dir
    n = args.count
    
    results = {}
    
    if gender in ["men", "both"]:
        # Load men's data
        _, _, teams_m, _, _ = load_mens_data(data_dir)
        
        # Determine ratings file
        ratings_file = args.ratings_file
        if not ratings_file:
            ratings_file = "elo_ratings_men_2025.json"
        
        # Load ratings
        rating_dict_men = load_ratings(ratings_file)
        if not rating_dict_men:
            logger.warning(f"No men's ratings found in {ratings_file}.")
        
        # Create team name dictionary
        team_name_dict_men = make_team_name_dict(teams_m)
        
        # Sort by rating and get top N
        top_men = []
        for team_id, rating in sorted(rating_dict_men.items(), key=lambda x: x[1], reverse=True)[:n]:
            team_name = team_name_dict_men.get(team_id, f"Team {team_id}")
            top_men.append({"TeamID": team_id, "TeamName": team_name, "Rating": rating})
        
        logger.info(f"Top {n} Men's Teams:")
        for i, team in enumerate(top_men):
            logger.info(f"{i+1}. {team['TeamName']} (ID: {team['TeamID']}) - "
                        f"Rating: {team['Rating']:.1f}")
        
        results["men"] = top_men
    
    if gender in ["women", "both"]:
        # Load women's data
        _, _, teams_w, _, _ = load_womens_data(data_dir)
        
        # Determine ratings file
        ratings_file = args.ratings_file
        if not ratings_file:
            ratings_file = "elo_ratings_women_2025.json"
        
        # Load ratings
        rating_dict_women = load_ratings(ratings_file)
        if not rating_dict_women:
            logger.warning(f"No women's ratings found in {ratings_file}.")
        
        # Create team name dictionary
        team_name_dict_women = make_team_name_dict(teams_w)
        
        # Sort by rating and get top N
        top_women = []
        for team_id, rating in sorted(rating_dict_women.items(), key=lambda x: x[1], reverse=True)[:n]:
            team_name = team_name_dict_women.get(team_id, f"Team {team_id}")
            top_women.append({"TeamID": team_id, "TeamName": team_name, "Rating": rating})
        
        logger.info(f"Top {n} Women's Teams:")
        for i, team in enumerate(top_women):
            logger.info(f"{i+1}. {team['TeamName']} (ID: {team['TeamID']}) - "
                        f"Rating: {team['Rating']:.1f}")
        
        results["women"] = top_women
    
    return results

def main():
    """Main entry point for the ELO ratings system."""
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)
    
    try:
        if args.command == "run":
            # Run full ELO calculation
            if not args.gender:
                logger.error("Gender is required for 'run' command.")
                return 1
            
            run_elo_calculation(args.gender, args.start_year, args.end_year, args.data_dir)
        
        elif args.command == "update":
            # Update ELO for a single game
            if not args.gender or not args.team_a or not args.team_b or args.score_a is None or args.score_b is None:
                logger.error("Gender, team_a, team_b, score_a, and score_b are required for 'update' command.")
                return 1
            
            update_game(args)
        
        elif args.command == "spread":
            # Predict spread between two teams
            if not args.gender or not args.team_a or not args.team_b:
                logger.error("Gender, team_a, and team_b are required for 'spread' command.")
                return 1
            
            predict_spread(args)
        
        elif args.command == "odds":
            # Predict probability and odds
            if not args.gender or not args.team_a or not args.team_b:
                logger.error("Gender, team_a, and team_b are required for 'odds' command.")
                return 1
            
            predict_odds(args)
        
        elif args.command == "top":
            # Show top teams
            if not args.gender:
                logger.error("Gender is required for 'top' command.")
                return 1
            
            show_top_n_teams(args)
        
        elif args.command == "interactive":
            # Run interactive mode
            if not args.gender:
                logger.error("Gender is required for 'interactive' command.")
                return 1
            
            interactive_mode(args.gender, args.data_dir)
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user.")
        return 1
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 