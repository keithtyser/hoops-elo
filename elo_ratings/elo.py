"""
Core ELO rating system module.

Contains functions for calculating ELO ratings, performing parameter searches,
and generating predictions based on ELO ratings.
"""

import math
import json
import os
import logging
import numpy as np
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from scipy.special import expit  # sigmoid function
from .config import load_config, get_gender_config

logger = logging.getLogger(__name__)

# Helper functions for ELO calculations

def get_hca_offset(w_loc, home_court):
    """
    Calculate home-court advantage offset based on location.
    
    Parameters
    ----------
    w_loc : str
        Game location from winner's perspective ('H', 'A', or 'N')
    home_court : int
        Home court advantage in ELO points
        
    Returns
    -------
    tuple
        (winner_offset, loser_offset)
    """
    if w_loc == 'H':
        return (home_court, 0)
    elif w_loc == 'A':
        return (0, home_court)
    else:  # 'N' or anything else => neutral
        return (0, 0)

def mov_multiplier(score_diff, rating_diff, mov_formula, alpha=None):
    """
    Calculate margin-of-victory multiplier for ELO updates.
    
    Parameters
    ----------
    score_diff : int
        Score difference (winner score - loser score)
    rating_diff : float
        ELO rating difference (winner rating - loser rating)
    mov_formula : str
        Formula to use for margin-of-victory calculation ('linear', 'log_538', 'log_basic')
    alpha : float, optional
        Parameter for linear formula
        
    Returns
    -------
    float
        Multiplier to apply to ELO update
    """
    if mov_formula is None:
        return 1.0
    
    elif mov_formula == "linear":
        if alpha is None or alpha == 0:
            return 1.0
        return score_diff / alpha
    
    elif mov_formula == "log_538":
        # 538's diminishing-returns approach
        pd = abs(score_diff)
        mov_538 = math.log(pd + 1.0) * (2.2 / ((0.001 * rating_diff) + 2.2))
        return mov_538
    
    elif mov_formula == "log_basic":
        # Simpler approach: just log(pd+1)
        return math.log(score_diff + 1.0)
    
    else:
        # Default
        return 1.0

def assign_weights(df_in, weighting_scheme):
    """
    Assign weights to games based on specified weighting scheme.
    
    Parameters
    ----------
    df_in : DataFrame
        DataFrame containing game data
    weighting_scheme : str or None
        Weighting scheme to use (None, 'late_progressive', 'tourney_boost', 'both')
        
    Returns
    -------
    DataFrame
        DataFrame with 'weight' column modified or added
    """
    df = df_in.copy()
    
    if weighting_scheme is None:
        # Keep existing weights or set to 1 if none
        if 'weight' not in df.columns:
            df['weight'] = 1.0
        df['weight'] = df['weight'].fillna(1.0)
        return df
    
    # Override existing weights
    df['weight'] = 1.0
    
    if weighting_scheme == "late_progressive":
        # Scale weight by day so that later day => bigger weight
        df['weight'] = 1.0 + (df['DayNum'] / 100.0)
    
    elif weighting_scheme == "tourney_boost":
        # Tournament games get double weight
        df.loc[df['tourney'] == 1, 'weight'] = 2.0
    
    elif weighting_scheme == "both":
        # Combine day-based growth + tournament multiplier
        base_weight = 1.0 + (df['DayNum'] / 100.0)
        df['weight'] = base_weight
        df.loc[df['tourney'] == 1, 'weight'] = base_weight * 2.0
    
    return df

def predict_probability_elo(r_a, r_b, elo_scale=400):
    """
    Calculate win probability based on ELO ratings.
    
    Parameters
    ----------
    r_a : float
        ELO rating of team A
    r_b : float
        ELO rating of team B
    elo_scale : float, optional
        ELO scaling factor, defaults to 400
        
    Returns
    -------
    float
        Probability of team A winning
    """
    return 1.0 / (1.0 + 10**((r_b - r_a) / elo_scale))

# Core ELO functions

def create_elo_data(
    teams,
    data,
    gender="men",
    initial_rating=None,
    k=None,
    alpha=None,
    carryover=None,
    carryover_factor=None,
    home_court=None,
    mov_formula=None,
    weighting_scheme=None,
    lowerlim=float("-inf"),
    start_year=1985,
    end_year=2025
):
    """
    Calculate ELO ratings for teams across multiple seasons.
    
    Parameters
    ----------
    teams : list
        List of team IDs
    data : DataFrame
        DataFrame containing game data
    gender : str, optional
        Gender ('men' or 'women') to determine default parameters
    initial_rating : int, optional
        Initial ELO rating for all teams, defaults to config value
    k : int, optional
        K-factor for ELO updates, defaults to config value
    alpha : int, optional
        Parameter for margin-of-victory formula, defaults to config value
    carryover : bool, optional
        Whether to carry over ratings between seasons, defaults to config value
    carryover_factor : float, optional
        Factor to apply to carried-over ratings, defaults to config value
    home_court : int, optional
        Home-court advantage in ELO points, defaults to config value
    mov_formula : str, optional
        Formula to use for margin-of-victory calculation, defaults to config value
    weighting_scheme : str, optional
        Weighting scheme to use for game importance, defaults to config value
    lowerlim : float, optional
        Lower limit for ELO ratings
    start_year : int, optional
        First season to process
    end_year : int, optional
        Last season to process
        
    Returns
    -------
    tuple
        (ratings_df, brier_score)
        
    Notes
    -----
    The ratings_df contains columns:
    TeamID, Season, Rating_Mean, Rating_Median, Rating_Std, Rating_Min, Rating_Max, Rating_Last, Rating_Trend
    """
    # Load configuration for this gender
    config = load_config()
    gender_config = get_gender_config(gender, config)
    
    # Use parameters from configuration if not explicitly provided
    initial_rating = initial_rating if initial_rating is not None else gender_config.get('initial_rating', 1500)
    k = k if k is not None else gender_config.get('k_factor', 32)
    alpha = alpha if alpha is not None else gender_config.get('alpha', 40)
    carryover = carryover if carryover is not None else gender_config.get('carryover', True)
    carryover_factor = carryover_factor if carryover_factor is not None else gender_config.get('carryover_factor', 0.95)
    home_court = home_court if home_court is not None else gender_config.get('home_court', 75)
    mov_formula = mov_formula if mov_formula is not None else gender_config.get('mov_formula', 'linear')
    weighting_scheme = weighting_scheme if weighting_scheme is not None else gender_config.get('weighting_scheme', None)
    
    df = data.copy()
    
    # Apply weighting scheme
    df = assign_weights(df, weighting_scheme)
    df.sort_values(['Season', 'DayNum'], inplace=True)
    
    rating_dict = {}
    rating_rows = []
    brier_list = []
    
    seasons_sorted = sorted(df['Season'].unique())
    for season in seasons_sorted:
        if season < start_year or season > end_year:
            continue
        
        df_season = df[df['Season'] == season].copy()
        if len(df_season) == 0:
            continue
        
        # Handle season transition with carryover
        if not carryover or (season == seasons_sorted[0] and not rating_dict):
            # First time => reset
            rating_dict = {t: initial_rating for t in teams}
        else:
            # Partial revert
            for t in rating_dict:
                old_r = rating_dict[t]
                rating_dict[t] = initial_rating + carryover_factor * (old_r - initial_rating)
        
        df_season.sort_values('DayNum', inplace=True)
        
        # Process each game for the season
        for idx, row in df_season.iterrows():
            wteam = row['WTeamID']
            lteam = row['LTeamID']
            ws = row['WScore']
            ls = row['LScore']
            wloc = row.get('WLoc', 'N')
            wgt = row['weight']
            tour = (row['tourney'] == 1)
            
            # Ensure teams have ratings
            if wteam not in rating_dict:
                rating_dict[wteam] = initial_rating
            if lteam not in rating_dict:
                rating_dict[lteam] = initial_rating
            
            # Apply home-court advantage
            offset_w, offset_l = get_hca_offset(wloc, home_court)
            rating_diff = ((rating_dict[wteam] + offset_w) - (rating_dict[lteam] + offset_l))
            
            # Clamp rating difference for numerical stability
            MAX_EXP_DIFF = 10000
            rd_clamped = max(min(rating_diff, MAX_EXP_DIFF), -MAX_EXP_DIFF)
            
            # Calculate expected win probability
            exp_w = 1 / (1 + 10**((-rd_clamped) / 400))
            
            # Calculate margin-of-victory factor
            sd = ws - ls
            mov_mult = mov_multiplier(sd, rating_diff, mov_formula, alpha)
            
            # Update ratings
            rating_dict[wteam] += wgt * k * mov_mult * (1 - exp_w)
            rating_dict[lteam] += wgt * k * mov_mult * (0 - (1 - exp_w))
            
            # Apply lower limit if specified
            if rating_dict[wteam] < lowerlim:
                rating_dict[wteam] = lowerlim
            if rating_dict[lteam] < lowerlim:
                rating_dict[lteam] = lowerlim
            
            # Track Brier score for tournament games
            if tour:
                brier_list.append((1 - exp_w)**2)
            
            # Record ratings
            rating_rows.append({
                'Season': season,
                'DayNum': row['DayNum'],
                'TeamID': wteam,
                'Rating': rating_dict[wteam],
                'Tourney': row['tourney']
            })
            rating_rows.append({
                'Season': season,
                'DayNum': row['DayNum'],
                'TeamID': lteam,
                'Rating': rating_dict[lteam],
                'Tourney': row['tourney']
            })
    
    # Calculate average Brier score for tournament games
    if len(brier_list) > 0:
        brier_tourney = np.mean(brier_list)
    else:
        brier_tourney = np.nan
    
    logger.info(f"Brier Score: {brier_tourney:.6f} (Tournament Games Only)")
    
    # Create and process ratings DataFrame
    rating_df = pd.DataFrame(rating_rows)
    rating_df.sort_values(['TeamID', 'Season', 'DayNum'], inplace=True)
    
    # Summaries excluding tournament games
    rating_df_no_t = rating_df[rating_df['Tourney'] == 0]
    grouped = rating_df_no_t.groupby(['TeamID', 'Season'])
    
    results = grouped['Rating'].agg(['mean', 'median', 'std', 'min', 'max', 'last'])
    results.columns = [
        'Rating_Mean', 'Rating_Median', 'Rating_Std', 
        'Rating_Min', 'Rating_Max', 'Rating_Last'
    ]
    
    # Calculate rating trend
    def slope_func(x):
        idxs = np.arange(len(x))
        return linregress(idxs, x['Rating']).slope
    
    results['Rating_Trend'] = grouped.apply(slope_func)
    results.reset_index(inplace=True)
    
    return results, brier_tourney

def update_single_game(
    win_team_id: int, 
    lose_team_id: int, 
    win_score: int, 
    lose_score: int, 
    location: str = "N", 
    rating_dict: Optional[Dict[int, float]] = None,
    k: float = 32.0,
    alpha: float = 40.0,
    home_court: float = 75.0,
    mov_formula: str = "linear"
) -> Dict[int, float]:
    """
    Update ratings for a single game.
    
    Parameters
    ----------
    win_team_id : int
        Winning team ID
    lose_team_id : int
        Losing team ID
    win_score : int
        Winning team score
    lose_score : int
        Losing team score
    location : str
        Game location ('N' for neutral, 'H' for winner's home, 'A' for loser's home)
    rating_dict : Dict[int, float], optional
        Dictionary of team IDs to their ratings, by default None
    k : float
        K-factor (maximum rating change)
    alpha : float
        Parameter for margin of victory adjustment
    home_court : float
        Home court advantage in rating points
    mov_formula : str
        Formula to use for margin of victory adjustment ('linear', 'log', or 'logistic')
        
    Returns
    -------
    Dict[int, float]
        Dictionary with updated ratings for the teams involved
    """
    # Get configuration if not explicitly provided
    config = load_config()
    gender_config = None
    
    if rating_dict is None:
        rating_dict = {}
        
    # Get current ratings or use default
    if gender_config:
        default_rating = gender_config.get('initial_rating', 1500.0)
    else:
        default_rating = 1500.0  # Fallback default
        
    win_rating = rating_dict.get(win_team_id, default_rating)
    lose_rating = rating_dict.get(lose_team_id, default_rating)
    
    # Calculate margin
    margin = win_score - lose_score
    if margin <= 0:
        raise ValueError("Winner score must be greater than loser score")
    
    # Calculate rating changes
    win_change, lose_change = calculate_elo_update(
        win_rating, lose_rating, margin, k, location, home_court, mov_formula, alpha
    )
    
    # Create a new dictionary with just the updated ratings
    updated_ratings = {
        win_team_id: win_rating + win_change,
        lose_team_id: lose_rating + lose_change
    }
    
    return updated_ratings

def save_ratings(rating_dict, filename="elo_ratings.json"):
    """
    Save ELO ratings to a JSON file.
    
    Parameters
    ----------
    rating_dict : dict
        Dictionary mapping team IDs to ELO ratings
    filename : str, optional
        Output filename
    """
    # Convert keys to strings for JSON compatibility
    str_dict = {str(k): v for k, v in rating_dict.items()}
    
    with open(filename, 'w') as f:
        json.dump(str_dict, f, indent=2)
    
    logger.info(f"Saved {len(rating_dict)} team ratings to {filename}")

def save_rating_changes(rating_changes, filename="rating_changes.json"):
    """
    Save rating changes history to a JSON file.
    
    Parameters
    ----------
    rating_changes : dict
        Dictionary mapping team IDs to their rating change history
    filename : str, optional
        Output filename
    """
    # Convert keys to strings for JSON compatibility
    str_dict = {str(k): v for k, v in rating_changes.items()}
    
    with open(filename, 'w') as f:
        json.dump(str_dict, f, indent=2)
    
    logger.info(f"Saved rating changes for {len(rating_changes)} teams to {filename}")

def load_ratings(ratings_file: str) -> Dict[int, float]:
    """
    Load ratings from a file. Return empty dict if file doesn't exist or is invalid.
    
    Parameters
    ----------
    ratings_file : str
        File from which to load ratings
        
    Returns
    -------
    Dict[int, float]
        Dictionary of team IDs to their ratings
    """
    rating_dict = {}
    try:
        if os.path.exists(ratings_file):
            with open(ratings_file, 'r') as f:
                rating_dict = json.load(f)
            
            # Convert string keys to integers
            rating_dict = {int(k): float(v) for k, v in rating_dict.items()}
    except Exception as e:
        print(f"Error loading ratings from {ratings_file}: {e}")
    
    return rating_dict

def load_rating_changes(changes_file: str) -> Dict[int, List[Dict]]:
    """
    Load rating changes from a file. Return empty dict if file doesn't exist or is invalid.
    
    Parameters
    ----------
    changes_file : str
        File from which to load rating changes
        
    Returns
    -------
    Dict[int, List[Dict]]
        Dictionary of team IDs to their rating change history
    """
    rating_changes = {}
    try:
        if os.path.exists(changes_file):
            with open(changes_file, 'r') as f:
                rating_changes = json.load(f)
            
            # Convert string keys to integers
            rating_changes = {int(k): v for k, v in rating_changes.items()}
    except Exception as e:
        logger.error(f"Error loading rating changes from {changes_file}: {e}")
    
    return rating_changes

def get_final_ratings_dict(elo_df, season=2025):
    """
    Extract final ELO ratings for a specific season.
    
    Parameters
    ----------
    elo_df : DataFrame
        DataFrame containing ELO ratings history
    season : int, optional
        Season to extract ratings for
        
    Returns
    -------
    dict
        Dictionary mapping team IDs to final ratings
    """
    df_season = elo_df[elo_df['Season'] == season]
    return df_season.set_index('TeamID')['Rating_Last'].to_dict()

# Parameter search functions

def elo_param_search_extended(
    teams,
    data,
    gender="men",
    start_year=2002,
    end_year=2025
):
    """
    Perform grid search over ELO parameters.
    
    Parameters
    ----------
    teams : list
        List of team IDs
    data : DataFrame
        DataFrame containing game data
    gender : str, optional
        Gender ('men' or 'women') to determine default parameter ranges
    start_year : int, optional
        First season to process
    end_year : int, optional
        Last season to process
        
    Returns
    -------
    DataFrame
        DataFrame containing parameter combinations and Brier scores
    """
    from tqdm import tqdm
    import itertools
    
    # Load config to get defaults for this gender
    config = load_config()
    gender_config = get_gender_config(gender, config)
    
    # Define search ranges around the defaults
    # These are only search ranges, not the actual values used in production
    possible_initial_ratings = [1200, 1300, gender_config['initial_rating'], 1700]
    possible_k = [
        max(10, gender_config['k_factor'] - 50),
        max(20, gender_config['k_factor'] - 20),
        gender_config['k_factor'],
        gender_config['k_factor'] + 20,
        gender_config['k_factor'] + 50
    ]
    possible_alpha = [
        gender_config['alpha'] // 2,
        gender_config['alpha'] - 10,
        gender_config['alpha'],
        gender_config['alpha'] + 10,
        gender_config['alpha'] * 2
    ]
    carryover_options = [gender_config['carryover']]
    carryover_factors = [
        max(0.5, gender_config['carryover_factor'] - 0.2),
        gender_config['carryover_factor'],
        min(1.0, gender_config['carryover_factor'] + 0.1)
    ]
    mov_formulas = [gender_config['mov_formula'], "log", "logistic"]
    weighting_schemes = [gender_config['weighting_scheme'], "late_progressive", "tourney_boost"]
    home_court_values = [
        max(0, gender_config['home_court'] - 30),
        gender_config['home_court'] - 15,
        gender_config['home_court'],
        gender_config['home_court'] + 15,
        gender_config['home_court'] + 30
    ]
    
    # Remove duplicates and None values
    possible_initial_ratings = sorted(list(set([r for r in possible_initial_ratings if r is not None])))
    possible_k = sorted(list(set([k for k in possible_k if k is not None])))
    possible_alpha = sorted(list(set([a for a in possible_alpha if a is not None])))
    carryover_factors = sorted(list(set([f for f in carryover_factors if f is not None])))
    mov_formulas = list(set([f for f in mov_formulas if f is not None]))
    weighting_schemes = list(set([s for s in weighting_schemes if s is not None]))
    home_court_values = sorted(list(set([h for h in home_court_values if h is not None])))
    
    results_list = []
    
    # Build all parameter combinations
    combos = []
    for init_r in possible_initial_ratings:
        for k_ in possible_k:
            for alpha_ in possible_alpha:
                for carry_ in carryover_options:
                    if carry_:
                        cof_list = carryover_factors
                    else:
                        cof_list = [None]
                    for cof in cof_list:
                        for movf in mov_formulas:
                            for ws_scheme in weighting_schemes:
                                for hc in home_court_values:
                                    combos.append((init_r, k_, alpha_, carry_, cof, movf, ws_scheme, hc))
    
    logger.info(f"Running parameter search with {len(combos)} combinations")
    
    # Evaluate each parameter combination
    for (init_r, k_, alpha_, carry_, cof, movf, ws_scheme, hc) in tqdm(combos, desc="Parameter Grid Search"):
        # Run multi-year backtest with these parameters
        results, brier = create_elo_data(
            teams=teams,
            data=data,
            start_year=start_year,
            end_year=end_year,
            initial_rating=init_r,
            k=k_,
            alpha=alpha_,
            carryover=carry_,
            carryover_factor=(cof if cof else 0.0),
            mov_formula=movf,
            weighting_scheme=ws_scheme,
            home_court=hc
        )
        
        if not pd.isna(brier):
            results_list.append({
                'initial_rating': init_r,
                'k': k_,
                'alpha': alpha_,
                'carryover': carry_,
                'carryover_factor': cof,
                'mov_formula': movf,
                'weighting_scheme': ws_scheme,
                'home_court': hc,
                'brier': brier,
                'n_games': len(results)  # Approximate
            })
    
    results_df = pd.DataFrame(results_list)
    if len(results_df) > 0:
        results_df.sort_values('brier', inplace=True)
        results_df.reset_index(drop=True, inplace=True)
    
    return results_df

# Spread prediction functions

def replay_and_collect_diff_margin(
    teams,
    data,
    gender="men",
    start_year=1985,
    end_year=2025,
    initial_rating=None,
    k=None,
    alpha=None,
    carryover=None,
    carryover_factor=None,
    home_court=None,
    mov_formula=None,
    weighting_scheme=None
):
    """
    Replay seasons and collect ELO rating differences vs. actual margins.
    
    Parameters
    ----------
    teams : list
        List of team IDs
    data : DataFrame
        DataFrame containing game data
    gender : str, optional
        Gender ('men' or 'women') to determine default parameters
    start_year : int, optional
        First season to process
    end_year : int, optional
        Last season to process
    initial_rating : int, optional
        Initial ELO rating for all teams, defaults to config value
    k : int, optional
        K-factor for ELO updates, defaults to config value
    alpha : int, optional
        Parameter for margin-of-victory formula, defaults to config value
    carryover : bool, optional
        Whether to carry over ratings between seasons, defaults to config value
    carryover_factor : float, optional
        Factor to apply to carried-over ratings, defaults to config value
    home_court : int, optional
        Home-court advantage in ELO points, defaults to config value
    mov_formula : str, optional
        Formula to use for margin-of-victory calculation, defaults to config value
    weighting_scheme : str, optional
        Weighting scheme to use for game importance, defaults to config value
        
    Returns
    -------
    DataFrame
        DataFrame containing 'EloDiff' and 'ActualMargin' columns
    """
    # Load configuration for this gender
    config = load_config()
    gender_config = get_gender_config(gender, config)
    
    # Use parameters from configuration if not explicitly provided
    initial_rating = initial_rating if initial_rating is not None else gender_config.get('initial_rating', 1500)
    k = k if k is not None else gender_config.get('k_factor', 32)
    alpha = alpha if alpha is not None else gender_config.get('alpha', 40)
    carryover = carryover if carryover is not None else gender_config.get('carryover', True)
    carryover_factor = carryover_factor if carryover_factor is not None else gender_config.get('carryover_factor', 0.95)
    home_court = home_court if home_court is not None else gender_config.get('home_court', 75)
    mov_formula = mov_formula if mov_formula is not None else gender_config.get('mov_formula', 'linear')
    weighting_scheme = weighting_scheme if weighting_scheme is not None else gender_config.get('weighting_scheme', None)
    
    df = data.copy()
    df.sort_values(['Season', 'DayNum'], inplace=True)
    df = assign_weights(df, weighting_scheme)
    
    rating_dict = {}
    records = []
    
    seasons_sorted = sorted(df['Season'].unique())
    for season in seasons_sorted:
        if season < start_year or season > end_year:
            continue
        
        df_season = df[df['Season'] == season].copy()
        if len(df_season) == 0:
            continue
        
        # Handle season transition with carryover
        if (not carryover) or (season == seasons_sorted[0] and not rating_dict):
            rating_dict = {t: initial_rating for t in teams}
        else:
            for t in rating_dict:
                old_r = rating_dict[t]
                rating_dict[t] = initial_rating + carryover_factor * (old_r - initial_rating)
        
        df_season.sort_values('DayNum', inplace=True)
        
        for idx, row in df_season.iterrows():
            wteam = row['WTeamID']
            lteam = row['LTeamID']
            ws = row['WScore']
            ls = row['LScore']
            wloc = row.get('WLoc', 'N')
            wgt = row['weight']
            
            if wteam not in rating_dict:
                rating_dict[wteam] = initial_rating
            if lteam not in rating_dict:
                rating_dict[lteam] = initial_rating
            
            # Apply home-court advantage
            offW, offL = get_hca_offset(wloc, home_court)
            rating_diff = ((rating_dict[wteam] + offW) - (rating_dict[lteam] + offL))
            
            # Record rating difference and margin
            actual_margin = ws - ls
            records.append({
                'EloDiff': rating_diff,
                'ActualMargin': actual_margin
            })
            
            # Update ratings
            safe_diff = max(min(rating_diff, 10000), -10000)
            exp_w = 1.0 / (1.0 + 10**((-safe_diff) / 400))
            score_diff = ws - ls
            mov_mult = mov_multiplier(score_diff, rating_diff, mov_formula, alpha)
            
            rating_dict[wteam] += wgt * k * mov_mult * (1 - exp_w)
            rating_dict[lteam] += wgt * k * mov_mult * (0 - (1 - exp_w))
    
    return pd.DataFrame(records)

def find_best_slope(df_diff_margin):
    """
    Fit linear regression to predict margin from ELO difference.
    
    Parameters
    ----------
    df_diff_margin : DataFrame
        DataFrame containing 'EloDiff' and 'ActualMargin' columns
        
    Returns
    -------
    float
        Regression slope
    """
    x = df_diff_margin['EloDiff'].values
    y = df_diff_margin['ActualMargin'].values
    reg = linregress(x, y)
    return reg.slope

def predict_spread_single(
    teamA_id: int, 
    teamB_id: int, 
    location: str = "N", 
    rating_dict: Optional[Dict[int, float]] = None,
    men_slope: Optional[float] = None,
    women_slope: Optional[float] = None,
    home_court_men: Optional[float] = None,
    home_court_women: Optional[float] = None,
    team_name_dict_men: Optional[Dict[int, str]] = None,
    team_name_dict_women: Optional[Dict[int, str]] = None
) -> float:
    """
    Predict spread for a single game between two teams.
    
    Parameters
    ----------
    teamA_id : int
        Team A ID
    teamB_id : int
        Team B ID
    location : str
        Location code (N=neutral, A=Team A home, B=Team B home)
    rating_dict : Dict[int, float], optional
        Dictionary mapping team IDs to ratings
    men_slope : float, optional
        Slope for men's game spread calculation
    women_slope : float, optional
        Slope for women's game spread calculation
    home_court_men : float, optional
        Home court advantage in points for men's games
    home_court_women : float, optional
        Home court advantage in points for women's games
    team_name_dict_men : Dict[int, str], optional
        Dictionary mapping team IDs to names for men's teams
    team_name_dict_women : Dict[int, str], optional
        Dictionary mapping team IDs to names for women's teams
        
    Returns
    -------
    float
        Predicted spread (positive = Team A favored, negative = Team B favored)
    """
    # Load configuration if needed
    config = load_config()
    
    # Set default values using configuration
    if men_slope is None:
        men_slope = get_gender_config('men', config).get('slope', 0.033)
    
    if women_slope is None:
        women_slope = get_gender_config('women', config).get('slope', 0.0434)
    
    if home_court_men is None:
        home_court_men = get_gender_config('men', config).get('home_court', 75.0)
    
    if home_court_women is None:
        home_court_women = get_gender_config('women', config).get('home_court', 50.0)
    
    # Determine if this is men's or women's based on which dictionary is provided
    if rating_dict is None:
        raise ValueError("Rating dictionary must be provided")
    
    # Default ratings if not found
    default_rating = 1500.0
    
    # Get team ratings
    rating_a = rating_dict.get(teamA_id, default_rating)
    rating_b = rating_dict.get(teamB_id, default_rating)
    
    # Determine which parameters to use
    slope = men_slope if men_slope is not None else women_slope
    home_court = home_court_men if home_court_men is not None else home_court_women
    
    # Calculate home court adjustment
    if location == "A":  # Team A is home
        rating_a += home_court
    elif location == "B":  # Team B is home
        rating_b += home_court
    
    # Calculate spread
    spread = (rating_a - rating_b) * slope
    
    return spread

def fuzzy_team_search(query, teams_df, exact=False):
    """
    Search for teams by partial name match.
    
    Parameters
    ----------
    query : str
        Search query
    teams_df : DataFrame
        DataFrame containing team data (must have 'TeamID' and 'TeamName' columns)
    exact : bool, optional
        Whether to require exact match
        
    Returns
    -------
    DataFrame
        Matching teams with columns 'TeamID' and 'TeamName'
    """
    query_lower = query.lower()
    
    if exact:
        # Exact match (case insensitive)
        matches = teams_df[teams_df['TeamName'].str.lower() == query_lower]
    else:
        # Partial match (case insensitive)
        matches = teams_df[teams_df['TeamName'].str.lower().str.contains(query_lower)]
    
    return matches[['TeamID', 'TeamName']]

def calculate_elo_update(
    winner_rating: float, 
    loser_rating: float, 
    margin: int, 
    k: float = 32.0, 
    location: str = "N", 
    home_court: float = 75.0,
    mov_formula: str = "linear",
    alpha: float = 40.0
) -> Tuple[float, float]:
    """
    Calculate ELO updates for a single game.
    
    Parameters
    ----------
    winner_rating : float
        Winner's pre-game rating
    loser_rating : float
        Loser's pre-game rating
    margin : int
        Margin of victory (winner's score - loser's score)
    k : float
        K-factor (maximum rating change)
    location : str
        Game location ('N' for neutral, 'H' for winner's home, 'A' for loser's home)
    home_court : float
        Home court advantage in rating points
    mov_formula : str
        Formula to use for margin of victory adjustment ('linear', 'log', or 'logistic')
    alpha : float
        Parameter for margin of victory adjustment
        
    Returns
    -------
    Tuple[float, float]
        (winner_rating_change, loser_rating_change)
    """
    # Adjust for home court advantage
    if location == "H":
        # Winner is at home, give them a boost
        effective_winner_rating = winner_rating + home_court
        effective_loser_rating = loser_rating
    elif location == "A":
        # Winner is away, give loser a boost but they still lost
        effective_winner_rating = winner_rating
        effective_loser_rating = loser_rating + home_court
    else:  # neutral court
        effective_winner_rating = winner_rating
        effective_loser_rating = loser_rating
        
    # Calculate expected score (probability of winning)
    # Formula: 1 / (1 + 10^((r_b - r_a) / 400))
    expected_winner = 1.0 / (1.0 + 10.0 ** ((effective_loser_rating - effective_winner_rating) / 400.0))
    expected_loser = 1.0 - expected_winner
    
    # Basic ELO update (winner gets 1, loser gets 0)
    base_winner_change = k * (1 - expected_winner)
    base_loser_change = k * (0 - expected_loser)
    
    # Adjust for margin of victory
    if mov_formula == "linear":
        # Linear adjustment based on margin and rating difference
        adjustment = margin / alpha
    elif mov_formula == "log":
        # Logarithmic adjustment (diminishing returns for blowouts)
        adjustment = np.log1p(margin) / alpha
    elif mov_formula == "logistic":
        # Logistic adjustment
        rating_diff = effective_winner_rating - effective_loser_rating
        adjustment = margin * (2.2 / (2 + np.exp(rating_diff / alpha)))
    else:
        # Default to no adjustment
        adjustment = 1.0
        
    # Apply adjustment - higher margins lead to higher rating changes
    # but still cap the maximum at k
    winner_change = min(base_winner_change * adjustment, k)
    loser_change = max(base_loser_change * adjustment, -k)
    
    return winner_change, loser_change

def calculate_ratings(
    games_df: pd.DataFrame, 
    teams_df: pd.DataFrame, 
    team_name_dict: Dict[int, str],
    gender: str = "men",
    initial_rating: float = 1500.0, 
    k_factor: float = 32.0, 
    alpha: float = 40.0, 
    home_court: float = 75.0,
    mov_formula: str = "linear",
    carryover: bool = True,
    carryover_factor: float = 0.75,
    previous_ratings: Optional[Dict[int, float]] = None,
    verbose: bool = False
) -> Dict[int, float]:
    """
    Calculate ELO ratings for all teams based on game history.
    
    Parameters
    ----------
    games_df : pd.DataFrame
        DataFrame of games with columns: Season, DayNum, WTeamID, LTeamID, WScore, LScore, WLoc
    teams_df : pd.DataFrame
        DataFrame of teams with columns: TeamID, TeamName
    team_name_dict : Dict[int, str]
        Dictionary mapping team IDs to names
    gender : str
        'men' or 'women' to determine gender-specific parameters
    initial_rating : float
        Initial rating for all teams
    k_factor : float
        K-factor (maximum rating change per game)
    alpha : float
        Parameter for margin of victory adjustment
    home_court : float
        Home court advantage in rating points
    mov_formula : str
        Formula to use for margin of victory adjustment ('linear', 'log', or 'logistic')
    carryover : bool
        Whether to carry over ratings between seasons
    carryover_factor : float
        Factor to regress ratings toward the mean between seasons (0-1)
    previous_ratings : Dict[int, float], optional
        Previous ratings to use as starting point, by default None
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    Dict[int, float]
        Dictionary mapping team IDs to their final ELO ratings
    """
    # Load configuration
    config = load_config()
    gender_config = get_gender_config(gender, config)
    
    # Use parameters from configuration if not explicitly provided
    initial_rating = initial_rating or gender_config.get('initial_rating', 1500.0)
    k_factor = k_factor or gender_config.get('k_factor', 32.0)
    alpha = alpha or gender_config.get('alpha', 40.0)
    home_court = home_court or gender_config.get('home_court', 75.0)
    mov_formula = mov_formula or gender_config.get('mov_formula', 'linear')
    carryover = carryover if carryover is not None else gender_config.get('carryover', True)
    carryover_factor = carryover_factor or gender_config.get('carryover_factor', 0.75)
    
    # Create a dictionary to store ELO ratings
    ratings = {}
    
    # Get unique team IDs from the games dataframe
    team_ids = set()
    for _, row in games_df.iterrows():
        team_ids.add(row['WTeamID'])
        team_ids.add(row['LTeamID'])
    
    # Initialize ratings for all teams
    for team_id in team_ids:
        if previous_ratings and team_id in previous_ratings:
            # Apply carryover regression if using previous ratings
            ratings[team_id] = (previous_ratings[team_id] - initial_rating) * carryover_factor + initial_rating
        else:
            ratings[team_id] = initial_rating
    
    # Get unique seasons and sort them
    seasons = sorted(games_df['Season'].unique())
    
    # Iterate through each season
    for season in seasons:
        if verbose:
            print(f"Processing season {season}...")
        
        # Get games for this season, sorted by day number
        season_games = games_df[games_df['Season'] == season].sort_values('DayNum')
        
        # Iterate through games in chronological order
        for _, game in season_games.iterrows():
            winner_id = game['WTeamID']
            loser_id = game['LTeamID']
            
            # Skip if either team isn't in our ratings (shouldn't happen with proper initialization)
            if winner_id not in ratings or loser_id not in ratings:
                continue
            
            # Get margin of victory
            margin = game['WScore'] - game['LScore']
            
            # Get location code - N: neutral, H: winner's home, A: loser's home
            location = game['WLoc']
            
            # Calculate rating changes
            win_change, lose_change = calculate_elo_update(
                ratings[winner_id], ratings[loser_id], margin, 
                k=k_factor, location=location, home_court=home_court, 
                mov_formula=mov_formula, alpha=alpha
            )
            
            # Update ratings
            ratings[winner_id] += win_change
            ratings[loser_id] += lose_change
        
        # End of season, apply carryover if needed
        if carryover and season != seasons[-1]:
            for team_id in ratings:
                # Regress each team's rating toward the mean
                ratings[team_id] = (ratings[team_id] - initial_rating) * carryover_factor + initial_rating
    
    return ratings 