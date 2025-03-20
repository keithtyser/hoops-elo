"""
Utility functions for the ELO ratings system.

Contains functions for probability adjustments, odds conversion, 
and logging setup.
"""

import math
import logging
import sys

def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up logging configuration.
    
    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.INFO)
    log_file : str, optional
        Path to log file, if None, only console logging is enabled
    """
    # Create logger
    logger = logging.getLogger('elo_ratings')
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Goto conversion functions

def goto_conversion(odds_list, total=1, eps=1e-6):
    """
    Apply Goto conversion to a list of odds.
    
    Parameters
    ----------
    odds_list : list
        List of odds values
    total : float, optional
        Target sum for adjusted probabilities
    eps : float, optional
        Small epsilon value to avoid division by zero
        
    Returns
    -------
    list
        Adjusted probabilities that sum to 'total'
    """
    probabilities = [1.0/x for x in odds_list]
    ses = [math.sqrt((p - p**2) / p) for p in probabilities]
    step = (sum(probabilities) - total) / sum(ses)
    
    adjusted_probs = []
    for x, y in zip(probabilities, ses):
        val = x - (y * step)
        val_clamped = min(max(val, eps), 1.0)
        adjusted_probs.append(val_clamped)
    
    return adjusted_probs

def preprocessed_goto_conversion(odds_list, total=1):
    """
    Apply preprocessed Goto conversion to a list of odds.
    
    Parameters
    ----------
    odds_list : list
        List of odds values
    total : float, optional
        Target sum for adjusted probabilities
        
    Returns
    -------
    list
        Adjusted probabilities that sum to 'total'
    """
    probabilities = [1.0/x for x in odds_list]
    
    if sum(probabilities) < total:
        multiplier = 0.99 / max(probabilities)
        probabilities = [p * multiplier for p in probabilities]
        reverse_odds = [1/(1 - p) for p in probabilities]
        reverse_probs = goto_conversion(reverse_odds, total=total)
        adjusted_probs = [1.0 - rp for rp in reverse_probs]
    else:
        adjusted_probs = goto_conversion(odds_list, total=total)
    
    return adjusted_probs

def apply_longshot(p_raw):
    """
    Apply longshot bias to a raw probability.
    
    Parameters
    ----------
    p_raw : float
        Raw probability value (0-1)
        
    Returns
    -------
    float
        Adjusted probability with longshot bias applied
    """
    if p_raw <= 0.0:
        p_raw = 1e-8
    elif p_raw >= 1.0:
        p_raw = 1 - 1e-8
    
    odds1 = 1.0 / p_raw
    odds2 = 1.0 / (1.0 - p_raw)
    
    p1_adj, p2_adj = preprocessed_goto_conversion([odds1, odds2], total=1)
    
    return p1_adj

def to_american_odds(prob, longshot=False):
    """
    Convert probability to American odds.
    
    Parameters
    ----------
    prob : float
        Probability value (0-1)
    longshot : bool, optional
        Whether to apply longshot bias before conversion
        
    Returns
    -------
    float
        American odds
        - Negative odds if prob > 0.5 (e.g., -300)
        - Positive odds if prob < 0.5 (e.g., +120)
    
    Examples
    --------
    >>> to_american_odds(0.75)
    -300.0
    >>> to_american_odds(0.25)
    300.0
    >>> to_american_odds(0.5)
    -100.0
    >>> to_american_odds(0.0)
    inf
    >>> to_american_odds(1.0)
    -inf
    """
    # Edge cases
    if prob <= 0.0:
        return float('inf')
    if prob >= 1.0:
        return float('-inf')
    
    # Apply longshot bias if requested
    if longshot:
        prob = apply_longshot(prob)
    
    # Convert to American odds
    if prob > 0.5:
        # Favorite (negative odds)
        return -100.0 * prob / (1.0 - prob)
    else:
        # Underdog (positive odds)
        return 100.0 * (1.0 - prob) / prob

def prob_and_odds(team_a_rating, team_b_rating, apply_ls=False, elo_scale=400):
    """
    Calculate probability and American odds for a matchup.
    
    Parameters
    ----------
    team_a_rating : float
        ELO rating of team A
    team_b_rating : float
        ELO rating of team B
    apply_ls : bool, optional
        Whether to apply longshot bias
    elo_scale : float, optional
        ELO scaling factor
        
    Returns
    -------
    dict
        Dictionary containing raw probability, adjusted probability,
        and American odds for both teams
    """
    # Raw probability for team A
    p_raw = 1.0 / (1.0 + 10**((team_b_rating - team_a_rating) / elo_scale))
    
    # Apply longshot bias if requested
    if apply_ls:
        p_adj = apply_longshot(p_raw)
    else:
        p_adj = p_raw
    
    # American odds
    odds_a = to_american_odds(p_adj)
    odds_b = to_american_odds(1.0 - p_adj)
    
    return {
        'team_a': {
            'raw_prob': p_raw,
            'adj_prob': p_adj,
            'odds': odds_a
        },
        'team_b': {
            'raw_prob': 1.0 - p_raw,
            'adj_prob': 1.0 - p_adj,
            'odds': odds_b
        }
    } 