"""
Data loader module for the ELO ratings system.

Provides functions to load men's and women's NCAA basketball data.
"""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_mens_data(data_dir="data"):
    """
    Load men's NCAA basketball data.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the data files
        
    Returns
    -------
    tuple
        (regular_season_data, tournament_data, teams_data, combined_data, teams_list)
    """
    logger.info("Loading men's basketball data")
    
    # Load data files
    regular = pd.read_csv(os.path.join(data_dir, 'MRegularSeasonCompactResults.csv'))
    tourney = pd.read_csv(os.path.join(data_dir, 'MNCAATourneyCompactResults.csv'))
    teams = pd.read_csv(os.path.join(data_dir, 'MTeams.csv'))
    
    # Add metadata columns
    regular['tourney'] = 0
    tourney['tourney'] = 1
    regular['weight'] = 1.0
    tourney['weight'] = 0.7
    
    # Combine and sort data
    data = pd.concat([regular, tourney], ignore_index=True)
    data.sort_values(['Season', 'DayNum'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Extract unique team IDs
    teams_list = teams['TeamID'].unique()
    
    logger.info(f"Loaded {len(regular)} regular season games and {len(tourney)} tournament games")
    
    return regular, tourney, teams, data, teams_list

def load_womens_data(data_dir="data"):
    """
    Load women's NCAA basketball data.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the data files
        
    Returns
    -------
    tuple
        (regular_season_data, tournament_data, teams_data, combined_data, teams_list)
    """
    logger.info("Loading women's basketball data")
    
    # Load data files
    regular = pd.read_csv(os.path.join(data_dir, 'WRegularSeasonCompactResults.csv'))
    tourney = pd.read_csv(os.path.join(data_dir, 'WNCAATourneyCompactResults.csv'))
    teams = pd.read_csv(os.path.join(data_dir, 'WTeams.csv'))
    
    # Add metadata columns
    regular['tourney'] = 0
    tourney['tourney'] = 1
    regular['weight'] = 1.0
    tourney['weight'] = 0.7
    
    # Combine and sort data
    data = pd.concat([regular, tourney], ignore_index=True)
    data.sort_values(['Season', 'DayNum'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Extract unique team IDs
    teams_list = teams['TeamID'].unique()
    
    logger.info(f"Loaded {len(regular)} regular season games and {len(tourney)} tournament games")
    
    return regular, tourney, teams, data, teams_list

def make_team_name_dict(teams_df):
    """
    Create a dictionary mapping team IDs to team names.
    
    Parameters
    ----------
    teams_df : DataFrame
        DataFrame containing TeamID and TeamName columns
        
    Returns
    -------
    dict
        Dictionary mapping team IDs to team names
    """
    return teams_df.set_index('TeamID')['TeamName'].to_dict() 