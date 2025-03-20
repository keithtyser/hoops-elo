"""
Analysis module for the ELO ratings system.

Contains functions for running backtests, analyzing results,
and visualizing ELO ratings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from .elo import create_elo_data, replay_and_collect_diff_margin, find_best_slope
from .config import load_config, get_gender_config

logger = logging.getLogger(__name__)

def run_mens_elo(teams, data):
    """
    Run men's ELO rating calculation with optimal parameters.
    
    Parameters
    ----------
    teams : list
        List of team IDs
    data : DataFrame
        DataFrame containing game data
        
    Returns
    -------
    tuple
        (elo_df, brier_score)
    """
    logger.info("Running men's ELO with parameters from configuration")
    
    # Load configuration
    config = load_config()
    men_config = get_gender_config('men', config)
    
    elo_df, brier = create_elo_data(
        teams=teams,
        data=data,
        initial_rating=men_config['initial_rating'],
        k=men_config['k_factor'],
        alpha=men_config['alpha'],
        carryover=men_config['carryover'],
        carryover_factor=men_config['carryover_factor'],
        home_court=men_config['home_court'],
        mov_formula=men_config['mov_formula'],
        weighting_scheme=men_config['weighting_scheme'],
        start_year=1985,
        end_year=2025
    )
    
    logger.info(f"Men's ELO calculation complete: {len(elo_df)} records, Brier score: {brier:.6f}")
    
    return elo_df, brier

def run_womens_elo(teams, data):
    """
    Run women's ELO rating calculation with optimal parameters.
    
    Parameters
    ----------
    teams : list
        List of team IDs
    data : DataFrame
        DataFrame containing game data
        
    Returns
    -------
    tuple
        (elo_df, brier_score)
    """
    logger.info("Running women's ELO with parameters from configuration")
    
    # Load configuration
    config = load_config()
    women_config = get_gender_config('women', config)
    
    elo_df, brier = create_elo_data(
        teams=teams,
        data=data,
        initial_rating=women_config['initial_rating'],
        k=women_config['k_factor'],
        alpha=women_config['alpha'],
        carryover=women_config['carryover'],
        carryover_factor=women_config['carryover_factor'],
        home_court=women_config['home_court'],
        mov_formula=women_config['mov_formula'],
        weighting_scheme=women_config['weighting_scheme'],
        start_year=1985,
        end_year=2025
    )
    
    logger.info(f"Women's ELO calculation complete: {len(elo_df)} records, Brier score: {brier:.6f}")
    
    return elo_df, brier

def show_top_teams(elo_df_men, elo_df_women, teams_m, teams_w, season=2025, n=10):
    """
    Show top teams by ELO rating.
    
    Parameters
    ----------
    elo_df_men : DataFrame
        DataFrame containing men's ELO ratings
    elo_df_women : DataFrame
        DataFrame containing women's ELO ratings
    teams_m : DataFrame
        DataFrame containing men's team data
    teams_w : DataFrame
        DataFrame containing women's team data
    season : int, optional
        Season to show rankings for
    n : int, optional
        Number of teams to show
        
    Returns
    -------
    tuple
        (top_men_teams, top_women_teams)
    """
    # Merge team names with ratings
    tmp_df_men = pd.merge(elo_df_men, teams_m, on='TeamID', how='left')
    tmp_df_men_season = tmp_df_men[tmp_df_men['Season'] == season]
    
    top_men_teams = (
        tmp_df_men_season
        .sort_values('Rating_Last', ascending=False)
        .head(n)[['TeamName', 'Rating_Last', 'Rating_Trend']]
    )
    
    tmp_df_women = pd.merge(elo_df_women, teams_w, on='TeamID', how='left')
    tmp_df_women_season = tmp_df_women[tmp_df_women['Season'] == season]
    
    top_women_teams = (
        tmp_df_women_season
        .sort_values('Rating_Last', ascending=False)
        .head(n)[['TeamName', 'Rating_Last', 'Rating_Trend']]
    )
    
    logger.info(f"Top {n} men's teams for {season}:")
    logger.info(top_men_teams)
    
    logger.info(f"Top {n} women's teams for {season}:")
    logger.info(top_women_teams)
    
    return top_men_teams, top_women_teams

def calculate_margin_slopes(teams_m_list, data_m, teams_w_list, data_w):
    """
    Calculate margin-of-victory slopes for men's and women's games.
    
    Parameters
    ----------
    teams_m_list : list
        List of men's team IDs
    data_m : DataFrame
        DataFrame containing men's game data
    teams_w_list : list
        List of women's team IDs
    data_w : DataFrame
        DataFrame containing women's game data
        
    Returns
    -------
    tuple
        (men_slope, women_slope)
    """
    # Load configuration
    config = load_config()
    men_config = get_gender_config('men', config)
    women_config = get_gender_config('women', config)
    
    logger.info("Calculating ELO difference vs. margin relationship for men's games")
    
    df_men_diff = replay_and_collect_diff_margin(
        teams=teams_m_list,
        data=data_m,
        start_year=1985,
        end_year=2025,
        initial_rating=men_config['initial_rating'],
        k=men_config['k_factor'],
        alpha=men_config['alpha'],
        carryover=men_config['carryover'],
        carryover_factor=men_config['carryover_factor'],
        home_court=men_config['home_court'],
        mov_formula=men_config['mov_formula'],
        weighting_scheme=men_config['weighting_scheme']
    )
    
    slope_men = find_best_slope(df_men_diff)
    
    logger.info(f"Men's margin slope: {slope_men:.4f} pts per 1 Elo => {slope_men*400:.2f} pts per 400 Elo")
    
    logger.info("Calculating ELO difference vs. margin relationship for women's games")
    
    df_women_diff = replay_and_collect_diff_margin(
        teams=teams_w_list,
        data=data_w,
        start_year=1985,
        end_year=2025,
        initial_rating=women_config['initial_rating'],
        k=women_config['k_factor'],
        alpha=women_config['alpha'],
        carryover=women_config['carryover'],
        carryover_factor=women_config['carryover_factor'],
        home_court=women_config['home_court'],
        mov_formula=women_config['mov_formula'],
        weighting_scheme=women_config['weighting_scheme']
    )
    
    slope_women = find_best_slope(df_women_diff)
    
    logger.info(f"Women's margin slope: {slope_women:.4f} pts per 1 Elo => {slope_women*400:.2f} pts per 400 Elo")
    
    return slope_men, slope_women

def plot_team_rating_history(team_id, elo_df, teams_df, start_year=None, end_year=None):
    """
    Plot ELO rating history for a specific team.
    
    Parameters
    ----------
    team_id : int
        Team ID
    elo_df : DataFrame
        DataFrame containing ELO ratings
    teams_df : DataFrame
        DataFrame containing team data
    start_year : int, optional
        First year to include in plot
    end_year : int, optional
        Last year to include in plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    team_name = teams_df.loc[teams_df['TeamID'] == team_id, 'TeamName'].iloc[0]
    
    team_data = elo_df[elo_df['TeamID'] == team_id].copy()
    
    if start_year:
        team_data = team_data[team_data['Season'] >= start_year]
    if end_year:
        team_data = team_data[team_data['Season'] <= end_year]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot rating history by season
    for season, group in team_data.groupby('Season'):
        ax.plot(group['DayNum'], group['Rating'], label=f'Season {season}')
    
    # Get mean, max, min ratings
    mean_rating = team_data['Rating'].mean()
    max_rating = team_data['Rating'].max()
    min_rating = team_data['Rating'].min()
    
    # Add reference lines
    ax.axhline(y=mean_rating, color='k', linestyle='--', alpha=0.5)
    
    # Add labels and title
    ax.set_title(f'ELO Rating History for {team_name}')
    ax.set_xlabel('Day Number')
    ax.set_ylabel('ELO Rating')
    
    # Add legend and stats
    ax.legend(loc='best')
    
    # Add text box with stats
    stats_text = (
        f"Mean Rating: {mean_rating:.1f}\n"
        f"Max Rating: {max_rating:.1f}\n"
        f"Min Rating: {min_rating:.1f}"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    return fig

def plot_elo_diff_vs_margin(diff_margin_df, slope=None):
    """
    Plot ELO difference vs. actual margin.
    
    Parameters
    ----------
    diff_margin_df : DataFrame
        DataFrame containing 'EloDiff' and 'ActualMargin' columns
    slope : float, optional
        Regression slope to plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Limit the number of points for plotting
    if len(diff_margin_df) > 5000:
        plot_df = diff_margin_df.sample(5000, random_state=42)
    else:
        plot_df = diff_margin_df
    
    # Scatter plot of individual games
    ax.scatter(plot_df['EloDiff'], plot_df['ActualMargin'], alpha=0.1, s=5)
    
    # If slope is provided, plot regression line
    if slope is not None:
        x_vals = np.array([plot_df['EloDiff'].min(), plot_df['EloDiff'].max()])
        y_vals = slope * x_vals
        ax.plot(x_vals, y_vals, 'r-', lw=2, label=f'Slope: {slope:.4f}')
    
    # Add labels and title
    ax.set_xlabel('ELO Difference')
    ax.set_ylabel('Actual Margin')
    ax.set_title('ELO Difference vs. Actual Margin')
    
    # Add legend
    if slope is not None:
        ax.legend()
    
    # Add zero reference lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    return fig 