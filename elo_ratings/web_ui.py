"""
Web UI for the ELO ratings system.

Provides a web interface for interactively updating and querying ELO ratings.
"""

import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pandas as pd
import numpy as np

from .data_loader import load_mens_data, load_womens_data, make_team_name_dict
from .elo import (
    load_ratings, save_ratings, update_single_game, 
    fuzzy_team_search, predict_spread_single, get_final_ratings_dict,
    load_rating_changes, save_rating_changes
)
from .utils import to_american_odds, apply_longshot, prob_and_odds
from .config import load_config, get_gender_config

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'))
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_flash_messages')

# Initialize app state
app_state = {
    'gender': None,
    'data_dir': None,
    'teams_df': None,
    'team_name_dict': None,
    'rating_dict': None,
    'previous_rating_dict': None,  # For revert functionality
    'last_update_info': None,      # For revert info display
    'rating_changes': {},          # For tracking rating changes
    'ratings_file': None,
    'changes_file': None,          # File for rating changes
    'is_men': None,
    'home_court': None,
    'slope': None,
    'config': load_config()        # Load configuration
}

def load_data(gender, data_dir='data'):
    """
    Load data for the specified gender.
    
    Parameters
    ----------
    gender : str
        'men' or 'women'
    data_dir : str
        Directory containing data files
    """
    app.logger.info(f"load_data called with gender={gender}, data_dir={data_dir}")
    app_state['gender'] = gender
    app_state['data_dir'] = data_dir
    
    # Get gender-specific configuration
    try:
        app.logger.info(f"Loading gender configuration for {gender}")
        gender_config = get_gender_config(gender, app_state['config'])
        app.logger.info(f"Gender configuration loaded: {gender_config}")
    except Exception as e:
        app.logger.error(f"Error loading gender configuration: {str(e)}", exc_info=True)
        raise
    
    try:
        if gender == "men":
            app.logger.info("Loading men's basketball data")
            _, _, teams_df, _, _ = load_mens_data(data_dir)
            app_state['ratings_file'] = f"elo_ratings_men_2025.json"
            app_state['changes_file'] = f"rating_changes_men_2025.json"
            app_state['is_men'] = True
            app.logger.info(f"Men's ratings file: {app_state['ratings_file']}")
            app.logger.info(f"Men's changes file: {app_state['changes_file']}")
        else:  # women
            app.logger.info("Loading women's basketball data")
            _, _, teams_df, _, _ = load_womens_data(data_dir)
            app_state['ratings_file'] = f"elo_ratings_women_2025.json"
            app_state['changes_file'] = f"rating_changes_women_2025.json"
            app_state['is_men'] = False
            app.logger.info(f"Women's ratings file: {app_state['ratings_file']}")
            app.logger.info(f"Women's changes file: {app_state['changes_file']}")
    
        # Set parameters from configuration
        app_state['home_court'] = gender_config['home_court']
        app_state['slope'] = gender_config['slope']
        
        app_state['teams_df'] = teams_df
        app.logger.info(f"Loaded teams_df with {len(teams_df)} teams")
        
        app_state['team_name_dict'] = make_team_name_dict(teams_df)
        app.logger.info(f"Created team_name_dict with {len(app_state['team_name_dict'])} entries")
        
        # Load ratings
        app.logger.info(f"Loading ratings from {app_state['ratings_file']}")
        app_state['rating_dict'] = load_ratings(app_state['ratings_file'])
        if not app_state['rating_dict']:
            app.logger.warning(f"No ratings found in {app_state['ratings_file']}. Starting with default ratings.")
            app_state['rating_dict'] = {}
            
            # Create empty ratings file if it doesn't exist
            if not os.path.exists(app_state['ratings_file']):
                app.logger.info(f"Creating empty ratings file: {app_state['ratings_file']}")
                save_ratings(app_state['rating_dict'], app_state['ratings_file'])
        else:
            app.logger.info(f"Loaded {len(app_state['rating_dict'])} team ratings")
            
        # Load rating changes
        app.logger.info(f"Loading rating changes from {app_state['changes_file']}")
        app_state['rating_changes'] = load_rating_changes(app_state['changes_file'])
        if not app_state['rating_changes']:
            app.logger.warning(f"No rating changes found in {app_state['changes_file']}. Starting with empty changes.")
            app_state['rating_changes'] = {}
            
            # Create empty changes file if it doesn't exist
            if not os.path.exists(app_state['changes_file']):
                app.logger.info(f"Creating empty changes file: {app_state['changes_file']}")
                save_rating_changes(app_state['rating_changes'], app_state['changes_file'])
        else:
            total_changes = sum(len(changes) for changes in app_state['rating_changes'].values())
            app.logger.info(f"Loaded rating changes for {len(app_state['rating_changes'])} teams with {total_changes} total changes")
    except Exception as e:
        app.logger.error(f"Error in load_data: {str(e)}", exc_info=True)
        raise

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/set_gender', methods=['POST'])
def set_gender():
    """Set the gender for the session."""
    gender = request.form.get('gender')
    app.logger.info(f"set_gender called with gender={gender}")
    app.logger.info(f"Form data received: {request.form}")
    
    if gender not in ['men', 'women']:
        app.logger.error(f"Invalid gender selection: '{gender}'")
        flash('Invalid gender selection.', 'error')
        return redirect(url_for('index'))
    
    try:
        app.logger.info(f"Loading data for gender: {gender}")
        load_data(gender)
        app.logger.info(f"Successfully loaded {gender}'s basketball data")
        flash(f'Loaded {gender}\'s basketball data.', 'success')
        return redirect(url_for('dashboard'))
    except Exception as e:
        app.logger.error(f"Error loading data for gender {gender}: {str(e)}", exc_info=True)
        flash(f'Error loading data: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Render the main dashboard."""
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    return render_template('dashboard.html', 
                          gender=app_state['gender'],
                          total_teams=len(app_state['rating_dict']),
                          can_revert=app_state['previous_rating_dict'] is not None,
                          last_update_info=app_state['last_update_info']['score'] if app_state['last_update_info'] else None)

@app.route('/search')
def search():
    """Search form for teams."""
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    return render_template('search_form.html', gender=app_state['gender'])

@app.route('/search_team', methods=['GET', 'POST'])
def search_team():
    """Search for a team."""
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        query = request.form.get('query', '')
        if not query:
            flash('Please enter a team name.', 'warning')
            return redirect(url_for('search_team'))
        
        matches = fuzzy_team_search(query, app_state['teams_df'])
        results = []
        
        for _, row in matches.iterrows():
            team_id = row['TeamID']
            rating = app_state['rating_dict'].get(team_id, 1500.0)
            results.append({
                'id': team_id,
                'name': row['TeamName'],
                'rating': rating
            })
        
        return render_template('search_results.html', 
                              query=query, 
                              results=results, 
                              gender=app_state['gender'])
    
    return render_template('search_form.html', gender=app_state['gender'])

@app.route('/spread_prediction', methods=['GET'])
def spread_prediction():
    """Show spread prediction form."""
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    return render_template('spread_form.html', gender=app_state['gender'])

@app.route('/predict_spread', methods=['GET', 'POST'])
def predict_spread():
    """Predict spread between two teams."""
    # Check if gender is provided in query parameters first
    gender_param = request.args.get('gender')
    if gender_param in ['men', 'women'] and gender_param != app_state['gender']:
        load_data(gender_param)
    
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        team_a = request.form.get('team_a', '')
        team_b = request.form.get('team_b', '')
        location = request.form.get('location', 'N')
        
        # Convert location code
        if location == "A":
            home_loc = "H"  # Team A is home
        elif location == "B":
            home_loc = "A"  # Team B is home (A from winner's perspective)
        else:
            home_loc = "N"  # Neutral
        
        # Find matching teams
        team_a_matches = None
        team_b_matches = None
        
        try:
            team_a_id = int(team_a)
            team_a_matches = app_state['teams_df'][app_state['teams_df']['TeamID'] == team_a_id]
        except ValueError:
            team_a_matches = fuzzy_team_search(team_a, app_state['teams_df'])
        
        try:
            team_b_id = int(team_b)
            team_b_matches = app_state['teams_df'][app_state['teams_df']['TeamID'] == team_b_id]
        except ValueError:
            team_b_matches = fuzzy_team_search(team_b, app_state['teams_df'])
        
        if team_a_matches.empty:
            flash(f"No teams found matching '{team_a}'", 'error')
            return redirect(url_for('predict_spread'))
        
        if team_b_matches.empty:
            flash(f"No teams found matching '{team_b}'", 'error')
            return redirect(url_for('predict_spread'))
        
        if len(team_a_matches) > 1 or len(team_b_matches) > 1:
            # Handle multiple matches in UI by showing options
            session['team_a_matches'] = team_a_matches.to_dict('records')
            session['team_b_matches'] = team_b_matches.to_dict('records')
            session['location'] = location
            return render_template('resolve_teams.html', 
                                 a_matches=team_a_matches.to_dict('records'),
                                 b_matches=team_b_matches.to_dict('records'),
                                 action='predict_spread',
                                 team_name_dict=app_state['team_name_dict'])
        
        team_a_id = team_a_matches.iloc[0]['TeamID']
        team_b_id = team_b_matches.iloc[0]['TeamID']
        
        # Get team names
        name_a = app_state['team_name_dict'].get(team_a_id, f"Team {team_a_id}")
        name_b = app_state['team_name_dict'].get(team_b_id, f"Team {team_b_id}")
        
        # Predict spread
        spread = predict_spread_single(
            teamA_id=team_a_id,
            teamB_id=team_b_id,
            location=home_loc,
            rating_dict=app_state['rating_dict'],
            men_slope=app_state['slope'] if app_state['is_men'] else None,
            women_slope=app_state['slope'] if not app_state['is_men'] else None,
            home_court_men=app_state['home_court'] if app_state['is_men'] else None,
            home_court_women=app_state['home_court'] if not app_state['is_men'] else None,
            team_name_dict_men=app_state['team_name_dict'] if app_state['is_men'] else None,
            team_name_dict_women=app_state['team_name_dict'] if not app_state['is_men'] else None
        )
        
        if spread > 0:
            result = f"{name_a} is favored by {spread:.2f} points over {name_b}"
            favorite = name_a
            underdog = name_b
            spread_value = abs(spread)
        elif spread < 0:
            result = f"{name_b} is favored by {abs(spread):.2f} points over {name_a}"
            favorite = name_b
            underdog = name_a
            spread_value = abs(spread)
        else:
            result = f"Pick'em: {name_a} vs {name_b}"
            favorite = None
            underdog = None
            spread_value = 0
        
        return render_template('spread_result.html', 
                              result=result,
                              team_a=name_a,
                              team_b=name_b, 
                              spread=spread,
                              favorite=favorite,
                              underdog=underdog,
                              spread_value=spread_value,
                              location_desc="neutral court" if home_loc == "N" else 
                                           f"{name_a}'s home court" if home_loc == "H" else 
                                           f"{name_b}'s home court",
                              gender=app_state['gender'])
    
    return render_template('spread_form.html', gender=app_state['gender'])

@app.route('/odds_prediction', methods=['GET'])
def odds_prediction():
    """Show odds prediction form."""
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    return render_template('odds_form.html', gender=app_state['gender'])

@app.route('/predict_odds', methods=['GET', 'POST'])
def predict_odds():
    """Predict probability and odds."""
    # Check if gender is provided in query parameters first
    gender_param = request.args.get('gender')
    if gender_param in ['men', 'women'] and gender_param != app_state['gender']:
        load_data(gender_param)
    
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        team_a = request.form.get('team_a', '')
        team_b = request.form.get('team_b', '')
        apply_ls = request.form.get('longshot', 'n') == 'y'
        
        # Find matching teams
        team_a_matches = None
        team_b_matches = None
        
        try:
            team_a_id = int(team_a)
            team_a_matches = app_state['teams_df'][app_state['teams_df']['TeamID'] == team_a_id]
        except ValueError:
            team_a_matches = fuzzy_team_search(team_a, app_state['teams_df'])
        
        try:
            team_b_id = int(team_b)
            team_b_matches = app_state['teams_df'][app_state['teams_df']['TeamID'] == team_b_id]
        except ValueError:
            team_b_matches = fuzzy_team_search(team_b, app_state['teams_df'])
        
        if team_a_matches.empty:
            flash(f"No teams found matching '{team_a}'", 'error')
            return redirect(url_for('predict_odds'))
        
        if team_b_matches.empty:
            flash(f"No teams found matching '{team_b}'", 'error')
            return redirect(url_for('predict_odds'))
        
        if len(team_a_matches) > 1 or len(team_b_matches) > 1:
            # Handle multiple matches in UI by showing options
            session['team_a_matches'] = team_a_matches.to_dict('records')
            session['team_b_matches'] = team_b_matches.to_dict('records')
            session['apply_longshot'] = apply_ls
            return render_template('resolve_teams.html', 
                                 a_matches=team_a_matches.to_dict('records'),
                                 b_matches=team_b_matches.to_dict('records'),
                                 action='predict_odds',
                                 team_name_dict=app_state['team_name_dict'])
        
        team_a_id = team_a_matches.iloc[0]['TeamID']
        team_b_id = team_b_matches.iloc[0]['TeamID']
        
        # Get team ratings
        r_a = app_state['rating_dict'].get(team_a_id, 1500.0)
        r_b = app_state['rating_dict'].get(team_b_id, 1500.0)
        
        # Get team names
        name_a = app_state['team_name_dict'].get(team_a_id, f"Team {team_a_id}")
        name_b = app_state['team_name_dict'].get(team_b_id, f"Team {team_b_id}")
        
        # Calculate probability and odds
        result = prob_and_odds(r_a, r_b, apply_ls=apply_ls)
        
        # Format odds for display
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
        
        return render_template('odds_result.html',
                              team_a=name_a,
                              team_b=name_b,
                              raw_prob_a=result['team_a']['raw_prob'],
                              raw_prob_b=result['team_b']['raw_prob'],
                              adj_prob_a=result['team_a'].get('adj_prob'),
                              adj_prob_b=result['team_b'].get('adj_prob'),
                              odds_a=odds_a_str,
                              odds_b=odds_b_str,
                              applied_longshot=apply_ls,
                              gender=app_state['gender'])
    
    return render_template('odds_form.html', gender=app_state['gender'])

@app.route('/resolve_teams', methods=['POST'])
def resolve_teams():
    """Resolve multiple team matches."""
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    action = request.form.get('action')
    team_a_id = int(request.form.get('team_a_id'))
    team_b_id = int(request.form.get('team_b_id'))
    
    if action == 'predict_spread':
        location = request.form.get('location', 'N')
        return redirect(url_for('predict_spread_resolved', 
                               team_a_id=team_a_id, 
                               team_b_id=team_b_id, 
                               location=location))
    
    elif action == 'predict_odds':
        apply_ls = request.form.get('apply_longshot', 'n') == 'y'
        return redirect(url_for('predict_odds_resolved', 
                               team_a_id=team_a_id, 
                               team_b_id=team_b_id, 
                               apply_longshot=apply_ls))
    
    elif action == 'update_game':
        score_a = int(request.form.get('score_a'))
        score_b = int(request.form.get('score_b'))
        location = request.form.get('location', 'N')
        return redirect(url_for('update_game_resolved', 
                               team_a_id=team_a_id, 
                               team_b_id=team_b_id, 
                               score_a=score_a,
                               score_b=score_b,
                               location=location))
    
    flash('Invalid action.', 'error')
    return redirect(url_for('dashboard'))

@app.route('/predict_spread_resolved')
def predict_spread_resolved():
    """Handle resolved team prediction."""
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    team_a_id = int(request.args.get('team_a_id'))
    team_b_id = int(request.args.get('team_b_id'))
    location = request.args.get('location', 'N')
    
    # Convert location code
    if location == "A":
        home_loc = "H"  # Team A is home
    elif location == "B":
        home_loc = "A"  # Team B is home (A from winner's perspective)
    else:
        home_loc = "N"  # Neutral
    
    # Get team names from the dictionary
    name_a = app_state['team_name_dict'].get(team_a_id, f"Team {team_a_id}")
    name_b = app_state['team_name_dict'].get(team_b_id, f"Team {team_b_id}")
    
    # Predict spread
    spread = predict_spread_single(
        teamA_id=team_a_id,
        teamB_id=team_b_id,
        location=home_loc,
        rating_dict=app_state['rating_dict'],
        men_slope=app_state['slope'] if app_state['is_men'] else None,
        women_slope=app_state['slope'] if not app_state['is_men'] else None,
        home_court_men=app_state['home_court'] if app_state['is_men'] else None,
        home_court_women=app_state['home_court'] if not app_state['is_men'] else None,
        team_name_dict_men=app_state['team_name_dict'] if app_state['is_men'] else None,
        team_name_dict_women=app_state['team_name_dict'] if not app_state['is_men'] else None
    )
    
    if spread > 0:
        result = f"{name_a} is favored by {spread:.2f} points over {name_b}"
        favorite = name_a
        underdog = name_b
        spread_value = abs(spread)
    elif spread < 0:
        result = f"{name_b} is favored by {abs(spread):.2f} points over {name_a}"
        favorite = name_b
        underdog = name_a
        spread_value = abs(spread)
    else:
        result = f"Pick'em: {name_a} vs {name_b}"
        favorite = None
        underdog = None
        spread_value = 0
    
    return render_template('spread_result.html', 
                          result=result,
                          team_a=name_a,
                          team_b=name_b, 
                          spread=spread,
                          favorite=favorite,
                          underdog=underdog,
                          spread_value=spread_value,
                          location_desc="neutral court" if home_loc == "N" else 
                                       f"{name_a}'s home court" if home_loc == "H" else 
                                       f"{name_b}'s home court",
                          gender=app_state['gender'])

@app.route('/predict_odds_resolved')
def predict_odds_resolved():
    """Handle resolved team odds prediction."""
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    team_a_id = int(request.args.get('team_a_id'))
    team_b_id = int(request.args.get('team_b_id'))
    apply_ls = request.args.get('apply_longshot') == 'True'
    
    # Get team ratings
    r_a = app_state['rating_dict'].get(team_a_id, 1500.0)
    r_b = app_state['rating_dict'].get(team_b_id, 1500.0)
    
    # Get team names
    name_a = app_state['team_name_dict'].get(team_a_id, f"Team {team_a_id}")
    name_b = app_state['team_name_dict'].get(team_b_id, f"Team {team_b_id}")
    
    # Calculate probability and odds
    result = prob_and_odds(r_a, r_b, apply_ls=apply_ls)
    
    # Format odds for display
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
    
    return render_template('odds_result.html',
                          team_a=name_a,
                          team_b=name_b,
                          raw_prob_a=result['team_a']['raw_prob'],
                          raw_prob_b=result['team_b']['raw_prob'],
                          adj_prob_a=result['team_a'].get('adj_prob'),
                          adj_prob_b=result['team_b'].get('adj_prob'),
                          odds_a=odds_a_str,
                          odds_b=odds_b_str,
                          applied_longshot=apply_ls,
                          gender=app_state['gender'])

@app.route('/update_game', methods=['GET', 'POST'])
def update_game():
    """Update ratings for a single game."""
    # Check if gender is provided in query parameters first
    gender_param = request.args.get('gender')
    if gender_param in ['men', 'women'] and gender_param != app_state['gender']:
        load_data(gender_param)
    
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        team_a = request.form.get('team_a', '')  # Winning team
        team_b = request.form.get('team_b', '')  # Losing team
        score_a = int(request.form.get('score_a', 0))  # Winning score
        score_b = int(request.form.get('score_b', 0))  # Losing score
        location = request.form.get('location', 'N')
        
        # Convert location code
        if location == "A":
            wloc = "H"  # Winner is home
        elif location == "B":
            wloc = "A"  # Loser is home (A from winner's perspective)
        else:
            wloc = "N"  # Neutral
        
        # Find matching teams
        team_a_matches = None
        team_b_matches = None
        
        try:
            team_a_id = int(team_a)
            team_a_matches = app_state['teams_df'][app_state['teams_df']['TeamID'] == team_a_id]
        except ValueError:
            team_a_matches = fuzzy_team_search(team_a, app_state['teams_df'])
        
        try:
            team_b_id = int(team_b)
            team_b_matches = app_state['teams_df'][app_state['teams_df']['TeamID'] == team_b_id]
        except ValueError:
            team_b_matches = fuzzy_team_search(team_b, app_state['teams_df'])
        
        if team_a_matches.empty:
            flash(f"No teams found matching '{team_a}'", 'error')
            return redirect(url_for('update_game'))
        
        if team_b_matches.empty:
            flash(f"No teams found matching '{team_b}'", 'error')
            return redirect(url_for('update_game'))
        
        if len(team_a_matches) > 1 or len(team_b_matches) > 1:
            # Handle multiple matches in UI by showing options
            session['team_a_matches'] = team_a_matches.to_dict('records')
            session['team_b_matches'] = team_b_matches.to_dict('records')
            session['score_a'] = score_a
            session['score_b'] = score_b
            session['location'] = location
            return render_template('resolve_teams.html', 
                                 a_matches=team_a_matches.to_dict('records'),
                                 b_matches=team_b_matches.to_dict('records'),
                                 action='update_game',
                                 score_a=score_a,
                                 score_b=score_b,
                                 location=location,
                                 team_name_dict=app_state['team_name_dict'])
        
        team_a_id = team_a_matches.iloc[0]['TeamID']
        team_b_id = team_b_matches.iloc[0]['TeamID']
        
        # Get team names
        name_a = app_state['team_name_dict'].get(team_a_id, f"Team {team_a_id}")
        name_b = app_state['team_name_dict'].get(team_b_id, f"Team {team_b_id}")
        
        # Show confirmation page
        location_desc = "neutral court" if wloc == "N" else f"{name_a}'s home court" if wloc == "H" else f"{name_b}'s home court"
        
        return render_template('update_confirmation.html',
                              winner_id=team_a_id,
                              loser_id=team_b_id,
                              winner_name=name_a,
                              loser_name=name_b,
                              winner_score=score_a,
                              loser_score=score_b,
                              location=location,
                              location_desc=location_desc,
                              gender=app_state['gender'])
    
    return render_template('update_form.html', gender=app_state['gender'])

@app.route('/update_game_resolved')
def update_game_resolved():
    """Handle resolved team game update."""
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    team_a_id = int(request.args.get('team_a_id'))  # Winning team
    team_b_id = int(request.args.get('team_b_id'))  # Losing team
    score_a = int(request.args.get('score_a', 0))  # Winning score
    score_b = int(request.args.get('score_b', 0))  # Losing score
    location = request.args.get('location', 'N')
    
    # Convert location code
    if location == "A":
        wloc = "H"  # Winner is home
    elif location == "B":
        wloc = "A"  # Loser is home (A from winner's perspective)
    else:
        wloc = "N"  # Neutral
    
    # Get team names
    name_a = app_state['team_name_dict'].get(team_a_id, f"Team {team_a_id}")
    name_b = app_state['team_name_dict'].get(team_b_id, f"Team {team_b_id}")
    
    # Show confirmation page
    location_desc = "neutral court" if wloc == "N" else f"{name_a}'s home court" if wloc == "H" else f"{name_b}'s home court"
    
    return render_template('update_confirmation.html',
                          winner_id=team_a_id,
                          loser_id=team_b_id,
                          winner_name=name_a,
                          loser_name=name_b,
                          winner_score=score_a,
                          loser_score=score_b,
                          location=location,
                          location_desc=location_desc,
                          gender=app_state['gender'])

@app.route('/confirm_update', methods=['POST'])
def confirm_update():
    """Confirm and process rating update."""
    winner_id = int(request.form.get('winner_id'))
    loser_id = int(request.form.get('loser_id'))
    winner_score = int(request.form.get('winner_score'))
    loser_score = int(request.form.get('loser_score'))
    location = request.form.get('location', 'N')
    
    # Get team names
    winner_name = app_state['team_name_dict'].get(winner_id, f"Team {winner_id}")
    loser_name = app_state['team_name_dict'].get(loser_id, f"Team {loser_id}")
    
    # Store previous ratings for revert functionality
    app_state['previous_rating_dict'] = app_state['rating_dict'].copy()
    
    # Store last update info
    app_state['last_update_info'] = {
        'winner': {
            'id': winner_id,
            'name': winner_name,
            'old_rating': app_state['rating_dict'].get(winner_id, 1500)
        },
        'loser': {
            'id': loser_id,
            'name': loser_name,
            'old_rating': app_state['rating_dict'].get(loser_id, 1500)
        },
        'score': f"{winner_score}-{loser_score}",
        'location': location
    }
    
    # Get gender-specific configuration
    gender_config = get_gender_config(app_state['gender'], app_state['config'])
    
    # Update ratings
    if app_state['is_men']:
        k_factor = gender_config['k_factor']
        alpha_value = gender_config['alpha']
    else:
        k_factor = gender_config['k_factor']
        alpha_value = gender_config['alpha']
    
    # Update the ratings
    new_ratings = update_single_game(
        winner_id, loser_id, winner_score, loser_score, location, 
        app_state['rating_dict'], 
        k=k_factor,
        alpha=alpha_value,
        home_court=app_state['home_court'],
        mov_formula=gender_config['mov_formula']
    )
    
    # Record the old ratings
    old_rating_winner = app_state['rating_dict'].get(winner_id, 1500)
    old_rating_loser = app_state['rating_dict'].get(loser_id, 1500)
    
    # Update the rating dict with new values
    app_state['rating_dict'].update(new_ratings)
    
    # Update rating_changes dictionary with timestamp
    import datetime
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if winner_id not in app_state['rating_changes']:
        app_state['rating_changes'][winner_id] = []
    if loser_id not in app_state['rating_changes']:
        app_state['rating_changes'][loser_id] = []
    
    app_state['rating_changes'][winner_id].append({
        'timestamp': now,
        'old_rating': old_rating_winner,
        'new_rating': app_state['rating_dict'][winner_id],
        'change': app_state['rating_dict'][winner_id] - old_rating_winner,
        'opponent': loser_id,
        'opponent_name': loser_name,
        'won': True,
        'score': f"{winner_score}-{loser_score}"
    })
    
    app_state['rating_changes'][loser_id].append({
        'timestamp': now,
        'old_rating': old_rating_loser,
        'new_rating': app_state['rating_dict'][loser_id],
        'change': app_state['rating_dict'][loser_id] - old_rating_loser,
        'opponent': winner_id,
        'opponent_name': winner_name,
        'won': False,
        'score': f"{winner_score}-{loser_score}"
    })
    
    # Save to file
    save_ratings(app_state['rating_dict'], app_state['ratings_file'])
    
    # Save rating changes to file
    save_rating_changes(app_state['rating_changes'], app_state['changes_file'])
    
    # Compute rating changes
    new_rating_winner = app_state['rating_dict'][winner_id]
    new_rating_loser = app_state['rating_dict'][loser_id]
    winner_change = new_rating_winner - old_rating_winner
    loser_change = new_rating_loser - old_rating_loser
    
    # Store last update info with new ratings
    app_state['last_update_info']['winner']['new_rating'] = new_rating_winner
    app_state['last_update_info']['loser']['new_rating'] = new_rating_loser
    
    flash('Ratings successfully updated.', 'success')
    
    return render_template(
        'update_result.html',
        winner_name=winner_name,
        loser_name=loser_name,
        old_rating_winner=old_rating_winner,
        new_rating_winner=new_rating_winner,
        old_rating_loser=old_rating_loser,
        new_rating_loser=new_rating_loser,
        winner_change=winner_change,
        loser_change=loser_change,
        gender=app_state['gender']
    )

@app.route('/top_teams', methods=['GET', 'POST'])
def top_teams():
    """Show top teams."""
    # Check if gender is provided in query parameters first
    gender_param = request.args.get('gender')
    if gender_param in ['men', 'women'] and gender_param != app_state['gender']:
        load_data(gender_param)
    
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    count = 10  # Default
    
    if request.method == 'POST':
        count = int(request.form.get('count', 10))
    
    # Sort by rating and get top N
    top_teams_list = []
    
    for team_id, rating in sorted(app_state['rating_dict'].items(), key=lambda x: x[1], reverse=True)[:count]:
        team_name = app_state['team_name_dict'].get(team_id, f"Team {team_id}")
        
        # Calculate change or trend
        change = 0
        trend = "neutral"
        
        # Check if we have change history for this team
        if team_id in app_state['rating_changes'] and len(app_state['rating_changes'][team_id]) > 0:
            # Get the latest change
            latest_change = app_state['rating_changes'][team_id][-1]
            change = latest_change['change']
            
            # Determine trend based on recent changes
            trend_count = min(3, len(app_state['rating_changes'][team_id]))
            if trend_count > 0:
                recent_changes = [entry['change'] for entry in app_state['rating_changes'][team_id][-trend_count:]]
                # If most recent changes are positive, trend is up
                if sum(1 for c in recent_changes if c > 0) > trend_count / 2:
                    trend = "up"
                # If most recent changes are negative, trend is down
                elif sum(1 for c in recent_changes if c < 0) > trend_count / 2:
                    trend = "down"
        
        top_teams_list.append({
            'id': team_id,
            'name': team_name,
            'rating': rating,
            'change': change,
            'trend': trend
        })
    
    return render_template('top_teams.html', 
                          teams=top_teams_list, 
                          count=count,
                          gender=app_state['gender'])

@app.route('/revert_confirm')
def revert_confirm():
    """Show revert confirmation page."""
    # Check if gender is provided in query parameters first
    gender_param = request.args.get('gender')
    if gender_param in ['men', 'women'] and gender_param != app_state['gender']:
        load_data(gender_param)
    
    if app_state['gender'] is None or app_state['previous_rating_dict'] is None:
        flash('Nothing to revert.', 'warning')
        return redirect(url_for('dashboard'))
    
    return render_template('revert_confirm.html',
                          winner_name=app_state['last_update_info']['winner']['name'],
                          loser_name=app_state['last_update_info']['loser']['name'],
                          score=app_state['last_update_info']['score'],
                          location=app_state['last_update_info']['location'],
                          gender=app_state['gender'])

@app.route('/revert_update', methods=['POST'])
def revert_update():
    """Revert the last update."""
    if app_state['gender'] is None or app_state['previous_rating_dict'] is None:
        flash('Nothing to revert.', 'warning')
        return redirect(url_for('dashboard'))
    
    # Get details for display
    winner_info = app_state['last_update_info']['winner']
    loser_info = app_state['last_update_info']['loser']
    team1_name = winner_info['name']
    team2_name = loser_info['name']
    team1_old_rating = winner_info['old_rating']
    team2_old_rating = loser_info['old_rating']
    team1_new_rating = app_state['rating_dict'][winner_info['id']]
    team2_new_rating = app_state['rating_dict'][loser_info['id']]
    
    # Remove the last entry from the rating changes history for both teams
    if winner_info['id'] in app_state['rating_changes'] and len(app_state['rating_changes'][winner_info['id']]) > 0:
        app_state['rating_changes'][winner_info['id']].pop()  # Remove last entry
    
    if loser_info['id'] in app_state['rating_changes'] and len(app_state['rating_changes'][loser_info['id']]) > 0:
        app_state['rating_changes'][loser_info['id']].pop()  # Remove last entry
    
    # Save rating changes to file
    save_rating_changes(app_state['rating_changes'], app_state['changes_file'])
    
    # Restore previous ratings
    app_state['rating_dict'] = app_state['previous_rating_dict']
    
    # Save the reverted ratings
    save_ratings(app_state['rating_dict'], app_state['ratings_file'])
    
    flash(f'Reverted last update and saved to {app_state["ratings_file"]}', 'success')
    
    # Calculate changes for display
    team1_change = team1_old_rating - team1_new_rating
    team2_change = team2_old_rating - team2_new_rating
    
    # Clear revert data
    app_state['previous_rating_dict'] = None
    app_state['last_update_info'] = None
    
    return render_template('revert_result.html',
                          team1_name=team1_name,
                          team2_name=team2_name,
                          team1_old_rating=team1_new_rating,
                          team2_old_rating=team2_new_rating,
                          team1_new_rating=team1_old_rating,
                          team2_new_rating=team2_old_rating,
                          team1_change=team1_change,
                          team2_change=team2_change,
                          gender=app_state['gender'])

@app.route('/spread_result')
def spread_result():
    """Display spread prediction result."""
    if app_state['gender'] is None:
        flash('Please select a gender first.', 'warning')
        return redirect(url_for('index'))
    
    result = session.get('spread_result')
    
    if not result:
        flash('No prediction data available.', 'warning')
        return redirect(url_for('spread_prediction'))
    
    gender_config = get_gender_config(app_state['gender'], app_state['config'])
    
    return render_template('spread_result.html', 
                          team_a=result['team_a_name'],
                          team_b=result['team_b_name'],
                          spread=result['spread'],
                          spread_value=abs(result['spread']),
                          favorite=result['favorite'],
                          underdog=result['underdog'],
                          location_desc=result['location_desc'],
                          gender=app_state['gender'],
                          slope=gender_config['slope'],
                          home_court=gender_config['home_court'])

@app.route('/predict_spread', methods=['POST'])
def spread_prediction_post():
    """Handle spread prediction form submission."""
    if not app_state['rating_dict']:
        flash('No ratings data available. Please run an ELO calculation first.', 'warning')
        return redirect(url_for('spread_prediction'))
    
    team_a_query = request.form.get('team_a', '')
    team_b_query = request.form.get('team_b', '')
    location = request.form.get('location', 'N')
    
    # Check for empty inputs
    if not team_a_query or not team_b_query:
        flash('Please enter both teams.', 'danger')
        return redirect(url_for('spread_prediction'))
    
    # Get gender-specific configuration
    gender_config = get_gender_config(app_state['gender'], app_state['config'])
    
    # Search for teams
    team_a_matches = fuzzy_team_search(team_a_query, app_state['teams_df'], exact=False)
    team_b_matches = fuzzy_team_search(team_b_query, app_state['teams_df'], exact=False)
    
    # Handle multiple/no matches
    if len(team_a_matches) == 0:
        flash(f"No teams found matching '{team_a_query}'", 'danger')
        return redirect(url_for('spread_prediction'))
    
    if len(team_b_matches) == 0:
        flash(f"No teams found matching '{team_b_query}'", 'danger')
        return redirect(url_for('spread_prediction'))
    
    # If multiple matches for either team, go to resolve page
    if len(team_a_matches) > 1 or len(team_b_matches) > 1:
        # Store match data in session for resolve page
        session['a_matches'] = team_a_matches.to_dict('records')
        session['b_matches'] = team_b_matches.to_dict('records')
        session['action'] = 'spread'
        session['location'] = location
        
        return redirect(url_for('resolve_teams'))
    
    # Extract team IDs
    team_a_id = int(team_a_matches.iloc[0]['TeamID'])
    team_b_id = int(team_b_matches.iloc[0]['TeamID'])
    
    team_a_name = app_state['team_name_dict'].get(team_a_id, f"Team {team_a_id}")
    team_b_name = app_state['team_name_dict'].get(team_b_id, f"Team {team_b_id}")
    
    # Get location description
    if location == 'N':
        location_desc = "a neutral court"
    elif location == 'A':
        location_desc = f"{team_a_name}'s home court"
    else:  # location == 'B'
        location_desc = f"{team_b_name}'s home court"
    
    # Calculate spread
    spread = predict_spread_single(
        teamA_id=team_a_id,
        teamB_id=team_b_id,
        location=location,
        rating_dict=app_state['rating_dict'],
        men_slope=gender_config['slope'] if app_state['is_men'] else None,
        women_slope=gender_config['slope'] if not app_state['is_men'] else None,
        home_court_men=gender_config['home_court'] if app_state['is_men'] else None,
        home_court_women=gender_config['home_court'] if not app_state['is_men'] else None
    )
    
    # Determine favorite/underdog
    if spread > 0:
        favorite = team_a_name
        underdog = team_b_name
    elif spread < 0:
        favorite = team_b_name
        underdog = team_a_name
    else:
        favorite = None
        underdog = None
    
    # Store result in session
    result = {
        'team_a_name': team_a_name,
        'team_b_name': team_b_name,
        'spread': spread,
        'favorite': favorite,
        'underdog': underdog,
        'location_desc': location_desc
    }
    session['spread_result'] = result
    
    return redirect(url_for('spread_result'))

@app.route('/predict_odds', methods=['POST'])
def odds_prediction_post():
    """Handle odds prediction form submission."""
    if not app_state['rating_dict']:
        flash('No ratings data available. Please run an ELO calculation first.', 'warning')
        return redirect(url_for('odds_prediction'))
    
    team_a_query = request.form.get('team_a', '')
    team_b_query = request.form.get('team_b', '')
    apply_longshot_bias = request.form.get('longshot') == 'y'
    
    # Check for empty inputs
    if not team_a_query or not team_b_query:
        flash('Please enter both teams.', 'danger')
        return redirect(url_for('odds_prediction'))
    
    # Get gender-specific configuration
    gender_config = get_gender_config(app_state['gender'], app_state['config'])
    
    # Search for teams
    team_a_matches = fuzzy_team_search(team_a_query, app_state['teams_df'], exact=False)
    team_b_matches = fuzzy_team_search(team_b_query, app_state['teams_df'], exact=False)
    
    # Handle multiple/no matches
    if len(team_a_matches) == 0:
        flash(f"No teams found matching '{team_a_query}'", 'danger')
        return redirect(url_for('odds_prediction'))
    
    if len(team_b_matches) == 0:
        flash(f"No teams found matching '{team_b_query}'", 'danger')
        return redirect(url_for('odds_prediction'))
    
    # If multiple matches for either team, go to resolve page
    if len(team_a_matches) > 1 or len(team_b_matches) > 1:
        # Store match data in session for resolve page
        session['a_matches'] = team_a_matches.to_dict('records')
        session['b_matches'] = team_b_matches.to_dict('records')
        session['action'] = 'odds'
        session['apply_longshot'] = 'y' if apply_longshot_bias else 'n'
        
        return redirect(url_for('resolve_teams'))
    
    # Extract team IDs
    team_a_id = int(team_a_matches.iloc[0]['TeamID'])
    team_b_id = int(team_b_matches.iloc[0]['TeamID'])
    
    team_a_name = app_state['team_name_dict'].get(team_a_id, f"Team {team_a_id}")
    team_b_name = app_state['team_name_dict'].get(team_b_id, f"Team {team_b_id}")
    
    # Calculate spread
    spread = predict_spread_single(
        teamA_id=team_a_id,
        teamB_id=team_b_id,
        location='N',
        rating_dict=app_state['rating_dict'],
        men_slope=gender_config['slope'] if app_state['is_men'] else None,
        women_slope=gender_config['slope'] if not app_state['is_men'] else None,
        home_court_men=gender_config['home_court'] if app_state['is_men'] else None,
        home_court_women=gender_config['home_court'] if not app_state['is_men'] else None
    )
    
    # Get ratings
    r_a = app_state['rating_dict'].get(team_a_id, gender_config['initial_rating'])
    r_b = app_state['rating_dict'].get(team_b_id, gender_config['initial_rating'])
    
    # Calculate raw probabilities
    # 10^(r_a/400) / (10^(r_a/400) + 10^(r_b/400))
    expected_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
    expected_b = 1.0 - expected_a
    
    # Apply longshot bias if requested
    if apply_longshot_bias:
        adj_prob_a, adj_prob_b = apply_longshot(expected_a, expected_b)
    else:
        adj_prob_a, adj_prob_b = None, None
    
    # Calculate odds
    odds_a = to_american_odds(adj_prob_a if apply_longshot_bias else expected_a)
    odds_b = to_american_odds(adj_prob_b if apply_longshot_bias else expected_b)
    
    # Store result in session
    result = {
        'team_a': team_a_name,
        'team_b': team_b_name,
        'raw_prob_a': expected_a,
        'raw_prob_b': expected_b,
        'adj_prob_a': adj_prob_a,
        'adj_prob_b': adj_prob_b,
        'odds_a': odds_a,
        'odds_b': odds_b,
        'applied_longshot': apply_longshot_bias
    }
    session['odds_result'] = result
    
    return redirect(url_for('odds_result'))

def run_web_ui(host='0.0.0.0', port=5000, debug=True):
    """
    Run the web UI.
    
    Parameters
    ----------
    host : str
        Host to bind to
    port : int
        Port to bind to
    debug : bool
        Whether to run in debug mode
    """
    app.run(host=host, port=port, debug=debug) 