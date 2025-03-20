# Hoops ELO Rating System

A modular Python package for calculating and predicting ELO ratings for NCAA basketball teams. This system handles both men's and women's basketball data from 1985 through 2025, with support for:

- Full ELO calculation with margin-of-victory, home-court advantage, and partial season carryover
- Parameter tuning to find optimal ELO settings
- Single-game ELO updates
- Spread predictions based on ELO differences
- Probability and American odds calculations with optional longshot bias
- Team ranking and analysis

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/hoops-elo.git
   cd hoops-elo
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Files

Place all NCAA basketball data files in the `data/` directory. Required files include:

- `MRegularSeasonCompactResults.csv` - Men's regular season results
- `MNCAATourneyCompactResults.csv` - Men's tournament results
- `MTeams.csv` - Men's team information
- `WRegularSeasonCompactResults.csv` - Women's regular season results
- `WNCAATourneyCompactResults.csv` - Women's tournament results
- `WTeams.csv` - Women's team information

## Usage

### Running Full ELO Calculation

To run the full ELO calculation for both men's and women's basketball:

```
python main.py run --gender both --start-year 1985 --end-year 2025
```

This will:
1. Load all game data from 1985-2025
2. Calculate ELO ratings using optimized parameters
3. Save final ratings to JSON files
4. Display top teams

### Updating Ratings for a Single Game

Update ELO ratings after a single game result:

```
python main.py update --gender men --team-a "Duke" --team-b "North Carolina" --score-a 72 --score-b 67 --location A
```

This updates the men's ratings with Duke (home team) beating North Carolina 72-67.

### Predicting Game Spreads

Get a predicted point spread between two teams:

```
python main.py spread --gender women --team-a "Connecticut" --team-b "Stanford" --location N
```

This predicts the point spread for a neutral-site women's game between UConn and Stanford.

### Calculating Probabilities and Odds

Calculate win probability and American odds for a matchup:

```
python main.py odds --gender men --team-a "Gonzaga" --team-b "Baylor" --longshot
```

This predicts the probability and American odds with longshot bias applied.

### Interactive Mode

For a text-based interactive interface:

```
python main.py interactive --gender men
```

In interactive mode, you can:
- Search for teams by partial name
- Predict spreads between teams
- Calculate probabilities and odds
- Update ratings for single games
- View top-ranked teams

## Module Structure

- `elo_ratings/`: Main package
  - `elo.py`: Core ELO calculation logic
  - `data_loader.py`: Data loading functions
  - `utils.py`: Utility functions (odds conversion, etc.)
  - `analysis.py`: Functions for analysis and visualization
  - `cli.py`: Command-line interface
- `main.py`: Main entry point
- `requirements.txt`: Package dependencies
- `README.md`: This file

## Examples

### Updating a Single Game Result

```python
from elo_ratings.data_loader import load_mens_data, make_team_name_dict
from elo_ratings.elo import load_ratings, update_single_game, save_ratings, fuzzy_team_search

# Load team data and name dictionary
_, _, teams_df, _, _ = load_mens_data()
team_name_dict = make_team_name_dict(teams_df)

# Load current ratings
ratings = load_ratings("elo_ratings_men_2025.json")

# Find team IDs by partial name match
team_a_matches = fuzzy_team_search("Duke", teams_df)
team_b_matches = fuzzy_team_search("North Carolina", teams_df)

# Update ratings for the game
updated_ratings, _ = update_single_game(
    win_team_id=team_a_matches.iloc[0]['TeamID'],
    lose_team_id=team_b_matches.iloc[0]['TeamID'], 
    win_score=72,
    lose_score=67,
    location="H",  # Duke is home
    rating_dict=ratings,
    k=140,
    alpha=40,
    home_court=75,
    mov_formula="linear"
)

# Save updated ratings
save_ratings(updated_ratings, "elo_ratings_men_2025.json")
```

### Converting Probability to American Odds

```python
from elo_ratings.utils import to_american_odds, apply_longshot

# Raw probability
raw_prob = 0.75  # 75% chance to win

# Without longshot bias
american_odds = to_american_odds(raw_prob)
print(f"American odds: {american_odds}")  # -300

# With longshot bias
adj_prob = apply_longshot(raw_prob)
american_odds_ls = to_american_odds(adj_prob)
print(f"Adjusted probability: {adj_prob:.3f}")
print(f"American odds with longshot: {american_odds_ls}")
```

## License

[MIT License](LICENSE) 