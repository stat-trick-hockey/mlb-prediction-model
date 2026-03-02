"""
data/fetch_odds.py
Fetches MLB moneyline, run line, and O/U odds from The Odds API.
Free tier: 500 requests/month — use sparingly, cache results.
Sign up at https://the-odds-api.com
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY     = "baseball_mlb"
REGIONS       = "us"
MARKETS       = "h2h,spreads,totals"
BOOKMAKER_PRIORITY = ["fanduel", "draftkings", "betmgm", "caesars", "bovada"]


def fetch_mlb_odds(game_date: str = None) -> pd.DataFrame:
    """
    Fetch current MLB odds for all upcoming games.
    Returns DataFrame with moneyline, run line, and O/U per game.
    """
    if not ODDS_API_KEY:
        print("WARNING: ODDS_API_KEY not set. Returning mock odds.")
        return _mock_odds()

    url = f"{ODDS_BASE_URL}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey":  ODDS_API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": "american",
        "dateFormat":  "iso",
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    games = resp.json()

    # Log remaining requests
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  Odds API requests remaining: {remaining}")

    rows = []
    for game in games:
        row = _parse_game_odds(game)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def _parse_game_odds(game: dict) -> dict:
    """Extract best available line across bookmakers for each market."""
    home_team = game.get("home_team", "")
    away_team = game.get("away_team", "")
    commence  = game.get("commence_time", "")

    result = {
        "game_id":    game.get("id"),
        "home_team":  home_team,
        "away_team":  away_team,
        "commence":   commence,
        # Moneyline
        "ml_home":    None,
        "ml_away":    None,
        "ml_implied_home": None,
        "ml_implied_away": None,
        # Run line (spread)
        "rl_home_line":  None,
        "rl_home_odds":  None,
        "rl_away_line":  None,
        "rl_away_odds":  None,
        # Over/Under
        "ou_total":      None,
        "ou_over_odds":  None,
        "ou_under_odds": None,
        "ou_implied_over":  None,
    }

    bookmakers = game.get("bookmakers", [])
    # Prefer specific bookmakers in order
    bookmakers_sorted = sorted(
        bookmakers,
        key=lambda b: BOOKMAKER_PRIORITY.index(b["key"]) if b["key"] in BOOKMAKER_PRIORITY else 99
    )

    for bm in bookmakers_sorted:
        for market in bm.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])

            if key == "h2h" and result["ml_home"] is None:
                for o in outcomes:
                    if o["name"] == home_team:
                        result["ml_home"] = o["price"]
                        result["ml_implied_home"] = american_to_implied(o["price"])
                    elif o["name"] == away_team:
                        result["ml_away"] = o["price"]
                        result["ml_implied_away"] = american_to_implied(o["price"])

            elif key == "spreads" and result["rl_home_line"] is None:
                for o in outcomes:
                    if o["name"] == home_team:
                        result["rl_home_line"] = o.get("point")
                        result["rl_home_odds"] = o["price"]
                    elif o["name"] == away_team:
                        result["rl_away_line"] = o.get("point")
                        result["rl_away_odds"] = o["price"]

            elif key == "totals" and result["ou_total"] is None:
                for o in outcomes:
                    if o["name"] == "Over":
                        result["ou_total"]      = o.get("point")
                        result["ou_over_odds"]  = o["price"]
                        result["ou_implied_over"] = american_to_implied(o["price"])
                    elif o["name"] == "Under":
                        result["ou_under_odds"] = o["price"]

    return result


def american_to_implied(american_odds: float) -> float:
    """Convert American moneyline odds to implied probability (0–1)."""
    if american_odds is None:
        return np.nan
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def implied_to_american(prob: float) -> int:
    """Convert implied probability to American odds."""
    if prob >= 0.5:
        return int(-(prob / (1 - prob)) * 100)
    else:
        return int(((1 - prob) / prob) * 100)


def save_odds(df: pd.DataFrame, game_date: str):
    out_dir = f"data/raw/{game_date}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/odds.csv"
    df.to_csv(path, index=False)
    print(f"Saved odds to {path}")


# ── Mock data ─────────────────────────────────────────────────────────────────

def _mock_odds(n_games: int = 8) -> pd.DataFrame:
    """Generate mock odds for testing."""
    np.random.seed(55)
    rows = []
    teams = ["NYY", "BOS", "LAD", "HOU", "ATL", "NYM", "CHC", "SF",
             "PHI", "SEA", "SD", "TB", "MIN", "CLE", "BAL", "TOR"]
    pairs = [(teams[i], teams[i+1]) for i in range(0, min(n_games*2, len(teams)), 2)]

    for home, away in pairs:
        ml_home_odds = np.random.choice([-140, -130, -120, -110, +105, +115, +125])
        ml_away_odds = implied_to_american(1 - american_to_implied(ml_home_odds) + 0.04)

        rows.append({
            "home_team":         home,
            "away_team":         away,
            "ml_home":           ml_home_odds,
            "ml_away":           ml_away_odds,
            "ml_implied_home":   round(american_to_implied(ml_home_odds), 4),
            "ml_implied_away":   round(american_to_implied(ml_away_odds), 4),
            "rl_home_line":      -1.5,
            "rl_home_odds":      np.random.choice([+130, +140, +150, +160]),
            "rl_away_line":      +1.5,
            "rl_away_odds":      np.random.choice([-140, -150, -160, -170]),
            "ou_total":          round(np.random.uniform(7.5, 10.0) * 2) / 2,
            "ou_over_odds":      -110,
            "ou_under_odds":     -110,
            "ou_implied_over":   0.524,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Fetching MLB odds...")
    df = fetch_mlb_odds()
    print(df[["home_team", "away_team", "ml_home", "ml_away", "ou_total"]])
    save_odds(df, date.today().strftime("%Y-%m-%d"))
