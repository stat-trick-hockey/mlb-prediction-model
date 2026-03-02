"""
data/fetch_schedule.py
Fetches today's (or any date's) MLB schedule with probable pitchers,
venue, game time, and team info from the free MLB Stats API.
"""

import requests
import pandas as pd
from datetime import date, datetime
from typing import Optional
import json
import os

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def fetch_schedule(game_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch MLB schedule for a given date (YYYY-MM-DD). Defaults to today.
    Returns a DataFrame with one row per game.
    """
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    url = f"{MLB_API_BASE}/schedule"
    params = {
        "sportId": 1,
        "date": game_date,
        "hydrate": "probablePitcher,venue,team,linescore",
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            games.append(_parse_game(game))

    df = pd.DataFrame(games)
    return df


def _parse_game(game: dict) -> dict:
    """Parse a single game dict from the API response."""
    home = game.get("teams", {}).get("home", {})
    away = game.get("teams", {}).get("away", {})

    home_team = home.get("team", {})
    away_team = away.get("team", {})

    home_pitcher = home.get("probablePitcher", {})
    away_pitcher = away.get("probablePitcher", {})

    venue = game.get("venue", {})

    return {
        "game_pk":            game.get("gamePk"),
        "game_date":          game.get("officialDate"),
        "game_datetime":      game.get("gameDate"),
        "status":             game.get("status", {}).get("abstractGameState"),
        "venue_name":         venue.get("name"),
        "home_team_id":       home_team.get("id"),
        "home_team_name":     home_team.get("name"),
        "home_team_abb":      home_team.get("abbreviation"),
        "away_team_id":       away_team.get("id"),
        "away_team_name":     away_team.get("name"),
        "away_team_abb":      away_team.get("abbreviation"),
        "home_pitcher_id":    home_pitcher.get("id"),
        "home_pitcher_name":  home_pitcher.get("fullName"),
        "away_pitcher_id":    away_pitcher.get("id"),
        "away_pitcher_name":  away_pitcher.get("fullName"),
        "home_score":         home.get("score"),
        "away_score":         away.get("score"),
    }


def fetch_season_schedule(season: int) -> pd.DataFrame:
    """
    Fetch the full schedule for a given season including results.
    Used to build historical training data.
    """
    url = f"{MLB_API_BASE}/schedule"
    params = {
        "sportId": 1,
        "season": season,
        "gameType": "R",  # Regular season only
        "hydrate": "probablePitcher,venue,team,linescore",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            row = _parse_game(game)
            games.append(row)

    df = pd.DataFrame(games)
    print(f"  Fetched {len(df)} games for {season} season")
    return df


def save_schedule(df: pd.DataFrame, game_date: str):
    """Save schedule to raw data directory."""
    out_dir = f"data/raw/{game_date}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/schedule.csv"
    df.to_csv(path, index=False)
    print(f"Saved schedule to {path}")


if __name__ == "__main__":
    print("Fetching today's schedule...")
    df = fetch_schedule()
    print(df[["home_team_abb", "away_team_abb", "home_pitcher_name", "away_pitcher_name", "venue_name"]])
    save_schedule(df, date.today().strftime("%Y-%m-%d"))
