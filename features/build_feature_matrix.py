"""
features/build_feature_matrix.py
Assembles the master feature matrix: one row per game with all pitcher,
bullpen, team form, Statcast, park, and weather features.
Handles both historical (training) and today's (prediction) modes.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import date, datetime
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PARK_FACTORS, OUTDOOR_PARKS
from features.pitcher_features import (
    build_pitcher_features, add_pitcher_statcast, build_pitcher_matchup_advantage
)
from features.bullpen_features  import build_bullpen_features
from features.team_form_features import build_team_form_features


def build_game_features(
    game_row: pd.Series,
    pitcher_stats_df: pd.DataFrame,
    team_batting_df: pd.DataFrame,
    team_pitching_df: pd.DataFrame,
    statcast_df: Optional[pd.DataFrame] = None,
    results_df: Optional[pd.DataFrame] = None,
    bullpen_log_df: Optional[pd.DataFrame] = None,
    weather_row: Optional[dict] = None,
    pitcher_statcast_home: Optional[dict] = None,
    pitcher_statcast_away: Optional[dict] = None,
) -> dict:
    """
    Build the complete feature vector for a single game.
    Returns a flat dict of all features.
    """
    features = {}
    game_date = str(game_row.get("game_date", ""))

    # ── Identifiers (not model inputs, stored for lookup) ─────────────────────
    features["game_pk"]        = game_row.get("game_pk")
    features["game_date"]      = game_date
    features["home_team_abb"]  = game_row.get("home_team_abb")
    features["away_team_abb"]  = game_row.get("away_team_abb")
    features["home_pitcher"]   = game_row.get("home_pitcher_name")
    features["away_pitcher"]   = game_row.get("away_pitcher_name")
    features["venue"]          = game_row.get("venue_name")

    home_abb = game_row.get("home_team_abb", "")
    away_abb = game_row.get("away_team_abb", "")

    # ── Park factor ──────────────────────────────────────────────────────────
    features["park_factor"]   = PARK_FACTORS.get(home_abb, 100)
    features["is_outdoor"]    = int(home_abb in OUTDOOR_PARKS)

    # ── Weather (outdoor parks only) ──────────────────────────────────────────
    if weather_row:
        features["temp_f"]           = weather_row.get("temp_f", 72)
        features["wind_mph"]         = weather_row.get("wind_mph", 0)
        features["wind_out_to_cf"]   = weather_row.get("wind_out_to_cf", 0)
        features["precip_chance"]    = weather_row.get("precip_chance", 0)
    else:
        features["temp_f"]           = 72
        features["wind_mph"]         = 0
        features["wind_out_to_cf"]   = 0
        features["precip_chance"]    = 0

    # ── Starting pitcher features ─────────────────────────────────────────────
    home_sp = build_pitcher_features(game_row, pitcher_stats_df, side="home")
    away_sp = build_pitcher_features(game_row, pitcher_stats_df, side="away")

    if pitcher_statcast_home:
        home_sp = add_pitcher_statcast(home_sp, pitcher_statcast_home, "home_sp_")
    if pitcher_statcast_away:
        away_sp = add_pitcher_statcast(away_sp, pitcher_statcast_away, "away_sp_")

    features.update(home_sp)
    features.update(away_sp)
    features.update(build_pitcher_matchup_advantage(home_sp, away_sp))

    # ── Bullpen features ──────────────────────────────────────────────────────
    home_bp = build_bullpen_features(home_abb, game_date, bullpen_log_df, side="home")
    away_bp = build_bullpen_features(away_abb, game_date, bullpen_log_df, side="away")
    features.update(home_bp)
    features.update(away_bp)

    # ── Team form features ────────────────────────────────────────────────────
    home_form = build_team_form_features(
        team_abb=home_abb, game_date=game_date, is_home=True,
        results_df=results_df, statcast_df=statcast_df,
        team_batting_df=team_batting_df, team_pitching_df=team_pitching_df,
    )
    away_form = build_team_form_features(
        team_abb=away_abb, game_date=game_date, is_home=False,
        results_df=results_df, statcast_df=statcast_df,
        team_batting_df=team_batting_df, team_pitching_df=team_pitching_df,
    )
    features.update(home_form)
    features.update(away_form)

    # ── Differentials ─────────────────────────────────────────────────────────
    features["woba_diff"]       = features.get("home_team_woba", 0.320) - features.get("away_team_woba", 0.320)
    features["era_diff"]        = features.get("away_team_team_era", 4.2) - features.get("home_team_team_era", 4.2)
    features["rd_diff"]         = features.get("home_team_rd_last10", 0) - features.get("away_team_rd_last10", 0)
    features["win_pct_diff"]    = features.get("home_team_win_pct_last10", 0.5) - features.get("away_team_win_pct_last10", 0.5)
    features["xwoba_diff"]      = features.get("home_team_xwoba", 0.320) - features.get("away_team_xwoba", 0.320)
    features["barrel_pct_diff"] = features.get("home_team_barrel_pct", 0.075) - features.get("away_team_barrel_pct", 0.075)

    return features


def build_feature_matrix_for_date(
    schedule_df: pd.DataFrame,
    pitcher_stats_df: pd.DataFrame,
    team_batting_df: pd.DataFrame,
    team_pitching_df: pd.DataFrame,
    statcast_df: Optional[pd.DataFrame] = None,
    results_df: Optional[pd.DataFrame] = None,
    bullpen_log_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build feature matrix for all games in a schedule DataFrame.
    """
    rows = []
    for _, game in schedule_df.iterrows():
        weather_row = None
        if weather_df is not None and not weather_df.empty and "game_pk" in weather_df.columns:
            w = weather_df[weather_df["game_pk"] == game.get("game_pk")]
            if not w.empty:
                weather_row = w.iloc[0].to_dict()

        feats = build_game_features(
            game_row=game,
            pitcher_stats_df=pitcher_stats_df,
            team_batting_df=team_batting_df,
            team_pitching_df=team_pitching_df,
            statcast_df=statcast_df,
            results_df=results_df,
            bullpen_log_df=bullpen_log_df,
            weather_row=weather_row,
        )
        rows.append(feats)

    return pd.DataFrame(rows)


def build_historical_feature_matrix(
    seasons: list,
    output_path: str = "data/processed/training_data.csv",
) -> pd.DataFrame:
    """
    Build full historical feature matrix for model training.
    Loads pre-saved FanGraphs and Statcast data for each season.
    """
    from data.fetch_schedule import fetch_season_schedule

    all_rows = []

    for season in seasons:
        print(f"\n── Building features for {season} ──")

        # Load pre-saved data
        try:
            fg_batting  = pd.read_csv(f"data/processed/fangraphs/{season}/team_batting.csv")
            fg_pitching = pd.read_csv(f"data/processed/fangraphs/{season}/team_pitching.csv")
            fg_pitchers = pd.read_csv(f"data/processed/fangraphs/{season}/pitchers.csv")
        except FileNotFoundError:
            print(f"  WARNING: FanGraphs data not found for {season}. Run data/fetch_fangraphs.py first.")
            continue

        try:
            statcast = pd.read_csv(f"data/processed/statcast/{season}/team_statcast.csv")
            statcast = statcast[statcast["type"] == "batting"]
        except FileNotFoundError:
            statcast = None
            print(f"  WARNING: Statcast data not found for {season}.")

        # Fetch schedule with results
        print(f"  Fetching schedule for {season}...")
        schedule = fetch_season_schedule(season)
        schedule = schedule[schedule["status"] == "Final"]

        # Build results log from schedule
        results_df = _build_results_log(schedule)

        # Build features game by game
        print(f"  Building feature matrix ({len(schedule)} games)...")
        matrix = build_feature_matrix_for_date(
            schedule_df=schedule,
            pitcher_stats_df=fg_pitchers,
            team_batting_df=fg_batting,
            team_pitching_df=fg_pitching,
            statcast_df=statcast,
            results_df=results_df,
        )

        # Attach targets
        matrix["target_home_win"]     = (schedule["home_score"] > schedule["away_score"]).astype(int).values
        matrix["target_total_runs"]   = (schedule["home_score"] + schedule["away_score"]).values
        matrix["target_home_score"]   = schedule["home_score"].values
        matrix["target_away_score"]   = schedule["away_score"].values

        all_rows.append(matrix)
        print(f"  ✓ {len(matrix)} rows built for {season}")

    if not all_rows:
        print("No data built — check your data files.")
        return pd.DataFrame()

    full_df = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_csv(output_path, index=False)
    print(f"\n✓ Training matrix saved to {output_path} ({len(full_df)} rows, {len(full_df.columns)} cols)")

    return full_df


def _build_results_log(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Build a team-level results log from a schedule DataFrame.
    Returns one row per team per game for rolling form calculation.
    """
    rows = []
    for _, game in schedule.iterrows():
        if pd.isna(game.get("home_score")) or pd.isna(game.get("away_score")):
            continue
        home_win = int(game["home_score"] > game["away_score"])
        rows.append({
            "team_abb":     game["home_team_abb"],
            "game_date":    game["game_date"],
            "is_home":      True,
            "win":          home_win,
            "runs_scored":  game["home_score"],
            "runs_allowed": game["away_score"],
        })
        rows.append({
            "team_abb":     game["away_team_abb"],
            "game_date":    game["game_date"],
            "is_home":      False,
            "win":          1 - home_win,
            "runs_scored":  game["away_score"],
            "runs_allowed": game["home_score"],
        })
    return pd.DataFrame(rows)


def get_model_feature_cols(df: pd.DataFrame) -> list:
    """
    Return only the numeric feature columns suitable for model training.
    Excludes identifier columns, targets, and non-numeric columns.
    """
    exclude = {
        "game_pk", "game_date", "home_team_abb", "away_team_abb",
        "home_pitcher", "away_pitcher", "venue",
        "target_home_win", "target_total_runs",
        "target_home_score", "target_away_score",
    }
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


if __name__ == "__main__":
    from config import TRAINING_SEASONS
    print("Building historical feature matrix...")
    df = build_historical_feature_matrix(TRAINING_SEASONS)
    if not df.empty:
        print(f"Feature columns: {get_model_feature_cols(df)[:10]}...")
