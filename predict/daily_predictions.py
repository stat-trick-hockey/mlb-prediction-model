"""
predict/daily_predictions.py
Main daily runner. Fetches today's slate, builds features,
runs all three models, and outputs predictions with edge calculations.
Run this each morning after probable pitchers are confirmed (10am ET).
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import date, datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_schedule import fetch_schedule
from data.fetch_odds     import fetch_mlb_odds, save_odds
from data.fetch_weather  import fetch_weather_for_slate

from features.build_feature_matrix import build_feature_matrix_for_date, get_model_feature_cols

from models.train_moneyline import predict_moneyline
from models.train_runline   import predict_runline
from models.train_ou        import predict_ou

from predict.edge_calculator import calculate_game_edges, format_edge_report


def run_daily_predictions(
    game_date: str = None,
    output_dir: str = "predict/output",
) -> list:
    """
    Full daily prediction pipeline.

    1. Fetch schedule + probable pitchers
    2. Fetch Vegas odds
    3. Fetch weather
    4. Load FanGraphs + Statcast data
    5. Build feature matrix
    6. Run all three models
    7. Calculate edge vs. Vegas
    8. Save and print report

    Returns list of game result dicts.
    """
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    # ── Season gate ───────────────────────────────────────────────────────────
    from utils.season_gate import should_run_predictions
    should_run, reason = should_run_predictions(date.fromisoformat(game_date))
    if not should_run:
        print(f"\n⚾  {reason}")
        print(f"   Next regular season starts around late March.")
        print(f"   Skipping API calls to preserve rate limits.\n")
        return []

    print(f"\n{'═'*60}")
    print(f"  MLB Predictions — {game_date}")
    print(f"{'═'*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"data/raw/{game_date}", exist_ok=True)

    # ── Step 1: Schedule ──────────────────────────────────────────────────────
    print("1. Fetching schedule...")
    schedule = fetch_schedule(game_date)
    schedule = schedule[schedule["status"].isin(["Preview", "Pre-Game", None])]
    if schedule.empty:
        print(f"  No upcoming games found for {game_date}.")
        return []
    print(f"  Found {len(schedule)} games: {', '.join(schedule['home_team_abb'].dropna().tolist())}")
    schedule.to_csv(f"data/raw/{game_date}/schedule.csv", index=False)

    # ── Step 2: Odds ──────────────────────────────────────────────────────────
    print("\n2. Fetching odds...")
    odds_df = fetch_mlb_odds(game_date)
    save_odds(odds_df, game_date)

    # ── Step 3: Weather ───────────────────────────────────────────────────────
    print("\n3. Fetching weather...")
    schedule_with_weather = fetch_weather_for_slate(schedule)

    # ── Step 4: Load pre-built data ───────────────────────────────────────────
    print("\n4. Loading FanGraphs/Statcast data...")
    season = pd.to_datetime(game_date).year
    try:
        fg_batting   = pd.read_csv(f"data/processed/fangraphs/{season}/team_batting.csv")
        fg_pitching  = pd.read_csv(f"data/processed/fangraphs/{season}/team_pitching.csv")
        fg_pitchers  = pd.read_csv(f"data/processed/fangraphs/{season}/pitchers.csv")
        print(f"  Loaded FanGraphs data for {season}")
    except FileNotFoundError:
        print(f"  WARNING: No FanGraphs data for {season}. Using prior season or defaults.")
        try:
            fg_batting   = pd.read_csv(f"data/processed/fangraphs/{season-1}/team_batting.csv")
            fg_pitching  = pd.read_csv(f"data/processed/fangraphs/{season-1}/team_pitching.csv")
            fg_pitchers  = pd.read_csv(f"data/processed/fangraphs/{season-1}/pitchers.csv")
        except FileNotFoundError:
            fg_batting = fg_pitching = fg_pitchers = pd.DataFrame()

    try:
        statcast = pd.read_csv(f"data/processed/statcast/{season}/team_statcast.csv")
        statcast = statcast[statcast["type"] == "batting"]
    except FileNotFoundError:
        statcast = None

    # ── Step 5: Build feature matrix ──────────────────────────────────────────
    print("\n5. Building feature matrix...")
    feature_matrix = build_feature_matrix_for_date(
        schedule_df    = schedule_with_weather,
        pitcher_stats_df = fg_pitchers,
        team_batting_df  = fg_batting,
        team_pitching_df = fg_pitching,
        statcast_df      = statcast,
        weather_df       = schedule_with_weather,
    )
    print(f"  Built features for {len(feature_matrix)} games")

    # ── Step 6 + 7: Run models and calculate edge ─────────────────────────────
    print("\n6. Running models...")
    results = []

    # Merge odds into feature matrix by team name
    if not odds_df.empty:
        feature_matrix = feature_matrix.merge(
            odds_df,
            left_on=["home_team_abb", "away_team_abb"],
            right_on=["home_team", "away_team"],
            how="left",
        )

    for i, row in feature_matrix.iterrows():
        X = pd.DataFrame([row])
        game_info = row.to_dict()

        # Run models
        try:
            ml  = predict_moneyline(X)
            ou  = predict_ou(X, ou_line=row.get("ou_total", 9.0))
            rl  = predict_runline(X)
        except Exception as e:
            print(f"  WARNING: Model error for {row.get('home_team_abb')}: {e}")
            ml  = {"home_win_prob": 0.54, "away_win_prob": 0.46}
            ou  = {"predicted_total": 9.0, "over_prob": 0.50, "under_prob": 0.50}
            rl  = {"home_covers_prob": 0.40, "away_covers_prob": 0.60}

        # Calculate edge
        edge = calculate_game_edges(
            game          = game_info,
            home_win_prob = ml["home_win_prob"],
            predicted_total = ou["predicted_total"],
            over_prob     = ou["over_prob"],
            home_covers_prob = rl["home_covers_prob"],
        )
        results.append({**game_info, **ml, **ou, **rl, **edge})

    # ── Step 8: Output ────────────────────────────────────────────────────────
    print("\n7. Saving output...")
    results_df = pd.DataFrame(results)
    output_csv = f"{output_dir}/{game_date}_predictions.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"  Saved to {output_csv}")

    # JSON output for downstream use (Instagram cards, Tidbyt, etc.)
    output_json = f"{output_dir}/{game_date}_predictions.json"
    slim_results = _slim_results_for_json(results)
    with open(output_json, "w") as f:
        json.dump(slim_results, f, indent=2, default=str)
    print(f"  Saved JSON to {output_json}")

    # Print report
    report = format_edge_report(results)
    print("\n" + report)

    report_path = f"{output_dir}/{game_date}_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    return results


def _slim_results_for_json(results: list) -> list:
    """Reduce results to key fields for JSON output."""
    slim = []
    keep = [
        "home_team_abb", "away_team_abb", "home_pitcher", "away_pitcher",
        "home_win_prob", "away_win_prob", "predicted_total",
        "over_prob", "under_prob", "ou_total",
        "home_covers_prob",
        "ml_home_odds", "ml_away_odds",
        "ml_home_edge", "ml_away_edge",
        "ou_over_edge", "ou_under_edge",
        "rl_home_edge",
        "ml_home_flag", "ml_away_flag",
        "ou_over_flag", "ou_under_flag",
        "rl_home_flag", "any_edge",
        "temp_f", "wind_mph", "wind_out_to_cf",
    ]
    for r in results:
        slim.append({k: r.get(k) for k in keep})
    return slim


def load_predictions(game_date: str, output_dir: str = "predict/output") -> pd.DataFrame:
    """Load saved predictions for a given date."""
    path = f"{output_dir}/{game_date}_predictions.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No predictions found for {game_date}")
    return pd.read_csv(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run daily MLB predictions")
    parser.add_argument("--date", type=str, default=None, help="Game date YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    run_daily_predictions(game_date=args.date)
