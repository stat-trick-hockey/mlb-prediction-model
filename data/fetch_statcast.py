"""
data/fetch_statcast.py
Pulls Statcast metrics (xwOBA, barrel%, hard-hit%, exit velocity)
per team and pitcher via pybaseball / Baseball Savant.

Barrel is computed from launch_speed + launch_angle if the column
is missing from the raw pull (common in some date ranges).
"""

import pandas as pd
import numpy as np
import os
from datetime import date, timedelta
from typing import Optional

try:
    from pybaseball import statcast, statcast_pitcher, statcast_batter
    from pybaseball import playerid_lookup
    import pybaseball
    pybaseball.cache.enable()
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("WARNING: pybaseball not installed.")

try:
    from data.barrel_calc import ensure_barrel_column
except ModuleNotFoundError:
    from barrel_calc import ensure_barrel_column


# ── Team-level Statcast aggregation ───────────────────────────────────────────

def fetch_team_statcast(
    start_date: str,
    end_date: str,
    as_pitcher: bool = False
) -> pd.DataFrame:
    """
    Pull Statcast pitch/batted-ball data for a date range and aggregate by team.

    as_pitcher=True  → aggregate stats allowed by each team's pitchers
    as_pitcher=False → aggregate stats produced by each team's batters
    """
    if not PYBASEBALL_AVAILABLE:
        return _mock_team_statcast()

    print(f"  Fetching Statcast data {start_date} → {end_date}...")
    df = statcast(start_dt=start_date, end_dt=end_date)

    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure barrel column exists (compute if missing)
    df = ensure_barrel_column(df)

    # Only batted balls for quality-of-contact metrics
    batted = df[df["type"] == "X"].copy()
    if batted.empty:
        return pd.DataFrame()

    if as_pitcher:
        batted["fielding_team"] = np.where(
            batted["inning_topbot"] == "Top",
            batted["home_team"],
            batted["away_team"]
        )
        group_col = "fielding_team"
    else:
        batted["batting_team"] = np.where(
            batted["inning_topbot"] == "Top",
            batted["away_team"],
            batted["home_team"]
        )
        group_col = "batting_team"

    # Build aggregation dict — only include columns that exist
    agg_dict = {}

    if "estimated_woba_using_speedangle" in batted.columns:
        agg_dict["statcast_xwoba"] = ("estimated_woba_using_speedangle", "mean")

    # barrel is now guaranteed by ensure_barrel_column
    if "barrel" in batted.columns:
        agg_dict["statcast_barrel_pct"] = ("barrel", "mean")

    if "launch_speed" in batted.columns:
        agg_dict["statcast_hard_hit_pct"] = ("launch_speed", lambda x: (x >= 95).mean())
        agg_dict["statcast_avg_ev"]        = ("launch_speed", "mean")
        agg_dict["statcast_n_batted"]      = ("launch_speed", "count")

    if "launch_angle" in batted.columns:
        agg_dict["statcast_avg_la"] = ("launch_angle", "mean")

    if not agg_dict:
        print("  WARNING: No usable Statcast columns found in this date range")
        return pd.DataFrame()

    agg = batted.groupby(group_col).agg(**agg_dict).reset_index()
    agg.columns.name = None
    agg = agg.rename(columns={group_col: "team_abb"})

    # Guarantee all expected output columns exist
    for col in ["statcast_xwoba", "statcast_barrel_pct", "statcast_hard_hit_pct",
                "statcast_avg_ev", "statcast_avg_la", "statcast_n_batted"]:
        if col not in agg.columns:
            agg[col] = np.nan

    return agg


def fetch_pitcher_statcast(
    pitcher_id: int,
    start_date: str,
    end_date: str
) -> dict:
    """
    Pull Statcast metrics for a specific pitcher over a date range.
    Returns a dict of aggregated quality-of-contact metrics allowed.
    """
    if not PYBASEBALL_AVAILABLE:
        return _mock_pitcher_statcast()

    df = statcast_pitcher(start_dt=start_date, end_dt=end_date, player_id=pitcher_id)

    if df is None or df.empty:
        return {}

    df = ensure_barrel_column(df)

    batted = df[df["type"] == "X"]
    if batted.empty:
        return {}

    result = {}
    if "estimated_woba_using_speedangle" in batted.columns:
        result["p_statcast_xwoba"] = batted["estimated_woba_using_speedangle"].mean()
    if "barrel" in batted.columns:
        result["p_statcast_barrel_pct"] = batted["barrel"].mean()
    if "launch_speed" in batted.columns:
        result["p_statcast_hard_hit_pct"] = (batted["launch_speed"] >= 95).mean()
        result["p_statcast_avg_ev"]        = batted["launch_speed"].mean()

    return result


def fetch_rolling_team_statcast(
    team_abb: str,
    reference_date: str,
    days: int = 30
) -> dict:
    """
    Convenience wrapper: fetch rolling Statcast for a team over last N days.
    """
    end = pd.to_datetime(reference_date)
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    df = fetch_team_statcast(start, end_str, as_pitcher=False)
    if df.empty:
        return {}
    row = df[df["team_abb"] == team_abb]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


# ── Save / Load ────────────────────────────────────────────────────────────────

def save_season_statcast(season: int, chunk_days: int = 30):
    """
    Download and save Statcast data for a full season in monthly chunks.
    Statcast has rate limits so chunking prevents timeouts.
    """
    out_dir = f"data/processed/statcast/{season}"
    os.makedirs(out_dir, exist_ok=True)

    season_start = f"{season}-03-20"
    season_end   = f"{season}-10-05"

    start = pd.to_datetime(season_start)
    end   = pd.to_datetime(season_end)
    chunks = []

    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunk_start_str = current.strftime("%Y-%m-%d")
        chunk_end_str   = chunk_end.strftime("%Y-%m-%d")

        print(f"  Chunk: {chunk_start_str} → {chunk_end_str}")
        try:
            batting_agg  = fetch_team_statcast(chunk_start_str, chunk_end_str, as_pitcher=False)
            pitching_agg = fetch_team_statcast(chunk_start_str, chunk_end_str, as_pitcher=True)

            if not batting_agg.empty:
                batting_agg["period_start"] = chunk_start_str
                batting_agg["type"] = "batting"
                chunks.append(batting_agg)

            if not pitching_agg.empty:
                pitching_agg["period_start"] = chunk_start_str
                pitching_agg["type"] = "pitching"
                chunks.append(pitching_agg)

        except Exception as e:
            print(f"    WARNING: chunk failed: {e}")

        current = chunk_end + timedelta(days=1)

    if chunks:
        df = pd.concat(chunks, ignore_index=True)
        path = f"{out_dir}/team_statcast.csv"
        df.to_csv(path, index=False)
        print(f"Saved Statcast data for {season} → {path}")
    else:
        print(f"No Statcast data saved for {season}")


# ── Mock data ─────────────────────────────────────────────────────────────────

def _mock_team_statcast() -> pd.DataFrame:
    from config import TEAM_IDS
    teams = list(TEAM_IDS.keys())
    np.random.seed(7)
    n = len(teams)
    return pd.DataFrame({
        "team_abb":               teams,
        "statcast_xwoba":         np.random.uniform(0.290, 0.370, n).round(3),
        "statcast_barrel_pct":    np.random.uniform(0.05, 0.12, n).round(3),
        "statcast_hard_hit_pct":  np.random.uniform(0.35, 0.50, n).round(3),
        "statcast_avg_ev":        np.random.uniform(86, 92, n).round(1),
        "statcast_avg_la":        np.random.uniform(8, 16, n).round(1),
        "statcast_n_batted":      np.random.randint(500, 3000, n),
    })


def _mock_pitcher_statcast() -> dict:
    np.random.seed(13)
    return {
        "p_statcast_xwoba":        round(np.random.uniform(0.28, 0.38), 3),
        "p_statcast_barrel_pct":   round(np.random.uniform(0.04, 0.12), 3),
        "p_statcast_hard_hit_pct": round(np.random.uniform(0.30, 0.50), 3),
        "p_statcast_avg_ev":       round(np.random.uniform(85, 93), 1),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2022,2023,2024",
                        help="Comma-separated seasons to fetch")
    parser.add_argument("--chunk-days", type=int, default=30)
    parser.add_argument("--test", action="store_true",
                        help="Quick test: fetch one week of data only")
    args = parser.parse_args()

    if args.test:
        print("Running quick test fetch (2024-04-01 → 2024-04-07)...")
        df = fetch_team_statcast("2024-04-01", "2024-04-07", as_pitcher=False)
        print(df.head())
        import sys; sys.exit(0)

    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    print(f"Fetching Statcast data for seasons: {seasons}")
    print(f"WARNING: This will take 45-90 minutes for 3 full seasons.")
    failed = []
    for season in seasons:
        try:
            save_season_statcast(season, chunk_days=args.chunk_days)
            # Verify output was actually written
            import os
            path = f"data/processed/statcast/{season}/team_statcast.csv"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Output file missing after fetch: {path}")
            rows = pd.read_csv(path)
            print(f"  ✓ Verified: {len(rows)} rows in {path}")
        except Exception as e:
            print(f"  ERROR fetching Statcast for {season}: {e}")
            failed.append(season)

    if failed:
        print(f"
FAILED seasons: {failed}")
        import sys; sys.exit(1)
    print("
✓ All Statcast data fetched successfully.")
