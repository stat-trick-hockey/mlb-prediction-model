"""
data/fetch_statcast.py
Pulls Statcast metrics (xwOBA, barrel%, hard-hit%, exit velocity)
per team and pitcher via pybaseball / Baseball Savant.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import date, timedelta
from typing import Optional

try:
    from pybaseball import statcast, statcast_pitcher
    import pybaseball
    pybaseball.cache.enable()
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("WARNING: pybaseball not installed — using mock data.")

try:
    from data.barrel_calc import ensure_barrel_column
except ModuleNotFoundError:
    from barrel_calc import ensure_barrel_column


# ── Team-level Statcast aggregation ───────────────────────────────────────────

def _fetch_raw_statcast(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch raw Statcast data for a date range and return batted balls only.
    Separated so save_season_statcast can call it once and aggregate both ways.
    """
    if not PYBASEBALL_AVAILABLE:
        return pd.DataFrame()

    print(f"    statcast({start_date} → {end_date})...", end=" ", flush=True)
    df = statcast(start_dt=start_date, end_dt=end_date, verbose=False)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        print("0 rows")
        return pd.DataFrame()

    print(f"{len(df)} rows")

    if "type" not in df.columns:
        print(f"    WARNING: 'type' column missing. Available: {list(df.columns[:10])}")
        return pd.DataFrame()

    df = ensure_barrel_column(df)
    batted = df[df["type"] == "X"].copy()

    if batted.empty:
        print(f"    WARNING: 0 batted balls (type=='X') in this range")
        return pd.DataFrame()

    required = {"inning_topbot", "home_team", "away_team"}
    missing = required - set(batted.columns)
    if missing:
        print(f"    WARNING: missing columns {missing}")
        return pd.DataFrame()

    print(f"    {len(batted)} batted balls")
    return batted


def _aggregate_batted(batted: pd.DataFrame, as_pitcher: bool) -> pd.DataFrame:
    """Aggregate batted ball DataFrame by team."""
    if batted.empty:
        return pd.DataFrame()

    if as_pitcher:
        batted = batted.copy()
        batted["team_col"] = np.where(
            batted["inning_topbot"] == "Top",
            batted["home_team"],
            batted["away_team"]
        )
    else:
        batted = batted.copy()
        batted["team_col"] = np.where(
            batted["inning_topbot"] == "Top",
            batted["away_team"],
            batted["home_team"]
        )

    agg_dict = {}
    if "estimated_woba_using_speedangle" in batted.columns:
        agg_dict["statcast_xwoba"] = ("estimated_woba_using_speedangle", "mean")
    if "barrel" in batted.columns:
        agg_dict["statcast_barrel_pct"] = ("barrel", "mean")
    if "launch_speed" in batted.columns:
        agg_dict["statcast_hard_hit_pct"] = ("launch_speed", lambda x: (x >= 95).mean())
        agg_dict["statcast_avg_ev"]        = ("launch_speed", "mean")
        agg_dict["statcast_n_batted"]      = ("launch_speed", "count")
    if "launch_angle" in batted.columns:
        agg_dict["statcast_avg_la"] = ("launch_angle", "mean")

    if not agg_dict:
        return pd.DataFrame()

    agg = batted.groupby("team_col").agg(**agg_dict).reset_index()
    agg = agg.rename(columns={"team_col": "team_abb"})

    for col in ["statcast_xwoba", "statcast_barrel_pct", "statcast_hard_hit_pct",
                "statcast_avg_ev", "statcast_avg_la", "statcast_n_batted"]:
        if col not in agg.columns:
            agg[col] = np.nan

    return agg


def fetch_team_statcast(
    start_date: str,
    end_date: str,
    as_pitcher: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pull raw Statcast data for a date range and aggregate by team.
    Note: for bulk season fetching, use save_season_statcast() which fetches
    each chunk once and aggregates both batting and pitching from the same pull.
    """
    if not PYBASEBALL_AVAILABLE:
        return _mock_team_statcast()

    batted = _fetch_raw_statcast(start_date, end_date)
    return _aggregate_batted(batted, as_pitcher)


def fetch_pitcher_statcast(pitcher_id: int, start_date: str, end_date: str) -> dict:
    if not PYBASEBALL_AVAILABLE:
        return _mock_pitcher_statcast()

    df = statcast_pitcher(start_dt=start_date, end_dt=end_date, player_id=pitcher_id)
    if df is None or df.empty:
        return {}

    df = ensure_barrel_column(df)
    batted = df[df["type"] == "X"] if "type" in df.columns else df
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


def fetch_rolling_team_statcast(team_abb: str, reference_date: str, days: int = 30) -> dict:
    end   = pd.to_datetime(reference_date)
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    df    = fetch_team_statcast(start, end.strftime("%Y-%m-%d"), verbose=False)
    if df.empty:
        return {}
    row = df[df["team_abb"] == team_abb]
    return row.iloc[0].to_dict() if not row.empty else {}


# ── Full season save ──────────────────────────────────────────────────────────

def save_season_statcast(season: int, chunk_days: int = 30) -> bool:
    """
    Download and save full-season Statcast data in monthly chunks.
    Returns True if successful, False if no data was saved.
    Prints chunk-level diagnostics to stdout.
    """
    out_dir = f"data/processed/statcast/{season}"
    os.makedirs(out_dir, exist_ok=True)

    season_start = f"{season}-03-20"
    season_end   = f"{season}-10-05"

    start   = pd.to_datetime(season_start)
    end     = pd.to_datetime(season_end)
    chunks  = []
    current = start
    n_chunks = 0

    print(f"\n── Fetching Statcast for {season} ({season_start} → {season_end}) ──")

    while current < end:
        chunk_end       = min(current + timedelta(days=chunk_days), end)
        cs              = current.strftime("%Y-%m-%d")
        ce              = chunk_end.strftime("%Y-%m-%d")
        n_chunks       += 1

        print(f"  [{n_chunks}] {cs} → {ce}")
        try:
            # Fetch raw data ONCE, aggregate two ways — avoids double download
            batted = _fetch_raw_statcast(cs, ce)

            if batted.empty:
                print(f"    ✗ no batted ball data for this chunk")
            else:
                batting_agg  = _aggregate_batted(batted, as_pitcher=False)
                pitching_agg = _aggregate_batted(batted, as_pitcher=True)

                if not batting_agg.empty:
                    batting_agg["period_start"] = cs
                    batting_agg["type"]         = "batting"
                    chunks.append(batting_agg)
                    print(f"    ✓ batting: {len(batting_agg)} teams")
                else:
                    print(f"    ✗ batting: empty after aggregation")

                if not pitching_agg.empty:
                    pitching_agg["period_start"] = cs
                    pitching_agg["type"]         = "pitching"
                    chunks.append(pitching_agg)
                    print(f"    ✓ pitching: {len(pitching_agg)} teams")
                else:
                    print(f"    ✗ pitching: empty after aggregation")

        except Exception as e:
            import traceback
            print(f"    ERROR in chunk {cs} → {ce}:")
            traceback.print_exc()

        current = chunk_end + timedelta(days=1)

    if chunks:
        df   = pd.concat(chunks, ignore_index=True)
        path = f"{out_dir}/team_statcast.csv"
        df.to_csv(path, index=False)
        print(f"\n✓ {season}: saved {len(df)} rows to {path}")
        return True
    else:
        print(f"\n✗ {season}: NO data saved after {n_chunks} chunks")
        return False


# ── Mock data ─────────────────────────────────────────────────────────────────

def _mock_team_statcast() -> pd.DataFrame:
    try:
        from config import TEAM_IDS
        teams = list(TEAM_IDS.keys())
    except Exception:
        teams = ["NYY", "BOS", "LAD", "HOU", "ATL"]
    np.random.seed(7)
    n = len(teams)
    return pd.DataFrame({
        "team_abb":              teams,
        "statcast_xwoba":        np.random.uniform(0.290, 0.370, n).round(3),
        "statcast_barrel_pct":   np.random.uniform(0.05, 0.12, n).round(3),
        "statcast_hard_hit_pct": np.random.uniform(0.35, 0.50, n).round(3),
        "statcast_avg_ev":       np.random.uniform(86, 92, n).round(1),
        "statcast_avg_la":       np.random.uniform(8, 16, n).round(1),
        "statcast_n_batted":     np.random.randint(500, 3000, n),
    })


def _mock_pitcher_statcast() -> dict:
    np.random.seed(13)
    return {
        "p_statcast_xwoba":        round(np.random.uniform(0.28, 0.38), 3),
        "p_statcast_barrel_pct":   round(np.random.uniform(0.04, 0.12), 3),
        "p_statcast_hard_hit_pct": round(np.random.uniform(0.30, 0.50), 3),
        "p_statcast_avg_ev":       round(np.random.uniform(85, 93), 1),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2022,2023,2024")
    parser.add_argument("--chunk-days", type=int, default=30)
    parser.add_argument("--test", action="store_true",
                        help="Fetch one week only to verify connectivity")
    args = parser.parse_args()

    if args.test:
        print("Quick test: fetching 2024-04-01 → 2024-04-07...")
        df = fetch_team_statcast("2024-04-01", "2024-04-07", as_pitcher=False, verbose=True)
        if df.empty:
            print("FAIL: returned empty DataFrame")
            sys.exit(1)
        print(df)
        print("PASS")
        sys.exit(0)

    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    print(f"Fetching Statcast for seasons: {seasons}")
    print("NOTE: expect ~20-40 min per season (30-day chunks x 7 per season)")

    failed = []
    for season in seasons:
        ok = save_season_statcast(season, chunk_days=args.chunk_days)
        if not ok:
            failed.append(season)

    if failed:
        print("\nFAILED seasons: " + str(failed))
        sys.exit(1)

    print("\n✓ All seasons fetched successfully.")
