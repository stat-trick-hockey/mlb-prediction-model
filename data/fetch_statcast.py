"""
data/fetch_statcast.py
Pulls Statcast metrics (xwOBA, barrel%, hard-hit%, exit velocity)
per team and pitcher via pybaseball / Baseball Savant.
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

    if df.empty:
        return pd.DataFrame()

    # Only batted balls for quality-of-contact metrics
    batted = df[df["type"] == "X"].copy()

    if as_pitcher:
        group_col = "pitcher_team"  # not directly in statcast; join via pitcher id
        # Simpler: group by home_team / away_team based on inning_topbot
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

    agg = batted.groupby(group_col).agg(
        statcast_xwoba         = ("estimated_woba_using_speedangle", "mean"),
        statcast_barrel_pct    = ("barrel", "mean"),
        statcast_hard_hit_pct  = ("launch_speed", lambda x: (x >= 95).mean()),
        statcast_avg_ev        = ("launch_speed", "mean"),
        statcast_avg_la        = ("launch_angle", "mean"),
        statcast_n_batted      = ("launch_speed", "count"),
    ).reset_index()
    agg.columns.name = None
    agg = agg.rename(columns={group_col: "team_abb"})

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

    if df.empty:
        return {}

    batted = df[df["type"] == "X"]
    if batted.empty:
        return {}

    return {
        "p_statcast_xwoba":        batted["estimated_woba_using_speedangle"].mean(),
        "p_statcast_barrel_pct":   batted["barrel"].mean() if "barrel" in batted else np.nan,
        "p_statcast_hard_hit_pct": (batted["launch_speed"] >= 95).mean(),
        "p_statcast_avg_ev":       batted["launch_speed"].mean(),
    }


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

            batting_agg["period_start"]  = chunk_start_str
            pitching_agg["period_start"] = chunk_start_str
            batting_agg["type"]  = "batting"
            pitching_agg["type"] = "pitching"

            chunks.append(batting_agg)
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
    # Test with a small date range
    print("Testing Statcast fetch...")
    df = fetch_team_statcast("2024-04-01", "2024-04-07", as_pitcher=False)
    print(df.head())
