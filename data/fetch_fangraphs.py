"""
data/fetch_fangraphs.py
Pulls season-to-date and historical pitcher/team stats from FanGraphs
via the pybaseball library. Covers ERA, FIP, xFIP, K%, BB%, wOBA, etc.
"""

import pandas as pd
import numpy as np
import os
from datetime import date
from typing import Optional

try:
    import pybaseball
    from pybaseball import pitching_stats, batting_stats, team_pitching, team_batting
    pybaseball.cache.enable()
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("WARNING: pybaseball not installed. Run: pip install pybaseball")


# ── Pitcher Stats ──────────────────────────────────────────────────────────────

def fetch_pitcher_stats(season: int, min_ip: int = 10) -> pd.DataFrame:
    """
    Fetch season-to-date pitching stats from FanGraphs.
    Returns per-pitcher advanced stats: ERA, FIP, xFIP, K%, BB%, WHIP, etc.
    """
    if not PYBASEBALL_AVAILABLE:
        return _mock_pitcher_stats()

    print(f"  Fetching FanGraphs pitcher stats for {season}...")
    df = pitching_stats(season, qual=min_ip)

    # Standardize column names
    rename = {
        "Name":   "pitcher_name",
        "Team":   "team_abb",
        "ERA":    "era",
        "FIP":    "fip",
        "xFIP":   "xfip",
        "K/9":    "k_per_9",
        "BB/9":   "bb_per_9",
        "K%":     "k_pct",
        "BB%":    "bb_pct",
        "WHIP":   "whip",
        "IP":     "ip",
        "HR/9":   "hr_per_9",
        "GB%":    "gb_pct",
        "LOB%":   "lob_pct",
        "WAR":    "war",
        "IDfg":   "fangraphs_id",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["season"] = season

    # Convert K% and BB% from string "21.3 %" to float 0.213 if needed
    for pct_col in ["k_pct", "bb_pct"]:
        if pct_col in df.columns and df[pct_col].dtype == object:
            df[pct_col] = df[pct_col].str.replace(" %", "").astype(float) / 100

    keep_cols = [
        "pitcher_name", "team_abb", "season", "era", "fip", "xfip",
        "k_per_9", "bb_per_9", "k_pct", "bb_pct", "whip", "ip",
        "hr_per_9", "gb_pct", "lob_pct", "war", "fangraphs_id"
    ]
    available = [c for c in keep_cols if c in df.columns]
    return df[available].reset_index(drop=True)


def fetch_pitcher_splits(season: int) -> pd.DataFrame:
    """
    Fetch pitcher splits vs LHB and RHB.
    Returns pitcher_id, split (vL/vR), ERA, FIP, K%, wOBA.
    """
    if not PYBASEBALL_AVAILABLE:
        return pd.DataFrame()

    print(f"  Fetching pitcher splits for {season}...")
    # pybaseball doesn't have splits directly — use statcast aggregation
    # This is a placeholder; in practice you'd use Baseball Savant CSV exports
    return pd.DataFrame()


# ── Team Batting Stats ─────────────────────────────────────────────────────────

def fetch_team_batting_stats(season: int) -> pd.DataFrame:
    """
    Fetch team-level batting stats: wOBA, xwOBA, BB%, K%, ISO, etc.
    """
    if not PYBASEBALL_AVAILABLE:
        return _mock_team_batting()

    print(f"  Fetching FanGraphs team batting stats for {season}...")
    df = team_batting(season)

    rename = {
        "Team":   "team_abb",
        "wOBA":   "woba",
        "BB%":    "bb_pct",
        "K%":     "k_pct",
        "ISO":    "iso",
        "BABIP":  "babip",
        "wRC+":   "wrc_plus",
        "WAR":    "war",
        "R":      "runs",
        "HR":     "hr",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["season"] = season
    return df.reset_index(drop=True)


# ── Team Pitching Stats ────────────────────────────────────────────────────────

def fetch_team_pitching_stats(season: int) -> pd.DataFrame:
    """
    Fetch team-level pitching stats: ERA, FIP, K%, BB%, HR/9, etc.
    """
    if not PYBASEBALL_AVAILABLE:
        return _mock_team_pitching()

    print(f"  Fetching FanGraphs team pitching stats for {season}...")
    df = team_pitching(season)

    rename = {
        "Team":   "team_abb",
        "ERA":    "team_era",
        "FIP":    "team_fip",
        "K%":     "team_k_pct",
        "BB%":    "team_bb_pct",
        "HR/9":   "team_hr_per_9",
        "WHIP":   "team_whip",
        "WAR":    "team_war_pitching",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["season"] = season
    return df.reset_index(drop=True)


# ── Save / Load ────────────────────────────────────────────────────────────────

def save_fangraphs_data(season: int):
    """Fetch and save all FanGraphs data for a season."""
    out_dir = f"data/processed/fangraphs/{season}"
    os.makedirs(out_dir, exist_ok=True)

    pitcher_df = fetch_pitcher_stats(season)
    pitcher_df.to_csv(f"{out_dir}/pitchers.csv", index=False)

    batting_df = fetch_team_batting_stats(season)
    batting_df.to_csv(f"{out_dir}/team_batting.csv", index=False)

    pitching_df = fetch_team_pitching_stats(season)
    pitching_df.to_csv(f"{out_dir}/team_pitching.csv", index=False)

    print(f"Saved FanGraphs data for {season} to {out_dir}/")


def load_fangraphs_data(season: int) -> dict:
    """Load previously saved FanGraphs data."""
    base = f"data/processed/fangraphs/{season}"
    return {
        "pitchers":      pd.read_csv(f"{base}/pitchers.csv"),
        "team_batting":  pd.read_csv(f"{base}/team_batting.csv"),
        "team_pitching": pd.read_csv(f"{base}/team_pitching.csv"),
    }


# ── Mock data for testing without pybaseball ──────────────────────────────────

def _mock_pitcher_stats() -> pd.DataFrame:
    """Returns a minimal mock DataFrame for testing."""
    teams = ["NYY", "BOS", "LAD", "HOU", "ATL"] * 6
    np.random.seed(42)
    n = len(teams)
    return pd.DataFrame({
        "pitcher_name": [f"Pitcher {i}" for i in range(n)],
        "team_abb":     teams,
        "season":       [2024] * n,
        "era":          np.random.uniform(2.5, 5.5, n).round(2),
        "fip":          np.random.uniform(2.8, 5.0, n).round(2),
        "xfip":         np.random.uniform(3.0, 4.8, n).round(2),
        "k_pct":        np.random.uniform(0.15, 0.32, n).round(3),
        "bb_pct":       np.random.uniform(0.04, 0.12, n).round(3),
        "whip":         np.random.uniform(0.9, 1.5, n).round(2),
        "ip":           np.random.uniform(20, 180, n).round(1),
        "fangraphs_id": range(1000, 1000 + n),
    })


def _mock_team_batting() -> pd.DataFrame:
    from config import TEAM_IDS
    teams = list(TEAM_IDS.keys())
    np.random.seed(42)
    n = len(teams)
    return pd.DataFrame({
        "team_abb": teams,
        "season":   [2024] * n,
        "woba":     np.random.uniform(0.290, 0.360, n).round(3),
        "bb_pct":   np.random.uniform(0.06, 0.12, n).round(3),
        "k_pct":    np.random.uniform(0.18, 0.27, n).round(3),
        "iso":      np.random.uniform(0.12, 0.22, n).round(3),
        "wrc_plus": np.random.randint(80, 120, n),
        "runs":     np.random.randint(550, 850, n),
    })


def _mock_team_pitching() -> pd.DataFrame:
    from config import TEAM_IDS
    teams = list(TEAM_IDS.keys())
    np.random.seed(99)
    n = len(teams)
    return pd.DataFrame({
        "team_abb":         teams,
        "season":           [2024] * n,
        "team_era":         np.random.uniform(3.2, 5.2, n).round(2),
        "team_fip":         np.random.uniform(3.4, 5.0, n).round(2),
        "team_k_pct":       np.random.uniform(0.18, 0.28, n).round(3),
        "team_bb_pct":      np.random.uniform(0.06, 0.11, n).round(3),
        "team_hr_per_9":    np.random.uniform(0.8, 1.6, n).round(2),
        "team_whip":        np.random.uniform(1.10, 1.45, n).round(2),
    })


if __name__ == "__main__":
    for season in [2022, 2023, 2024]:
        save_fangraphs_data(season)
