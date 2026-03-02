"""
features/bullpen_features.py
Builds bullpen features: 7-day ERA/FIP, pitcher fatigue flags, closer availability.
Uses game log data from the MLB Stats API game feed.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def build_bullpen_features(
    team_abb: str,
    game_date: str,
    bullpen_log_df: Optional[pd.DataFrame] = None,
    side: str = "home"
) -> dict:
    """
    Build bullpen features for one team.

    bullpen_log_df expected cols:
        team_abb, pitcher_id, pitcher_name, game_date, ip, era, fip,
        is_closer, pitches_thrown

    Returns dict prefixed with f"{side}_bp_"
    """
    prefix = f"{side}_bp_"

    if bullpen_log_df is None or bullpen_log_df.empty:
        return _default_bullpen_features(prefix)

    ref_date = pd.to_datetime(game_date)
    team_log = bullpen_log_df[
        (bullpen_log_df["team_abb"] == team_abb) &
        (pd.to_datetime(bullpen_log_df["game_date"]) < ref_date)
    ].copy()

    if team_log.empty:
        return _default_bullpen_features(prefix)

    # ── 7-day bullpen ERA/FIP ─────────────────────────────────────────────────
    seven_days_ago = ref_date - timedelta(days=7)
    recent = team_log[pd.to_datetime(team_log["game_date"]) >= seven_days_ago]

    bp_era_7d = recent["era"].mean() if not recent.empty and "era" in recent.columns else 4.2
    bp_fip_7d = recent["fip"].mean() if not recent.empty and "fip" in recent.columns else 4.1
    bp_ip_7d  = recent["ip"].sum()   if not recent.empty and "ip" in recent.columns else 0

    # ── Fatigue flags ─────────────────────────────────────────────────────────
    fatigued_1d = _count_fatigued(team_log, ref_date, days_back=1)
    fatigued_2d = _count_fatigued(team_log, ref_date, days_back=2)
    fatigued_3d = _count_fatigued(team_log, ref_date, days_back=3)

    # ── Closer availability ───────────────────────────────────────────────────
    closer_available = _check_closer_available(team_log, ref_date)

    # ── High-leverage reliever usage ──────────────────────────────────────────
    hl_usage_pct = _high_leverage_usage(team_log, ref_date)

    return {
        f"{prefix}era_7d":          round(bp_era_7d, 3),
        f"{prefix}fip_7d":          round(bp_fip_7d, 3),
        f"{prefix}ip_7d":           round(bp_ip_7d, 1),
        f"{prefix}fatigued_1d":     fatigued_1d,
        f"{prefix}fatigued_2d":     fatigued_2d,
        f"{prefix}fatigued_3d":     fatigued_3d,
        f"{prefix}closer_available":int(closer_available),
        f"{prefix}hl_usage_pct":    round(hl_usage_pct, 3),
    }


def _count_fatigued(log: pd.DataFrame, ref_date: pd.Timestamp, days_back: int) -> int:
    """Count distinct pitchers who appeared in the last N days."""
    cutoff = ref_date - timedelta(days=days_back)
    recent = log[pd.to_datetime(log["game_date"]) >= cutoff]
    return recent["pitcher_id"].nunique() if "pitcher_id" in recent.columns else 0


def _check_closer_available(log: pd.DataFrame, ref_date: pd.Timestamp) -> bool:
    """Return True if closer did NOT pitch in last 2 days."""
    if "is_closer" not in log.columns:
        return True  # assume available if no data

    two_days_ago = ref_date - timedelta(days=2)
    recent_closer = log[
        (pd.to_datetime(log["game_date"]) >= two_days_ago) &
        (log["is_closer"] == True)
    ]
    return recent_closer.empty


def _high_leverage_usage(log: pd.DataFrame, ref_date: pd.Timestamp, days: int = 7) -> float:
    """
    Fraction of appearances by high-leverage relievers in last N days.
    High-leverage defined as: top 3 relievers by usage (proxy for role).
    """
    cutoff = ref_date - timedelta(days=days)
    recent = log[pd.to_datetime(log["game_date"]) >= cutoff]

    if recent.empty or "pitcher_id" not in recent.columns:
        return 0.5

    # Top 3 most-used pitchers = high-leverage proxy
    top_pitchers = recent["pitcher_id"].value_counts().head(3).index
    hl_apps = recent[recent["pitcher_id"].isin(top_pitchers)]
    return len(hl_apps) / max(len(recent), 1)


def _default_bullpen_features(prefix: str) -> dict:
    """League-average bullpen features."""
    return {
        f"{prefix}era_7d":          4.15,
        f"{prefix}fip_7d":          4.10,
        f"{prefix}ip_7d":           15.0,
        f"{prefix}fatigued_1d":     3,
        f"{prefix}fatigued_2d":     5,
        f"{prefix}fatigued_3d":     7,
        f"{prefix}closer_available":1,
        f"{prefix}hl_usage_pct":    0.50,
    }


def fetch_bullpen_log_from_api(team_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch bullpen game log from MLB Stats API for a given team and date range.
    Returns rows with pitcher_id, game_date, ip, era, etc.
    NOTE: This is a simplified version; for production, use the boxscore endpoint.
    """
    url = f"{MLB_API_BASE}/teams/{team_id}/stats"
    params = {
        "stats":  "gameLog",
        "group":  "pitching",
        "season": pd.to_datetime(start_date).year,
        "startDate": start_date,
        "endDate":   end_date,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for split in data.get("stats", [{}])[0].get("splits", []):
            player = split.get("player", {})
            stat   = split.get("stat", {})
            rows.append({
                "pitcher_id":    player.get("id"),
                "pitcher_name":  player.get("fullName"),
                "game_date":     split.get("date"),
                "ip":            _parse_ip(stat.get("inningsPitched", "0.0")),
                "era":           float(stat.get("era", 4.2)),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  Bullpen log fetch failed for team {team_id}: {e}")
        return pd.DataFrame()


def _parse_ip(ip_str: str) -> float:
    """Convert '6.1' innings pitched format to decimal (6.333...)."""
    try:
        parts = str(ip_str).split(".")
        full  = int(parts[0])
        third = int(parts[1]) if len(parts) > 1 else 0
        return full + third / 3.0
    except:
        return 0.0
