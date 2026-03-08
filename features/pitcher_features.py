"""
features/pitcher_features.py
Builds per-game pitcher features for the starting pitcher on each side.
Combines season stats (FanGraphs) with rolling 5-start window and rest days.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def build_pitcher_features(
    game_row: pd.Series,
    pitcher_stats_df: pd.DataFrame,
    game_log_df: Optional[pd.DataFrame] = None,
    side: str = "home"  # "home" or "away"
) -> dict:
    """
    Build pitcher features for one side of a game.

    Args:
        game_row: single row from schedule DataFrame
        pitcher_stats_df: FanGraphs season stats (all pitchers)
        game_log_df: historical game log for rolling stats (optional)
        side: "home" or "away"

    Returns:
        dict of features prefixed with f"{side}_sp_"
    """
    prefix = f"{side}_sp_"
    pitcher_name = game_row.get(f"{side}_pitcher_name")
    pitcher_id   = game_row.get(f"{side}_pitcher_id")
    game_date    = pd.to_datetime(game_row.get("game_date", datetime.today()))

    # Default features (used when pitcher is TBD or not found)
    defaults = _default_pitcher_features(prefix)

    # Guard against NaN (float) or empty pitcher name
    if not pitcher_name or not isinstance(pitcher_name, str) or pitcher_name.strip() == "":
        return defaults

    # Match pitcher in FanGraphs stats
    sp_row = _match_pitcher(pitcher_name, pitcher_stats_df)
    if sp_row is None:
        return defaults

    features = {
        f"{prefix}era":   sp_row.get("era", np.nan),
        f"{prefix}fip":   sp_row.get("fip", np.nan),
        f"{prefix}xfip":  sp_row.get("xfip", np.nan),
        f"{prefix}k_pct": sp_row.get("k_pct", np.nan),
        f"{prefix}bb_pct":sp_row.get("bb_pct", np.nan),
        f"{prefix}whip":  sp_row.get("whip", np.nan),
        f"{prefix}ip":    sp_row.get("ip", np.nan),
        f"{prefix}war":   sp_row.get("war", np.nan),
    }

    # Rolling 5-start stats (if game log available)
    if game_log_df is not None and not game_log_df.empty:
        rolling = _rolling_pitcher_stats(pitcher_id, game_date, game_log_df)
        features.update({
            f"{prefix}rolling_era":  rolling.get("rolling_era", features[f"{prefix}era"]),
            f"{prefix}rolling_fip":  rolling.get("rolling_fip", features[f"{prefix}fip"]),
            f"{prefix}rolling_k_pct":rolling.get("rolling_k_pct", features[f"{prefix}k_pct"]),
            f"{prefix}days_rest":    rolling.get("days_rest", 5),
        })
    else:
        features[f"{prefix}rolling_era"]   = features[f"{prefix}era"]
        features[f"{prefix}rolling_fip"]   = features[f"{prefix}fip"]
        features[f"{prefix}rolling_k_pct"] = features[f"{prefix}k_pct"]
        features[f"{prefix}days_rest"]     = 5  # assume normal rest

    return features


def _normalize_name(name: str) -> str:
    """Normalize a pitcher name for matching: lowercase, remove accents, strip suffixes."""
    import unicodedata
    # Remove accents (e.g. é → e, ñ → n)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    # Lowercase and strip whitespace
    name = name.lower().strip()
    # Remove common suffixes that cause mismatches
    for suffix in [" jr.", " sr.", " jr", " sr", " ii", " iii"]:
        name = name.replace(suffix, "")
    # Normalize hyphens and apostrophes
    name = name.replace("-", " ").replace("'", "")
    return name.strip()


def _match_pitcher(name: str, stats_df: pd.DataFrame) -> Optional[dict]:
    """Fuzzy-match a pitcher name to the stats DataFrame."""
    if stats_df.empty:
        return None

    # Guard: name must be a non-empty string
    if not name or not isinstance(name, str) or name.strip() == "":
        return None

    name = name.strip()

    # Build normalized name column once
    if "_name_normalized" not in stats_df.columns:
        stats_df = stats_df.copy()
        stats_df["_name_normalized"] = stats_df["pitcher_name"].apply(_normalize_name)

    name_norm = _normalize_name(name)

    # 1. Exact normalized match
    match = stats_df[stats_df["_name_normalized"] == name_norm]
    if not match.empty:
        return match.iloc[0].to_dict()

    # 2. Last-name normalized match
    last_name = name_norm.split()[-1]
    match = stats_df[stats_df["_name_normalized"].str.endswith(last_name)]
    if len(match) == 1:
        return match.iloc[0].to_dict()

    # 3. First initial + last name match (e.g. "J. Verlander" vs "Justin Verlander")
    parts = name_norm.split()
    if len(parts) >= 2:
        first_initial = parts[0][0]
        last = parts[-1]
        match = stats_df[
            stats_df["_name_normalized"].str.startswith(first_initial) &
            stats_df["_name_normalized"].str.endswith(last)
        ]
        if len(match) == 1:
            return match.iloc[0].to_dict()

    return None


def _rolling_pitcher_stats(
    pitcher_id: int,
    game_date: pd.Timestamp,
    game_log_df: pd.DataFrame,
    n_starts: int = 5
) -> dict:
    """
    Calculate rolling stats over last N starts for a pitcher.
    game_log_df expected cols: pitcher_id, game_date, era, fip, k_pct, ip
    """
    if pitcher_id is None:
        return {}

    log = game_log_df[
        (game_log_df["pitcher_id"] == pitcher_id) &
        (pd.to_datetime(game_log_df["game_date"]) < game_date)
    ].sort_values("game_date", ascending=False).head(n_starts)

    if log.empty:
        return {}

    # Days since last start
    last_start = pd.to_datetime(log.iloc[0]["game_date"])
    days_rest = (game_date - last_start).days

    return {
        "rolling_era":   log["era"].mean() if "era" in log.columns else np.nan,
        "rolling_fip":   log["fip"].mean() if "fip" in log.columns else np.nan,
        "rolling_k_pct": log["k_pct"].mean() if "k_pct" in log.columns else np.nan,
        "days_rest":     days_rest,
    }


def _default_pitcher_features(prefix: str) -> dict:
    """League-average fallback features when pitcher is TBD."""
    return {
        f"{prefix}era":          4.30,
        f"{prefix}fip":          4.20,
        f"{prefix}xfip":         4.25,
        f"{prefix}k_pct":        0.220,
        f"{prefix}bb_pct":       0.082,
        f"{prefix}whip":         1.28,
        f"{prefix}ip":           50.0,
        f"{prefix}war":          1.0,
        f"{prefix}rolling_era":  4.30,
        f"{prefix}rolling_fip":  4.20,
        f"{prefix}rolling_k_pct":0.220,
        f"{prefix}days_rest":    5,
    }


def add_pitcher_statcast(features: dict, statcast_dict: dict, prefix: str) -> dict:
    """Merge Statcast quality-of-contact metrics into pitcher features."""
    mapping = {
        "p_statcast_xwoba":        f"{prefix}statcast_xwoba",
        "p_statcast_barrel_pct":   f"{prefix}statcast_barrel_pct",
        "p_statcast_hard_hit_pct": f"{prefix}statcast_hard_hit_pct",
        "p_statcast_avg_ev":       f"{prefix}statcast_avg_ev",
    }
    for src, dst in mapping.items():
        features[dst] = statcast_dict.get(src, np.nan)
    return features


def build_pitcher_matchup_advantage(home_feats: dict, away_feats: dict) -> dict:
    """
    Calculate the differential between home and away starters.
    Positive values favor the home team.
    """
    return {
        "sp_fip_diff":   away_feats.get("away_sp_fip", 4.2) - home_feats.get("home_sp_fip", 4.2),
        "sp_k_pct_diff": home_feats.get("home_sp_k_pct", 0.22) - away_feats.get("away_sp_k_pct", 0.22),
        "sp_era_diff":   away_feats.get("away_sp_era", 4.3) - home_feats.get("home_sp_era", 4.3),
    }
