"""
features/team_form_features.py
Builds team form features: rolling W/L, run differential, home/away splits,
vs LHP/RHP splits, rest days, and Statcast batting metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def build_team_form_features(
    team_abb: str,
    game_date: str,
    is_home: bool,
    results_df: pd.DataFrame,
    statcast_df: Optional[pd.DataFrame] = None,
    team_batting_df: Optional[pd.DataFrame] = None,
    team_pitching_df: Optional[pd.DataFrame] = None,
    opposing_pitcher_hand: Optional[str] = None,
) -> dict:
    """
    Build team form features for one team.

    results_df expected cols:
        team_abb, game_date, is_home, win (0/1), runs_scored, runs_allowed,
        vs_pitcher_hand (L/R)

    Returns dict of features.
    """
    side   = "home" if is_home else "away"
    prefix = f"{side}_team_"
    ref    = pd.to_datetime(game_date)

    if results_df is None or results_df.empty:
        return {**_default_form_features(prefix, is_home),
                **_team_stats_features(team_abb, prefix, team_batting_df, team_pitching_df),
                **_statcast_batting_features(team_abb, prefix, statcast_df)}

    team_results = results_df[
        (results_df["team_abb"] == team_abb) &
        (pd.to_datetime(results_df["game_date"]) < ref)
    ].sort_values("game_date", ascending=False)

    # ── Rolling W/L ──────────────────────────────────────────────────────────
    last3  = team_results.head(3)
    last5  = team_results.head(5)
    last10 = team_results.head(10)

    wins3  = last3["win"].sum()  if not last3.empty  else 1.5
    wins5  = last5["win"].sum()  if not last5.empty  else 2.5
    wins10 = last10["win"].sum() if not last10.empty else 5.0

    # ── Run differential ──────────────────────────────────────────────────────
    if "runs_scored" in team_results.columns and "runs_allowed" in team_results.columns:
        rd10 = (last10["runs_scored"] - last10["runs_allowed"]).sum()
        avg_runs_scored  = last10["runs_scored"].mean()  if not last10.empty else 4.5
        avg_runs_allowed = last10["runs_allowed"].mean() if not last10.empty else 4.5
    else:
        rd10 = 0
        avg_runs_scored  = 4.5
        avg_runs_allowed = 4.5

    # ── Home/Away split ───────────────────────────────────────────────────────
    ha_filter = "home" if is_home else "away"
    if "is_home" in team_results.columns:
        ha_results = team_results[team_results["is_home"] == is_home].head(15)
        ha_win_pct = ha_results["win"].mean() if not ha_results.empty else 0.5
        ha_rd      = (ha_results["runs_scored"] - ha_results["runs_allowed"]).mean() \
                     if not ha_results.empty and "runs_scored" in ha_results.columns else 0.0
    else:
        ha_win_pct = 0.5
        ha_rd      = 0.0

    # ── vs LHP/RHP split ─────────────────────────────────────────────────────
    vs_hand_win_pct = 0.5
    if opposing_pitcher_hand and "vs_pitcher_hand" in team_results.columns:
        vs_hand = team_results[team_results["vs_pitcher_hand"] == opposing_pitcher_hand].head(20)
        vs_hand_win_pct = vs_hand["win"].mean() if not vs_hand.empty else 0.5

    # ── Rest days ─────────────────────────────────────────────────────────────
    if not team_results.empty:
        last_game = pd.to_datetime(team_results.iloc[0]["game_date"])
        rest_days = (ref - last_game).days
    else:
        rest_days = 3

    features = {
        f"{prefix}wins_last3":          int(wins3),
        f"{prefix}wins_last5":          int(wins5),
        f"{prefix}wins_last10":         int(wins10),
        f"{prefix}win_pct_last10":      round(wins10 / max(len(last10), 1), 3),
        f"{prefix}rd_last10":           round(rd10, 1),
        f"{prefix}avg_runs_scored":     round(avg_runs_scored, 2),
        f"{prefix}avg_runs_allowed":    round(avg_runs_allowed, 2),
        f"{prefix}ha_win_pct":          round(ha_win_pct, 3),
        f"{prefix}ha_rd":               round(ha_rd, 2),
        f"{prefix}vs_hand_win_pct":     round(vs_hand_win_pct, 3),
        f"{prefix}rest_days":           rest_days,
    }

    # Merge season stats and Statcast
    features.update(_team_stats_features(team_abb, prefix, team_batting_df, team_pitching_df))
    features.update(_statcast_batting_features(team_abb, prefix, statcast_df))

    return features


def _team_stats_features(
    team_abb: str,
    prefix: str,
    batting_df: Optional[pd.DataFrame],
    pitching_df: Optional[pd.DataFrame]
) -> dict:
    """Pull FanGraphs season stats for a team."""
    feats = {}

    if batting_df is not None and not batting_df.empty:
        row = batting_df[batting_df["team_abb"] == team_abb]
        if not row.empty:
            r = row.iloc[0]
            feats.update({
                f"{prefix}woba":    r.get("woba", 0.320),
                f"{prefix}wrc_plus":r.get("wrc_plus", 100),
                f"{prefix}bb_pct":  r.get("bb_pct", 0.085),
                f"{prefix}k_pct":   r.get("k_pct", 0.225),
                f"{prefix}iso":     r.get("iso", 0.165),
            })

    if pitching_df is not None and not pitching_df.empty:
        row = pitching_df[pitching_df["team_abb"] == team_abb]
        if not row.empty:
            r = row.iloc[0]
            feats.update({
                f"{prefix}team_era":    r.get("team_era", 4.2),
                f"{prefix}team_fip":    r.get("team_fip", 4.1),
                f"{prefix}team_k_pct":  r.get("team_k_pct", 0.22),
                f"{prefix}team_bb_pct": r.get("team_bb_pct", 0.085),
            })

    return feats


def _statcast_batting_features(
    team_abb: str,
    prefix: str,
    statcast_df: Optional[pd.DataFrame]
) -> dict:
    """Pull Statcast quality-of-contact metrics for a team."""
    if statcast_df is None or statcast_df.empty:
        return {
            f"{prefix}xwoba":          0.320,
            f"{prefix}barrel_pct":     0.075,
            f"{prefix}hard_hit_pct":   0.400,
            f"{prefix}avg_ev":         88.5,
        }

    row = statcast_df[statcast_df["team_abb"] == team_abb]
    if row.empty:
        return {
            f"{prefix}xwoba":          0.320,
            f"{prefix}barrel_pct":     0.075,
            f"{prefix}hard_hit_pct":   0.400,
            f"{prefix}avg_ev":         88.5,
        }

    r = row.iloc[0]
    return {
        f"{prefix}xwoba":          r.get("statcast_xwoba", 0.320),
        f"{prefix}barrel_pct":     r.get("statcast_barrel_pct", 0.075),
        f"{prefix}hard_hit_pct":   r.get("statcast_hard_hit_pct", 0.400),
        f"{prefix}avg_ev":         r.get("statcast_avg_ev", 88.5),
    }


def _default_form_features(prefix: str, is_home: bool) -> dict:
    """League-average form features."""
    return {
        f"{prefix}wins_last3":      1,
        f"{prefix}wins_last5":      2,
        f"{prefix}wins_last10":     5,
        f"{prefix}win_pct_last10":  0.500,
        f"{prefix}rd_last10":       0,
        f"{prefix}avg_runs_scored": 4.5,
        f"{prefix}avg_runs_allowed":4.5,
        f"{prefix}ha_win_pct":      0.540 if is_home else 0.460,
        f"{prefix}ha_rd":           0.0,
        f"{prefix}vs_hand_win_pct": 0.500,
        f"{prefix}rest_days":       1,
    }
