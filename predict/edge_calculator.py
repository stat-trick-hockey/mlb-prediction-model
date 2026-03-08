"""
predict/edge_calculator.py
Calculates betting edge and Kelly criterion sizing from model probabilities
vs. Vegas implied probabilities. Flags games worth acting on.
"""

import numpy as np
import pandas as pd
from typing import Optional

# Minimum edge to flag as actionable
# 5% is a realistic threshold — Vegas is accurate to ~2-3%, so 5% means
# the model needs to meaningfully disagree with the market to flag a bet.
# This targets ~25-35% bet rate rather than betting every game.
EDGE_THRESHOLD   = 0.05   # 5%
MAX_KELLY_FRAC   = 0.05   # Cap Kelly at 5% of bankroll (conservative)
MIN_KELLY_FRAC   = 0.005  # Ignore tiny Kelly fractions


def american_to_implied(american_odds: float) -> float:
    """Convert American odds to implied probability (no vig removed)."""
    if american_odds is None or np.isnan(american_odds):
        return np.nan
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def remove_vig(implied_home: float, implied_away: float) -> tuple:
    """
    Remove the bookmaker vig from a two-sided market.
    Returns fair implied probabilities that sum to 1.0.
    """
    total = implied_home + implied_away
    if total == 0:
        return 0.5, 0.5
    return implied_home / total, implied_away / total


def kelly_criterion(model_prob: float, implied_prob: float) -> float:
    """
    Kelly criterion fraction of bankroll to wager.
    f = (bp - q) / b
    where b = decimal odds - 1, p = model prob, q = 1 - p
    """
    if implied_prob <= 0 or implied_prob >= 1:
        return 0.0
    decimal_odds = 1 / implied_prob
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0.0, kelly)


def calculate_game_edges(
    game: dict,
    home_win_prob:  float,
    predicted_total: float,
    over_prob:       float,
    home_covers_prob: float,
) -> dict:
    """
    Calculate edge and Kelly sizing for all three markets in a game.

    Args:
        game: dict with Vegas lines (ml_home, ml_away, ou_total, rl_home_odds, etc.)
        home_win_prob: model's predicted home win probability
        predicted_total: model's predicted total runs
        over_prob: model's probability of going over the Vegas total
        home_covers_prob: model's probability of home covering -1.5

    Returns:
        dict with edge, Kelly, and flag for each market
    """
    results = {
        "home_team":     game.get("home_team_abb", game.get("home_team", "?")),
        "away_team":     game.get("away_team_abb", game.get("away_team", "?")),
        "home_pitcher":  game.get("home_pitcher", "TBD"),
        "away_pitcher":  game.get("away_pitcher", "TBD"),
    }

    # ── Moneyline ─────────────────────────────────────────────────────────────
    ml_home_odds = game.get("ml_home")
    ml_away_odds = game.get("ml_away")

    if ml_home_odds and ml_away_odds:
        raw_implied_home = american_to_implied(ml_home_odds)
        raw_implied_away = american_to_implied(ml_away_odds)
        fair_home, fair_away = remove_vig(raw_implied_home, raw_implied_away)

        ml_home_edge = home_win_prob - fair_home
        ml_away_edge = (1 - home_win_prob) - fair_away
        ml_home_kelly = kelly_criterion(home_win_prob, fair_home)
        ml_away_kelly = kelly_criterion(1 - home_win_prob, fair_away)

        results.update({
            "ml_home_odds":     ml_home_odds,
            "ml_away_odds":     ml_away_odds,
            "ml_home_prob":     round(home_win_prob, 4),
            "ml_away_prob":     round(1 - home_win_prob, 4),
            "ml_implied_home":  round(fair_home, 4),
            "ml_implied_away":  round(fair_away, 4),
            "ml_home_edge":     round(ml_home_edge, 4),
            "ml_away_edge":     round(ml_away_edge, 4),
            "ml_home_kelly":    round(min(ml_home_kelly, MAX_KELLY_FRAC), 4),
            "ml_away_kelly":    round(min(ml_away_kelly, MAX_KELLY_FRAC), 4),
            "ml_home_flag":     ml_home_edge >= EDGE_THRESHOLD and ml_home_kelly >= MIN_KELLY_FRAC,
            "ml_away_flag":     ml_away_edge >= EDGE_THRESHOLD and ml_away_kelly >= MIN_KELLY_FRAC,
        })
    else:
        results.update({
            "ml_home_edge": np.nan, "ml_away_edge": np.nan,
            "ml_home_flag": False,  "ml_away_flag": False,
        })

    # ── Over/Under ────────────────────────────────────────────────────────────
    ou_total     = game.get("ou_total")
    ou_over_odds = game.get("ou_over_odds", -110)
    ou_under_odds= game.get("ou_under_odds", -110)

    if ou_total:
        fair_over  = american_to_implied(ou_over_odds)  if ou_over_odds  else 0.5
        fair_under = american_to_implied(ou_under_odds) if ou_under_odds else 0.5
        fair_over, fair_under = remove_vig(fair_over, fair_under)

        ou_over_edge  = over_prob - fair_over
        ou_under_edge = (1 - over_prob) - fair_under
        ou_over_kelly = kelly_criterion(over_prob, fair_over)

        results.update({
            "ou_total":           ou_total,
            "predicted_total":    round(predicted_total, 2),
            "over_prob":          round(over_prob, 4),
            "under_prob":         round(1 - over_prob, 4),
            "ou_fair_over":       round(fair_over, 4),
            "ou_over_edge":       round(ou_over_edge, 4),
            "ou_under_edge":      round(ou_under_edge, 4),
            "ou_over_kelly":      round(min(ou_over_kelly, MAX_KELLY_FRAC), 4),
            "ou_over_flag":       ou_over_edge  >= EDGE_THRESHOLD,
            "ou_under_flag":      ou_under_edge >= EDGE_THRESHOLD,
        })
    else:
        results.update({
            "predicted_total": round(predicted_total, 2),
            "ou_over_edge": np.nan, "ou_under_edge": np.nan,
            "ou_over_flag": False,  "ou_under_flag": False,
        })

    # ── Run Line ──────────────────────────────────────────────────────────────
    rl_home_odds = game.get("rl_home_odds")
    rl_away_odds = game.get("rl_away_odds")

    if rl_home_odds and rl_away_odds:
        fair_rl_home, fair_rl_away = remove_vig(
            american_to_implied(rl_home_odds),
            american_to_implied(rl_away_odds)
        )
        rl_home_edge  = home_covers_prob - fair_rl_home
        rl_home_kelly = kelly_criterion(home_covers_prob, fair_rl_home)

        results.update({
            "rl_home_odds":    rl_home_odds,
            "rl_away_odds":    rl_away_odds,
            "rl_home_prob":    round(home_covers_prob, 4),
            "rl_fair_home":    round(fair_rl_home, 4),
            "rl_home_edge":    round(rl_home_edge, 4),
            "rl_home_kelly":   round(min(rl_home_kelly, MAX_KELLY_FRAC), 4),
            "rl_home_flag":    rl_home_edge >= EDGE_THRESHOLD,
        })
    else:
        results.update({"rl_home_edge": np.nan, "rl_home_flag": False})

    # ── Summary flag ──────────────────────────────────────────────────────────
    results["any_edge"] = any([
        results.get("ml_home_flag", False),
        results.get("ml_away_flag", False),
        results.get("ou_over_flag",  False),
        results.get("ou_under_flag", False),
        results.get("rl_home_flag",  False),
    ])

    return results


def format_edge_report(edge_results: list) -> str:
    """
    Format a list of game edge dicts into a readable daily report.
    """
    lines = ["═" * 75, "  MLB PREDICTION REPORT", "═" * 75]

    flagged = [g for g in edge_results if g.get("any_edge")]
    no_edge = [g for g in edge_results if not g.get("any_edge")]

    if flagged:
        lines.append(f"\n✅ EDGE GAMES ({len(flagged)}):\n")
        for g in flagged:
            home = g.get("home_team", "?")
            away = g.get("away_team", "?")
            lines.append(f"  {away} @ {home}")
            lines.append(f"    SP: {g.get('away_pitcher','TBD')} vs {g.get('home_pitcher','TBD')}")

            if g.get("ml_home_flag"):
                lines.append(f"    ★ ML {home}: {g['ml_home_prob']*100:.1f}% vs {g['ml_implied_home']*100:.1f}% implied → +{g['ml_home_edge']*100:.1f}% edge | Kelly: {g['ml_home_kelly']*100:.1f}%")
            if g.get("ml_away_flag"):
                lines.append(f"    ★ ML {away}: {g['ml_away_prob']*100:.1f}% vs {g['ml_implied_away']*100:.1f}% implied → +{g['ml_away_edge']*100:.1f}% edge | Kelly: {g['ml_away_kelly']*100:.1f}%")
            if g.get("ou_over_flag"):
                lines.append(f"    ★ OVER {g.get('ou_total')}: {g['over_prob']*100:.1f}% vs {g['ou_fair_over']*100:.1f}% implied → +{g['ou_over_edge']*100:.1f}% edge (proj: {g['predicted_total']} runs)")
            if g.get("ou_under_flag"):
                lines.append(f"    ★ UNDER {g.get('ou_total')}: {g['under_prob']*100:.1f}% vs {(1-g['ou_fair_over'])*100:.1f}% implied → +{g['ou_under_edge']*100:.1f}% edge (proj: {g['predicted_total']} runs)")
            if g.get("rl_home_flag"):
                lines.append(f"    ★ RL {home} -1.5: {g['rl_home_prob']*100:.1f}% vs {g['rl_fair_home']*100:.1f}% implied → +{g['rl_home_edge']*100:.1f}% edge")
            lines.append("")

    lines.append(f"\n── All Games ──")
    for g in edge_results:
        home = g.get("home_team", "?")
        away = g.get("away_team", "?")
        ml_h = g.get("ml_home_prob", 0.5)
        tot  = g.get("predicted_total", "?")
        lines.append(f"  {away} @ {home}  |  {home} win: {ml_h*100:.0f}%  |  Proj total: {tot}")

    lines.append("\n" + "═" * 75)
    return "\n".join(lines)
