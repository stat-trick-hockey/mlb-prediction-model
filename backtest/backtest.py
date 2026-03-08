"""
backtest/backtest.py
Walk-forward backtesting of all three models.
Simulates betting season week-by-week, tracking ROI and calibration.
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.build_feature_matrix import get_model_feature_cols
from predict.edge_calculator import calculate_game_edges, american_to_implied, remove_vig


def run_backtest(
    training_data_path: str = "data/processed/training_data.csv",
    backtest_season: int = 2024,
    refit_frequency: str = "month",  # "week" or "month"
    edge_threshold: float = 0.04,
    bankroll: float = 1000.0,
    output_dir: str = "backtest/results",
) -> pd.DataFrame:
    """
    Walk-forward backtest on a holdout season.

    For each week/month:
    1. Train models on all data before that period
    2. Predict on that period's games
    3. Simulate bets on games with edge
    4. Track ROI and calibration
    """
    print(f"── Walk-Forward Backtest: {backtest_season} ──")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(training_data_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Build run line target
    df = df.copy()  # defragment before adding columns
    if "target_home_score" in df.columns and "target_away_score" in df.columns:
        df["target_runline"] = ((df["target_home_score"] - df["target_away_score"]) >= 2).astype(int)
    else:
        df["target_runline"] = df.get("target_home_win", 0)

    # Split: all prior seasons for initial train, backtest season for eval
    prior   = df[df["game_date"].dt.year < backtest_season]
    holdout = df[df["game_date"].dt.year == backtest_season].sort_values("game_date")

    if holdout.empty:
        print(f"  No data for backtest season {backtest_season}")
        return pd.DataFrame()

    feature_cols = get_model_feature_cols(df)

    # Generate time periods for rolling refit
    periods = _generate_periods(
        holdout["game_date"].min(),
        holdout["game_date"].max(),
        frequency=refit_frequency,
    )
    print(f"  Holdout: {len(holdout)} games across {len(periods)} periods")

    all_predictions = []
    current_bankroll = bankroll

    for period_start, period_end in periods:
        # Training data: everything before this period
        train_cutoff = period_start
        train_data = df[df["game_date"] < train_cutoff]

        if len(train_data) < 500:
            print(f"  Skipping {period_start.date()} — insufficient training data")
            continue

        # Get this period's games
        period_games = holdout[
            (holdout["game_date"] >= period_start) &
            (holdout["game_date"] < period_end)
        ]

        if period_games.empty:
            continue

        print(f"  Period {period_start.date()} → {period_end.date()}: {len(period_games)} games", end="")

        # Train models for this period
        models = _train_period_models(train_data, feature_cols)
        train_meds = train_data[feature_cols].median()

        # Predict and evaluate
        period_results = _predict_period(
            period_games, models, feature_cols, train_meds,
            edge_threshold, current_bankroll
        )
        all_predictions.extend(period_results)

        # Update bankroll
        period_pnl = sum(r.get("pnl", 0) for r in period_results)
        current_bankroll += period_pnl
        wins = sum(1 for r in period_results if r.get("bet_placed") and r.get("pnl", 0) > 0)
        bets = sum(1 for r in period_results if r.get("bet_placed"))
        print(f" | Bets: {bets} | P&L: ${period_pnl:+.2f} | Bankroll: ${current_bankroll:.2f}")

    results_df = pd.DataFrame(all_predictions)

    if results_df.empty:
        print("  No predictions generated.")
        return results_df

    # ── Summary metrics ───────────────────────────────────────────────────────
    _print_backtest_summary(results_df, bankroll, current_bankroll)
    _plot_bankroll_curve(results_df, bankroll, output_dir)
    _plot_calibration_by_market(results_df, output_dir)

    results_df.to_csv(f"{output_dir}/{backtest_season}_backtest.csv", index=False)
    print(f"\n✓ Backtest results saved to {output_dir}/{backtest_season}_backtest.csv")

    return results_df


def _train_period_models(train_data: pd.DataFrame, feature_cols: list) -> dict:
    """Quickly train XGBoost models for one period. Returns dict of models."""
    from xgboost import XGBClassifier, XGBRegressor
    from models.calibrate import calibrate_classifier

    X = train_data[feature_cols].fillna(train_data[feature_cols].median())

    # Moneyline
    y_ml = train_data["target_home_win"].dropna()
    ml_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08,
                              random_state=42, n_jobs=-1, eval_metric="logloss")
    ml_model.fit(X.loc[y_ml.index], y_ml, verbose=False)
    ml_cal = calibrate_classifier(ml_model, X.loc[y_ml.index], y_ml, method="isotonic")

    # O/U
    y_ou = train_data["target_total_runs"].dropna()
    ou_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                             random_state=42, n_jobs=-1)
    ou_model.fit(X.loc[y_ou.index], y_ou, verbose=False)

    # Run line
    y_rl = train_data["target_runline"].dropna()
    rl_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08,
                              random_state=42, n_jobs=-1, eval_metric="logloss")
    rl_model.fit(X.loc[y_rl.index], y_rl, verbose=False)
    rl_cal = calibrate_classifier(rl_model, X.loc[y_rl.index], y_rl, method="isotonic")

    return {"moneyline": ml_cal, "ou": ou_model, "runline": rl_cal}


def _predict_period(
    games: pd.DataFrame,
    models: dict,
    feature_cols: list,
    train_meds: pd.Series,
    edge_threshold: float,
    bankroll: float,
) -> list:
    """Generate predictions and simulated bets for a period."""
    from scipy import stats

    results = []
    X = games[feature_cols].fillna(train_meds)

    ml_probs   = models["moneyline"].predict_proba(X)[:, 1]
    ou_preds   = models["ou"].predict(X)
    rl_probs   = models["runline"].predict_proba(X)[:, 1]

    for i, (idx, game) in enumerate(games.iterrows()):
        actual_total = game.get("target_total_runs", np.nan)
        actual_win   = game.get("target_home_win", np.nan)
        actual_rl    = game.get("target_runline", np.nan)

        home_win_prob   = float(ml_probs[i])
        predicted_total = float(ou_preds[i])
        home_covers_prob= float(rl_probs[i])

        # Over probability from predicted total
        ou_line = game.get("ou_total", 9.0)
        sigma = 2.5
        over_prob = 1 - stats.norm.cdf(ou_line + 0.5, loc=predicted_total, scale=sigma)

        # Mock odds if not in data (use typical market)
        game_dict = game.to_dict()
        if pd.isna(game_dict.get("ml_home")):
            game_dict["ml_home"]      = -120
            game_dict["ml_away"]      = +105
            game_dict["ou_total"]     = 9.0
            game_dict["ou_over_odds"] = -110
            game_dict["ou_under_odds"]= -110
            game_dict["rl_home_odds"] = +145
            game_dict["rl_away_odds"] = -165

        edge = calculate_game_edges(
            game=game_dict,
            home_win_prob=home_win_prob,
            predicted_total=predicted_total,
            over_prob=float(over_prob),
            home_covers_prob=home_covers_prob,
        )

        # Simulate best bet (highest Kelly, above threshold)
        bet_placed, bet_side, bet_odds, bet_stake, pnl = _simulate_best_bet(
            edge, bankroll, actual_win, actual_total, actual_rl, game_dict
        )

        results.append({
            "game_date":        game.get("game_date"),
            "home_team":        game.get("home_team_abb"),
            "away_team":        game.get("away_team_abb"),
            "actual_win":       actual_win,
            "actual_total":     actual_total,
            "home_win_prob":    round(home_win_prob, 4),
            "predicted_total":  round(predicted_total, 2),
            "ou_line":          ou_line,
            "over_prob":        round(float(over_prob), 4),
            "home_covers_prob": round(home_covers_prob, 4),
            "ml_home_edge":     edge.get("ml_home_edge"),
            "ml_away_edge":     edge.get("ml_away_edge"),
            "ou_over_edge":     edge.get("ou_over_edge"),
            "ou_under_edge":    edge.get("ou_under_edge"),
            "rl_home_edge":     edge.get("rl_home_edge"),
            "bet_placed":       bet_placed,
            "bet_side":         bet_side,
            "bet_odds":         bet_odds,
            "bet_stake":        round(bet_stake, 2),
            "pnl":              round(pnl, 2),
        })

    return results


def _simulate_best_bet(edge, bankroll, actual_win, actual_total, actual_rl, game_dict):
    """Simulate placing the single best-edge bet for a game."""
    candidates = []

    if edge.get("ml_home_flag") and edge.get("ml_home_kelly", 0) > 0:
        candidates.append(("ml_home", game_dict.get("ml_home", -110), edge["ml_home_kelly"]))
    if edge.get("ml_away_flag") and edge.get("ml_away_kelly", 0) > 0:
        candidates.append(("ml_away", game_dict.get("ml_away", +105), edge["ml_away_kelly"]))
    if edge.get("ou_over_flag") and edge.get("ou_over_kelly", 0) > 0:
        candidates.append(("over", game_dict.get("ou_over_odds", -110), edge["ou_over_kelly"]))
    if edge.get("ou_under_flag"):
        candidates.append(("under", game_dict.get("ou_under_odds", -110), 0.02))
    if edge.get("rl_home_flag") and edge.get("rl_home_kelly", 0) > 0:
        candidates.append(("rl_home", game_dict.get("rl_home_odds", +145), edge["rl_home_kelly"]))

    if not candidates:
        return False, None, None, 0.0, 0.0

    # Pick highest Kelly
    best = max(candidates, key=lambda x: x[2])
    side, odds, kelly = best
    stake = bankroll * min(kelly * 0.25, 0.05)  # Quarter-Kelly, max 5%

    # Resolve bet
    won = _resolve_bet(side, odds, actual_win, actual_total, game_dict.get("ou_total", 9.0), actual_rl)
    if won is None:
        return True, side, odds, stake, 0.0

    if won:
        if odds > 0:
            pnl = stake * (odds / 100)
        else:
            pnl = stake * (100 / abs(odds))
    else:
        pnl = -stake

    return True, side, odds, stake, pnl


def _resolve_bet(side, odds, actual_win, actual_total, ou_line, actual_rl):
    """Return True if bet won, False if lost, None if push/no data."""
    if side == "ml_home":
        return bool(actual_win == 1) if not pd.isna(actual_win) else None
    elif side == "ml_away":
        return bool(actual_win == 0) if not pd.isna(actual_win) else None
    elif side == "over":
        if pd.isna(actual_total): return None
        return actual_total > ou_line
    elif side == "under":
        if pd.isna(actual_total): return None
        return actual_total < ou_line
    elif side == "rl_home":
        return bool(actual_rl == 1) if not pd.isna(actual_rl) else None
    return None


def _generate_periods(start, end, frequency="month"):
    """Generate (start, end) tuples for walk-forward periods."""
    periods = []
    current = start
    while current < end:
        if frequency == "week":
            next_p = current + timedelta(days=7)
        else:  # month
            if current.month == 12:
                next_p = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_p = current.replace(month=current.month + 1, day=1)
        periods.append((current, min(next_p, end + timedelta(days=1))))
        current = next_p
    return periods


def _print_backtest_summary(df: pd.DataFrame, initial: float, final: float):
    """Print key backtest metrics."""
    bets = df[df["bet_placed"] == True]
    print(f"\n{'═'*55}")
    print(f"  BACKTEST SUMMARY")
    print(f"{'═'*55}")
    print(f"  Total games:   {len(df)}")
    print(f"  Bets placed:   {len(bets)}")
    print(f"  Bet rate:      {len(bets)/max(len(df),1)*100:.1f}%")

    if not bets.empty:
        wins = (bets["pnl"] > 0).sum()
        print(f"  Win rate:      {wins/len(bets)*100:.1f}%")
        print(f"  Total P&L:     ${bets['pnl'].sum():+.2f}")
        print(f"  ROI:           {bets['pnl'].sum()/bets['bet_stake'].sum()*100:+.1f}%")
        print(f"  Avg P&L/bet:   ${bets['pnl'].mean():+.2f}")

    print(f"  Initial bankroll: ${initial:.2f}")
    print(f"  Final bankroll:   ${final:.2f}")
    print(f"  Return:           {(final/initial - 1)*100:+.1f}%")
    print(f"{'═'*55}")


def _plot_bankroll_curve(df: pd.DataFrame, initial: float, output_dir: str):
    """Plot cumulative P&L curve over the backtest period."""
    bets = df[df["bet_placed"] == True].copy()
    if bets.empty: return

    bets = bets.sort_values("game_date")
    bets["cumulative_pnl"] = bets["pnl"].cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(bets)), bets["cumulative_pnl"], color="#457B9D", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.fill_between(range(len(bets)), bets["cumulative_pnl"], 0,
                    where=bets["cumulative_pnl"] >= 0, alpha=0.2, color="#2a9d8f")
    ax.fill_between(range(len(bets)), bets["cumulative_pnl"], 0,
                    where=bets["cumulative_pnl"] < 0,  alpha=0.2, color="#e63946")
    ax.set_xlabel("Bet number")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title("Backtest — Cumulative P&L")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bankroll_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved bankroll curve to {output_dir}/bankroll_curve.png")


def _plot_calibration_by_market(df: pd.DataFrame, output_dir: str):
    """Plot model probability calibration vs. actual outcomes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    markets = [
        ("home_win_prob", "actual_win",   axes[0], "Moneyline"),
        ("over_prob",     None,            axes[1], "O/U (Over)"),
        ("home_covers_prob", "actual_win",axes[2], "Run Line"),
    ]

    for prob_col, actual_col, ax, title in markets:
        if prob_col not in df.columns:
            ax.set_title(f"{title}\n(no data)")
            continue
        probs = df[prob_col].dropna()
        if actual_col and actual_col in df.columns:
            actuals = df.loc[probs.index, actual_col].dropna()
            probs = probs.loc[actuals.index]

            n_bins = 10
            bins = np.linspace(0, 1, n_bins + 1)
            bin_mids, bin_actuals = [], []
            for i in range(n_bins):
                mask = (probs >= bins[i]) & (probs < bins[i+1])
                if mask.sum() > 10:
                    bin_mids.append(probs[mask].mean())
                    bin_actuals.append(actuals[mask].mean())

            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
            ax.scatter(bin_mids, bin_actuals, color="#E63946", s=60, zorder=5)
            ax.plot(bin_mids, bin_actuals, color="#E63946", alpha=0.7)
        else:
            ax.hist(probs, bins=20, color="#457B9D", alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual rate")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_backtest(backtest_season=2024)
