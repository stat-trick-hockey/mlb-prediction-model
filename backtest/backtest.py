"""
backtest/backtest.py
Walk-forward accuracy backtest for all three models.

Measures predictive accuracy vs. naive baselines — no odds or betting simulation.
This is the honest way to evaluate model quality without real historical odds.

Metrics reported per period and overall:
  Moneyline : accuracy, log loss, Brier score, AUC-ROC vs. always-home baseline
  O/U       : MAE, RMSE vs. always-predict-mean baseline
  Run Line  : accuracy, log loss, AUC-ROC vs. always-away baseline (home -1.5 covers ~35%)
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import (
    log_loss, brier_score_loss, roc_auc_score,
    mean_absolute_error, mean_squared_error, accuracy_score
)
from xgboost import XGBClassifier, XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.build_feature_matrix import get_model_feature_cols
from models.calibrate import calibrate_classifier


def run_backtest(
    training_data_path: str = "data/processed/training_data.csv",
    backtest_season: int = 2024,
    refit_frequency: str = "month",
    output_dir: str = "backtest/results",
) -> pd.DataFrame:
    """
    Walk-forward accuracy backtest on a holdout season.

    For each period:
      1. Train on all data before that period
      2. Predict on that period's games
      3. Record predictions and actuals

    Then compute accuracy metrics across all periods.
    """
    print(f"── Walk-Forward Accuracy Backtest: {backtest_season} ──")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(training_data_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.copy()

    # Build run line target
    if "target_home_score" in df.columns and "target_away_score" in df.columns:
        df["target_runline"] = ((df["target_home_score"] - df["target_away_score"]) >= 2).astype(int)
    else:
        df["target_runline"] = df.get("target_home_win", 0)

    prior   = df[df["game_date"].dt.year < backtest_season]
    holdout = df[df["game_date"].dt.year == backtest_season].sort_values("game_date")

    if holdout.empty:
        print(f"  No data for backtest season {backtest_season}")
        return pd.DataFrame()

    feature_cols = get_model_feature_cols(df)
    periods      = _generate_periods(
        holdout["game_date"].min(),
        holdout["game_date"].max(),
        refit_frequency,
    )

    print(f"  Prior data:  {len(prior)} games ({prior['game_date'].dt.year.min()}–{prior['game_date'].dt.year.max()})")
    print(f"  Holdout:     {len(holdout)} games across {len(periods)} periods\n")

    all_rows = []

    for period_start, period_end in periods:
        period_games = holdout[
            (holdout["game_date"] >= period_start) &
            (holdout["game_date"] <  period_end)
        ]
        if period_games.empty:
            continue

        train_data = pd.concat([prior, holdout[holdout["game_date"] < period_start]])
        train_data = train_data.dropna(subset=["target_home_win"])

        models     = _train_period_models(train_data, feature_cols)
        period_rows = _predict_period(period_games, models, feature_cols, train_data)

        n = len(period_rows)
        ml_acc = np.mean([r["ml_correct"] for r in period_rows if r["ml_correct"] is not None])
        ou_mae = np.mean([abs(r["ou_error"]) for r in period_rows if r["ou_error"] is not None])

        print(f"  {period_start.strftime('%Y-%m-%d')} → {period_end.strftime('%Y-%m-%d')}: "
              f"{n} games | ML acc: {ml_acc:.3f} | O/U MAE: {ou_mae:.2f} runs")

        all_rows.extend(period_rows)

    if not all_rows:
        print("  No predictions generated.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_rows)
    _print_summary(results_df, prior, holdout, feature_cols)
    _plot_accuracy_over_time(results_df, output_dir, backtest_season)
    _plot_calibration(results_df, output_dir)

    out_path = f"{output_dir}/{backtest_season}_backtest.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n✓ Backtest results saved to {out_path}")

    return results_df


def _train_period_models(train_data: pd.DataFrame, feature_cols: list) -> dict:
    """Train fresh models on train_data for this period."""
    X = train_data[feature_cols].fillna(train_data[feature_cols].median())
    cal_cut = int(len(train_data) * 0.85)

    # Moneyline
    y_ml    = train_data["target_home_win"].dropna()
    X_ml    = X.loc[y_ml.index]
    ml_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        random_state=42, n_jobs=-1, eval_metric="logloss",
    )
    ml_model.fit(X_ml.iloc[:cal_cut], y_ml.iloc[:cal_cut], verbose=False)
    ml_cal = calibrate_classifier(ml_model, X_ml.iloc[cal_cut:], y_ml.iloc[cal_cut:])

    # O/U
    y_ou = train_data["target_total_runs"].dropna()
    ou_model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        random_state=42, n_jobs=-1,
    )
    ou_model.fit(X.loc[y_ou.index], y_ou, verbose=False)

    # Run line
    y_rl    = train_data["target_runline"].dropna()
    X_rl    = X.loc[y_rl.index]
    rl_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        random_state=42, n_jobs=-1, eval_metric="logloss",
    )
    rl_model.fit(X_rl.iloc[:cal_cut], y_rl.iloc[:cal_cut], verbose=False)
    rl_cal = calibrate_classifier(rl_model, X_rl.iloc[cal_cut:], y_rl.iloc[cal_cut:])

    return {"moneyline": ml_cal, "ou": ou_model, "runline": rl_cal}


def _predict_period(
    games: pd.DataFrame,
    models: dict,
    feature_cols: list,
    train_data: pd.DataFrame,
) -> list:
    """Generate predictions for a period, return list of result dicts."""
    train_meds = train_data[feature_cols].median()
    X = games[feature_cols].fillna(train_meds)

    ml_probs = models["moneyline"].predict_proba(X)[:, 1]
    ou_preds = models["ou"].predict(X)
    rl_probs = models["runline"].predict_proba(X)[:, 1]

    rows = []
    for i, (idx, game) in enumerate(games.iterrows()):
        home_win_prob    = float(ml_probs[i])
        predicted_total  = float(ou_preds[i])
        home_covers_prob = float(rl_probs[i])

        actual_win   = game.get("target_home_win")
        actual_total = game.get("target_total_runs")
        actual_rl    = game.get("target_runline")

        ml_pred    = int(home_win_prob >= 0.5)
        ml_correct = int(ml_pred == actual_win) if not pd.isna(actual_win) else None
        ou_error   = predicted_total - actual_total if not pd.isna(actual_total) else None
        rl_pred    = int(home_covers_prob >= 0.5)
        rl_correct = int(rl_pred == actual_rl) if not pd.isna(actual_rl) else None

        rows.append({
            "game_date":         game.get("game_date"),
            "home_team":         game.get("home_team_abb"),
            "away_team":         game.get("away_team_abb"),
            # Moneyline
            "home_win_prob":     round(home_win_prob, 4),
            "ml_pred":           ml_pred,
            "actual_win":        actual_win,
            "ml_correct":        ml_correct,
            # O/U
            "predicted_total":   round(predicted_total, 2),
            "actual_total":      actual_total,
            "ou_error":          round(ou_error, 2) if ou_error is not None else None,
            # Run line
            "home_covers_prob":  round(home_covers_prob, 4),
            "rl_pred":           rl_pred,
            "actual_rl":         actual_rl,
            "rl_correct":        rl_correct,
        })

    return rows


def _print_summary(
    results: pd.DataFrame,
    prior: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_cols: list,
):
    ml  = results.dropna(subset=["actual_win"])
    ou  = results.dropna(subset=["actual_total"])
    rl  = results.dropna(subset=["actual_rl"])

    home_win_rate = ml["actual_win"].mean()
    rl_cover_rate = rl["actual_rl"].mean()

    ml_acc      = accuracy_score(ml["actual_win"], ml["ml_pred"])
    ml_baseline = max(home_win_rate, 1 - home_win_rate)  # always pick majority
    ml_ll       = log_loss(ml["actual_win"], ml["home_win_prob"])
    ml_brier    = brier_score_loss(ml["actual_win"], ml["home_win_prob"])
    ml_auc      = roc_auc_score(ml["actual_win"], ml["home_win_prob"])

    ou_mae      = mean_absolute_error(ou["actual_total"], ou["predicted_total"])
    ou_rmse     = mean_squared_error(ou["actual_total"], ou["predicted_total"]) ** 0.5
    ou_baseline = mean_absolute_error(ou["actual_total"],
                                      [ou["actual_total"].mean()] * len(ou))

    rl_acc      = accuracy_score(rl["actual_rl"], rl["rl_pred"])
    rl_baseline = max(rl_cover_rate, 1 - rl_cover_rate)
    rl_auc      = roc_auc_score(rl["actual_rl"], rl["home_covers_prob"])

    print(f"""
{'═'*55}
  BACKTEST ACCURACY SUMMARY — {holdout['game_date'].dt.year.iloc[0]}
{'═'*55}
  Games evaluated: {len(ml)}

  MONEYLINE
    Accuracy:     {ml_acc:.3f}  (baseline: {ml_baseline:.3f}, +{ml_acc - ml_baseline:+.3f})
    AUC-ROC:      {ml_auc:.4f}
    Log loss:     {ml_ll:.4f}
    Brier score:  {ml_brier:.4f}
    Home win rate: {home_win_rate:.3f}

  OVER / UNDER
    MAE:          {ou_mae:.3f} runs  (baseline: {ou_baseline:.3f}, +{ou_mae - ou_baseline:+.3f})
    RMSE:         {ou_rmse:.3f} runs

  RUN LINE (home -1.5)
    Accuracy:     {rl_acc:.3f}  (baseline: {rl_baseline:.3f}, +{rl_acc - rl_baseline:+.3f})
    AUC-ROC:      {rl_auc:.4f}
    Home cover rate: {rl_cover_rate:.3f}
{'═'*55}""")


def _plot_accuracy_over_time(results: pd.DataFrame, output_dir: str, season: int):
    """Plot rolling 30-game accuracy for moneyline and run line."""
    df = results.dropna(subset=["ml_correct", "game_date"]).copy()
    df = df.sort_values("game_date")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"{season} Walk-Forward Accuracy", fontsize=14, fontweight="bold")

    for ax, col, label, color in [
        (axes[0], "ml_correct",  "Moneyline Accuracy (30-game rolling)", "#3B8EFF"),
        (axes[1], "rl_correct",  "Run Line Accuracy (30-game rolling)",  "#00E87A"),
    ]:
        rolling = df[col].rolling(30, min_periods=10).mean()
        ax.plot(df["game_date"], rolling, color=color, linewidth=1.5)
        ax.axhline(df[col].mean(), color=color, linestyle="--", alpha=0.5, label=f"Season avg: {df[col].mean():.3f}")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4, label="50% baseline")
        ax.set_ylabel("Accuracy")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.set_ylim(0.35, 0.70)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = f"{output_dir}/accuracy_over_time.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved accuracy chart to {path}")


def _plot_calibration(results: pd.DataFrame, output_dir: str):
    """Plot predicted probability vs actual win rate (calibration curve)."""
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Probability Calibration", fontsize=13, fontweight="bold")

    for ax, prob_col, actual_col, label in [
        (axes[0], "home_win_prob",    "actual_win", "Moneyline"),
        (axes[1], "home_covers_prob", "actual_rl",  "Run Line"),
    ]:
        sub = results.dropna(subset=[prob_col, actual_col])
        if sub.empty:
            continue
        frac_pos, mean_pred = calibration_curve(sub[actual_col], sub[prob_col], n_bins=10)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        ax.plot(mean_pred, frac_pos, "o-", color="#E63946", label=label)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual win rate")
        ax.set_title(f"{label} Calibration")
        ax.legend()
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = f"{output_dir}/calibration.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved calibration chart to {path}")


def _generate_periods(start, end, frequency="month"):
    periods = []
    current = start
    while current <= end:
        if frequency == "week":
            next_p = current + timedelta(days=7)
        else:
            if current.month == 12:
                next_p = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_p = current.replace(month=current.month + 1, day=1)
        periods.append((current, min(next_p, end + timedelta(days=1))))
        current = next_p
    return periods


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",   type=int, default=2024)
    parser.add_argument("--freq",     default="month", choices=["week", "month"])
    parser.add_argument("--data",     default="data/processed/training_data.csv")
    parser.add_argument("--out",      default="backtest/results")
    args = parser.parse_args()

    run_backtest(
        training_data_path=args.data,
        backtest_season=args.season,
        refit_frequency=args.freq,
        output_dir=args.out,
    )
