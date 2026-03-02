"""
predict/results_tracker.py
After each day's games finish, compare predictions to actuals
and append to a running accuracy log.
Run nightly after games complete (11pm ET).
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import date, datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fetch_schedule import fetch_schedule


def track_results(
    game_date: str = None,
    predictions_dir: str = "predict/output",
    accuracy_log: str = "predict/output/accuracy_log.csv",
):
    """
    Fetch final scores for a date, compare to predictions, and log accuracy.
    """
    if game_date is None:
        yesterday = date.today() - timedelta(days=1)
        game_date = yesterday.strftime("%Y-%m-%d")

    pred_path = f"{predictions_dir}/{game_date}_predictions.csv"
    if not os.path.exists(pred_path):
        print(f"No predictions found for {game_date}")
        return

    print(f"Tracking results for {game_date}...")

    # Load predictions
    preds = pd.read_csv(pred_path)

    # Fetch final scores
    results = fetch_schedule(game_date)
    results = results[results["status"] == "Final"]

    if results.empty:
        print(f"  No final scores found for {game_date}")
        return

    # Merge predictions with actuals
    merged = preds.merge(
        results[["home_team_abb", "away_team_abb", "home_score", "away_score"]],
        on=["home_team_abb", "away_team_abb"],
        how="inner",
    )

    if merged.empty:
        print("  Could not match predictions to results")
        return

    # Evaluate
    merged["actual_home_win"]  = (merged["home_score"] > merged["away_score"]).astype(int)
    merged["actual_total"]     = merged["home_score"] + merged["away_score"]
    merged["actual_home_cover"]= ((merged["home_score"] - merged["away_score"]) >= 2).astype(int)

    # Moneyline accuracy
    ml_correct = (merged["home_win_prob"] > 0.5) == (merged["actual_home_win"] == 1)
    ml_acc = ml_correct.mean()

    # O/U accuracy (predicted over/under vs. actual)
    ou_pred_over = merged["over_prob"] > 0.5
    ou_actual_over = merged["actual_total"] > merged.get("ou_total", 9.0)
    ou_acc = (ou_pred_over == ou_actual_over).mean() if "ou_total" in merged.columns else np.nan

    # MAE for total runs
    total_mae = np.abs(merged["predicted_total"] - merged["actual_total"]).mean()

    # Run line accuracy
    rl_correct = (merged["home_covers_prob"] > 0.5) == (merged["actual_home_cover"] == 1)
    rl_acc = rl_correct.mean()

    print(f"\n── Results for {game_date} ({len(merged)} games) ──")
    print(f"  Moneyline accuracy:  {ml_acc*100:.1f}%")
    print(f"  O/U accuracy:        {ou_acc*100:.1f}%  (when available)")
    print(f"  Run line accuracy:   {rl_acc*100:.1f}%")
    print(f"  Total runs MAE:      {total_mae:.2f}")

    # Flagged bets performance
    flagged = merged[merged.get("any_edge", False) == True] if "any_edge" in merged.columns else pd.DataFrame()
    if not flagged.empty:
        flag_ml_acc = ((flagged["home_win_prob"] > 0.5) == (flagged["actual_home_win"] == 1)).mean()
        print(f"  Flagged games ML acc:{flag_ml_acc*100:.1f}% ({len(flagged)} games)")

    # Append to accuracy log
    log_row = {
        "date":          game_date,
        "n_games":       len(merged),
        "ml_accuracy":   round(ml_acc, 4),
        "ou_accuracy":   round(ou_acc, 4) if not np.isnan(ou_acc) else None,
        "rl_accuracy":   round(rl_acc, 4),
        "total_mae":     round(total_mae, 3),
        "n_flagged":     len(flagged),
        "flagged_ml_acc":round(flag_ml_acc, 4) if not flagged.empty else None,
    }

    if os.path.exists(accuracy_log):
        log_df = pd.read_csv(accuracy_log)
        log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([log_row])

    log_df.to_csv(accuracy_log, index=False)
    print(f"\n  Accuracy log updated: {accuracy_log}")

    # Print rolling accuracy (last 30 days)
    if len(log_df) >= 5:
        recent = log_df.tail(30)
        print(f"\n── Rolling 30-day accuracy ──")
        print(f"  Moneyline: {recent['ml_accuracy'].mean()*100:.1f}%")
        print(f"  O/U:       {recent['ou_accuracy'].mean()*100:.1f}%")
        print(f"  Run line:  {recent['rl_accuracy'].mean()*100:.1f}%")
        print(f"  Total MAE: {recent['total_mae'].mean():.2f} runs")

    return log_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    track_results(game_date=args.date)
