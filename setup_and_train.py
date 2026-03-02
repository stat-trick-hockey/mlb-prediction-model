"""
setup_and_train.py
One-shot script to run all setup steps in order.
Run this once after cloning to build the full pipeline.
Corresponds to Steps 4–18 in the build guide.
"""

import os
import sys
import argparse

def run_step(step_name: str, fn, *args, **kwargs):
    print(f"\n{'═'*60}")
    print(f"  {step_name}")
    print(f"{'═'*60}")
    try:
        result = fn(*args, **kwargs)
        print(f"  ✓ {step_name} complete")
        return result
    except Exception as e:
        print(f"  ✗ {step_name} failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="MLB Prediction Model Setup")
    parser.add_argument("--skip-data",    action="store_true", help="Skip data fetching (use existing)")
    parser.add_argument("--skip-features",action="store_true", help="Skip feature matrix build")
    parser.add_argument("--skip-train",   action="store_true", help="Skip model training")
    parser.add_argument("--backtest-only",action="store_true", help="Run backtest only")
    parser.add_argument("--predict-today",action="store_true", help="Run today's predictions only")
    args = parser.parse_args()

    print("\n🔴 MLB Prediction Model — Setup & Training")
    print("=" * 60)

    # ── Phase 1: Data ──────────────────────────────────────────────────────────
    if not args.skip_data and not args.backtest_only and not args.predict_today:
        from data.fetch_fangraphs import save_fangraphs_data
        from data.fetch_statcast  import save_season_statcast

        for season in [2022, 2023, 2024]:
            run_step(f"FanGraphs data — {season}", save_fangraphs_data, season)

        for season in [2022, 2023, 2024]:
            run_step(
                f"Statcast data — {season} (this may take a while...)",
                save_season_statcast, season
            )

    # ── Phase 2: Feature Matrix ────────────────────────────────────────────────
    if not args.skip_features and not args.backtest_only and not args.predict_today:
        from features.build_feature_matrix import build_historical_feature_matrix
        run_step(
            "Build historical feature matrix (2022–2024)",
            build_historical_feature_matrix,
            seasons=[2022, 2023, 2024],
            output_path="data/processed/training_data.csv"
        )

    # ── Phase 3: Model Training ────────────────────────────────────────────────
    if not args.skip_train and not args.backtest_only and not args.predict_today:
        # Train in recommended order: O/U first (easiest), moneyline, run line last
        from models.train_ou        import train_ou_model
        from models.train_moneyline import train_moneyline_model
        from models.train_runline   import train_runline_model

        run_step("Train O/U model",       train_ou_model)
        run_step("Train Moneyline model", train_moneyline_model)
        run_step("Train Run Line model",  train_runline_model)

    # ── Phase 4: Backtest ─────────────────────────────────────────────────────
    if not args.predict_today:
        from backtest.backtest import run_backtest
        run_step("Walk-forward backtest (2024)", run_backtest, backtest_season=2024)

    # ── Phase 5: Today's Predictions ──────────────────────────────────────────
    if args.predict_today or not (args.skip_data or args.skip_features or args.skip_train):
        from predict.daily_predictions import run_daily_predictions
        run_step("Today's predictions", run_daily_predictions)

    print("\n✓ All steps complete!")
    print("\nNext steps:")
    print("  1. Add ODDS_API_KEY and WEATHER_API_KEY to .env")
    print("  2. Push to GitHub and add secrets in Settings → Secrets")
    print("  3. GitHub Actions will auto-run predictions at 10am ET daily")
    print("  4. Check predict/output/ for today's predictions")


if __name__ == "__main__":
    main()
