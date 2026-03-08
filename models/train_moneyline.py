"""
models/train_moneyline.py
Trains the moneyline (win/loss) classification model.
Uses XGBClassifier with isotonic calibration to produce well-calibrated
win probabilities. Evaluate with Brier score and log loss, not just accuracy.
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.build_feature_matrix import get_model_feature_cols
from models.calibrate import calibrate_classifier, evaluate_calibration, save_model

# ── Hyperparameters ───────────────────────────────────────────────────────────
ML_PARAMS = {
    "n_estimators":    500,
    "max_depth":       4,
    "learning_rate":   0.04,
    "subsample":       0.75,
    "colsample_bytree":0.75,
    "min_child_weight":5,
    "scale_pos_weight":1.0,  # adjust if class imbalance
    "reg_alpha":       0.2,
    "reg_lambda":      1.5,
        "eval_metric":    "logloss",
    "random_state":    42,
    "n_jobs":         -1,
}


def train_moneyline_model(
    training_data_path: str = "data/processed/training_data.csv",
    val_season: int = 2024,
    output_path: str = "models/moneyline_model.pkl",
) -> object:
    """
    Train, calibrate, and validate the moneyline model.
    Returns the calibrated classifier.
    """
    print("── Training Moneyline Model ──")
    df = pd.read_csv(training_data_path)
    df = df.dropna(subset=["target_home_win"])
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Date-based split: train on earlier seasons, validate on val_season
    train_full = df[df["game_date"].dt.year < val_season]
    val        = df[df["game_date"].dt.year == val_season]

    # Further split train into train + calibration (80/20 by date)
    cal_cutoff = int(len(train_full) * 0.80)
    train_full_sorted = train_full.sort_values("game_date")
    train = train_full_sorted.iloc[:cal_cutoff]
    cal   = train_full_sorted.iloc[cal_cutoff:]

    print(f"  Train: {len(train)} | Cal: {len(cal)} | Val: {len(val)}")
    print(f"  Home win rate — train: {train['target_home_win'].mean():.3f}, val: {val['target_home_win'].mean():.3f}")

    feature_cols = get_model_feature_cols(df)
    train_meds   = train[feature_cols].median()

    X_train = train[feature_cols].fillna(train_meds)
    y_train = train["target_home_win"]
    X_cal   = cal[feature_cols].fillna(train_meds)
    y_cal   = cal["target_home_win"]
    X_val   = val[feature_cols].fillna(train_meds)
    y_val   = val["target_home_win"]

    # Train base model
    model = XGBClassifier(**ML_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    # Evaluate pre-calibration
    print("\n── Pre-calibration metrics ──")
    raw_probs = model.predict_proba(X_val)[:, 1]
    print(f"  Log loss: {log_loss(y_val, raw_probs):.4f}")
    print(f"  Brier:    {brier_score_loss(y_val, raw_probs):.4f}")
    print(f"  AUC-ROC:  {roc_auc_score(y_val, raw_probs):.4f}")

    # Calibrate
    calibrated_model = calibrate_classifier(model, X_cal, y_cal, method="isotonic")

    print("\n── Post-calibration metrics ──")
    metrics = evaluate_calibration(calibrated_model, X_val, y_val, "Moneyline", plot=True)

    # Print top features
    _print_top_features(model, feature_cols)

    # Baseline: always predict home win at historical rate (54%)
    home_rate = y_train.mean()
    baseline_ll = log_loss(y_val, np.full(len(y_val), home_rate))
    print(f"\n  Baseline log loss (always predict {home_rate:.3f}): {baseline_ll:.4f}")

    # Save
    save_model(calibrated_model, output_path)
    meta = {
        "feature_cols": feature_cols,
        "train_medians": train_meds.to_dict(),
        "home_win_rate": float(home_rate),
    }
    joblib.dump(meta, output_path.replace(".pkl", "_meta.pkl"))
    print(f"\n✓ Moneyline model saved to {output_path}")

    return calibrated_model


def predict_moneyline(
    X: pd.DataFrame,
    model_path: str = "models/moneyline_model.pkl",
) -> dict:
    """
    Predict home win probability.
    Returns home_win_prob and away_win_prob.
    """
    model = joblib.load(model_path)
    meta  = joblib.load(model_path.replace(".pkl", "_meta.pkl"))

    X_in = X[meta["feature_cols"]].fillna(pd.Series(meta["train_medians"]))
    proba = model.predict_proba(X_in)[0]

    return {
        "home_win_prob": round(float(proba[1]), 4),
        "away_win_prob": round(float(proba[0]), 4),
    }


def _print_top_features(model: XGBClassifier, feature_cols: list, top_n: int = 15):
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False).head(top_n)
    print(f"\n── Top {top_n} Features ──")
    for feat, imp in importance.items():
        print(f"  {feat:45s} {imp:.4f}")


if __name__ == "__main__":
    train_moneyline_model()
