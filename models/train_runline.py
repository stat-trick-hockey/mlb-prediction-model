"""
models/train_runline.py
Trains the run line (spread) model.
Predicts whether the home team covers -1.5 (i.e., wins by 2+).
Hardest of the three models — expect 53-56% accuracy. That's fine.
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.build_feature_matrix import get_model_feature_cols
from models.calibrate import calibrate_classifier, evaluate_calibration, save_model

# ── Hyperparameters ───────────────────────────────────────────────────────────
RL_PARAMS = {
    "n_estimators":    500,
    "max_depth":       4,
    "learning_rate":   0.04,
    "subsample":       0.75,
    "colsample_bytree":0.70,
    "min_child_weight":5,
    "reg_alpha":       0.3,
    "reg_lambda":      2.0,  # Stronger regularization for harder task
    "use_label_encoder": False,
    "eval_metric":    "logloss",
    "random_state":    42,
    "n_jobs":         -1,
}


def _build_runline_target(df: pd.DataFrame) -> pd.Series:
    """
    Build binary target: home team covers -1.5 (wins by 2 or more).
    """
    if "target_home_score" in df.columns and "target_away_score" in df.columns:
        return ((df["target_home_score"] - df["target_away_score"]) >= 2).astype(int)
    elif "target_home_win" in df.columns:
        # Fallback: use win (imprecise but usable)
        print("  WARNING: Run differential not available, using moneyline as proxy")
        return df["target_home_win"]
    else:
        raise ValueError("Cannot build run line target — missing score columns")


def train_runline_model(
    training_data_path: str = "data/processed/training_data.csv",
    val_season: int = 2024,
    output_path: str = "models/runline_model.pkl",
) -> object:
    """
    Train, calibrate, and validate the run line model.
    """
    print("── Training Run Line Model ──")
    df = pd.read_csv(training_data_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Build target
    df["target_runline"] = _build_runline_target(df)
    df = df.dropna(subset=["target_runline"])

    # Date-based split
    train_full = df[df["game_date"].dt.year < val_season].sort_values("game_date")
    val        = df[df["game_date"].dt.year == val_season]

    cal_cutoff = int(len(train_full) * 0.80)
    train = train_full.iloc[:cal_cutoff]
    cal   = train_full.iloc[cal_cutoff:]

    print(f"  Train: {len(train)} | Cal: {len(cal)} | Val: {len(val)}")
    print(f"  Home covers rate — train: {train['target_runline'].mean():.3f}, val: {val['target_runline'].mean():.3f}")

    feature_cols = get_model_feature_cols(df)
    train_meds   = train[feature_cols].median()

    X_train = train[feature_cols].fillna(train_meds)
    y_train = train["target_runline"]
    X_cal   = cal[feature_cols].fillna(train_meds)
    y_cal   = cal["target_runline"]
    X_val   = val[feature_cols].fillna(train_meds)
    y_val   = val["target_runline"]

    # Train
    model = XGBClassifier(**RL_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    # Evaluate pre-calibration
    raw_probs = model.predict_proba(X_val)[:, 1]
    print(f"\n── Pre-calibration ──")
    print(f"  Log loss: {log_loss(y_val, raw_probs):.4f}")
    print(f"  Brier:    {brier_score_loss(y_val, raw_probs):.4f}")
    print(f"  AUC-ROC:  {roc_auc_score(y_val, raw_probs):.4f}")

    # Calibrate
    calibrated = calibrate_classifier(model, X_cal, y_cal, method="isotonic")

    print("\n── Post-calibration ──")
    metrics = evaluate_calibration(calibrated, X_val, y_val, "Run Line", plot=True)

    # Feature importance
    _print_top_features(model, feature_cols)

    # Save
    save_model(calibrated, output_path)
    meta = {
        "feature_cols":  feature_cols,
        "train_medians": train_meds.to_dict(),
        "covers_rate":   float(train["target_runline"].mean()),
    }
    joblib.dump(meta, output_path.replace(".pkl", "_meta.pkl"))
    print(f"\n✓ Run line model saved to {output_path}")

    return calibrated


def predict_runline(
    X: pd.DataFrame,
    model_path: str = "models/runline_model.pkl",
) -> dict:
    """
    Predict probability that home team covers -1.5.
    """
    model = joblib.load(model_path)
    meta  = joblib.load(model_path.replace(".pkl", "_meta.pkl"))

    X_in = X[meta["feature_cols"]].fillna(pd.Series(meta["train_medians"]))
    proba = model.predict_proba(X_in)[0]

    return {
        "home_covers_prob": round(float(proba[1]), 4),
        "away_covers_prob": round(float(proba[0]), 4),
    }


def _print_top_features(model, feature_cols, top_n=15):
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False).head(top_n)
    print(f"\n── Top {top_n} Features ──")
    for feat, imp in importance.items():
        print(f"  {feat:45s} {imp:.4f}")


if __name__ == "__main__":
    train_runline_model()
