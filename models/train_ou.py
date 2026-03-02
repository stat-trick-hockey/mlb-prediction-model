"""
models/train_ou.py
Trains the Over/Under (total runs) regression model.
Uses XGBRegressor to predict total runs, then thresholds vs. the Vegas line.
O/U is the most tractable of the three targets — start here.
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.build_feature_matrix import get_model_feature_cols
from models.calibrate import save_model

# ── Hyperparameters ───────────────────────────────────────────────────────────
OU_PARAMS = {
    "n_estimators":    400,
    "max_depth":       5,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "min_child_weight":3,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "random_state":    42,
    "n_jobs":         -1,
}


def train_ou_model(
    training_data_path: str = "data/processed/training_data.csv",
    val_season: int = 2024,
    output_path: str = "models/ou_model.pkl",
) -> XGBRegressor:
    """
    Train and validate the O/U regression model.

    Train/test split is by date (not random) to prevent data leakage.
    2022-2023 = train, 2024 = validation.
    """
    print("── Training O/U Model ──")
    df = pd.read_csv(training_data_path)

    # Drop rows missing the target
    df = df.dropna(subset=["target_total_runs"])

    # Date-based train/val split
    df["game_date"] = pd.to_datetime(df["game_date"])
    train = df[df["game_date"].dt.year < val_season]
    val   = df[df["game_date"].dt.year == val_season]

    print(f"  Train: {len(train)} games ({train['game_date'].dt.year.min()}–{train['game_date'].dt.year.max()-1})")
    print(f"  Val:   {len(val)} games ({val_season})")

    feature_cols = get_model_feature_cols(df)
    X_train = train[feature_cols].fillna(train[feature_cols].median())
    y_train = train["target_total_runs"]
    X_val   = val[feature_cols].fillna(train[feature_cols].median())
    y_val   = val["target_total_runs"]

    # Train model
    model = XGBRegressor(**OU_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Evaluate
    val_preds = model.predict(X_val)
    mae  = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    print(f"\n── Validation Metrics ──")
    print(f"  MAE:  {mae:.3f} runs  (Vegas is typically ~0.5)")
    print(f"  RMSE: {rmse:.3f} runs")

    # Compare vs. Vegas O/U line (if available in val set)
    if "ou_total" in val.columns:
        vegas_mae = mean_absolute_error(y_val, val["ou_total"].fillna(y_val.mean()))
        print(f"  Vegas MAE: {vegas_mae:.3f} runs  (baseline)")
        print(f"  Model vs. Vegas: {'+' if mae > vegas_mae else ''}{mae - vegas_mae:.3f} runs")

    # Feature importance
    _print_top_features(model, feature_cols)
    _plot_predictions(y_val.values, val_preds, "O/U Model")

    save_model(model, output_path)
    # Also save feature columns and training medians for inference
    meta = {
        "feature_cols": feature_cols,
        "train_medians": train[feature_cols].median().to_dict(),
    }
    joblib.dump(meta, output_path.replace(".pkl", "_meta.pkl"))

    return model


def predict_ou(
    X: pd.DataFrame,
    ou_line: float,
    model_path: str = "models/ou_model.pkl",
) -> dict:
    """
    Predict total runs and over/under probability vs. a given Vegas line.

    Returns:
        predicted_total: model's run total prediction
        over_prob: estimated probability of going over the line
        under_prob: estimated probability of going under the line
    """
    model = joblib.load(model_path)
    meta  = joblib.load(model_path.replace(".pkl", "_meta.pkl"))

    X_in = X[meta["feature_cols"]].fillna(pd.Series(meta["train_medians"]))
    pred_total = float(model.predict(X_in)[0])

    # Convert point prediction to probability using empirical scoring distribution
    # Fitted from historical total run distributions (~2.5 run std dev)
    sigma = 2.5
    from scipy import stats
    over_prob  = 1 - stats.norm.cdf(ou_line + 0.5, loc=pred_total, scale=sigma)
    under_prob = stats.norm.cdf(ou_line - 0.5, loc=pred_total, scale=sigma)

    return {
        "predicted_total": round(pred_total, 2),
        "over_prob":       round(over_prob, 4),
        "under_prob":      round(under_prob, 4),
    }


def _print_top_features(model: XGBRegressor, feature_cols: list, top_n: int = 15):
    """Print top N features by importance."""
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False).head(top_n)
    print(f"\n── Top {top_n} Features ──")
    for feat, imp in importance.items():
        print(f"  {feat:45s} {imp:.4f}")


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    """Plot predicted vs. actual total runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(y_pred, y_true, alpha=0.3, s=10, color="#457B9D")
    lims = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    ax1.plot(lims, lims, "r--", alpha=0.7)
    ax1.set_xlabel("Predicted total runs")
    ax1.set_ylabel("Actual total runs")
    ax1.set_title(f"{title} — Predicted vs. Actual")
    ax1.grid(True, alpha=0.3)

    residuals = y_true - y_pred
    ax2.hist(residuals, bins=40, color="#E63946", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="black", linestyle="--")
    ax2.set_xlabel("Residual (actual − predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"{title} — Residuals")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("models/plots", exist_ok=True)
    plt.savefig(f"models/plots/ou_model.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    train_ou_model()
