"""
models/calibrate.py
Probability calibration utilities.
Wraps XGBoost classifiers with isotonic or Platt calibration.
Also provides evaluation metrics for probability calibration quality.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt


def calibrate_classifier(
    model,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    method: str = "isotonic",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Wrap a trained classifier with probability calibration.
    Uses isotonic regression (better for larger datasets, nonlinear).

    Args:
        model: trained XGBClassifier
        X_cal: calibration set features (held-out from training)
        y_cal: calibration set targets
        method: "isotonic" or "sigmoid" (Platt scaling)
        cv: "prefit" uses X_cal directly; int uses cross-val on X_cal

    Returns:
        calibrated model
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def evaluate_calibration(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Model",
    n_bins: int = 10,
    plot: bool = True,
) -> dict:
    """
    Evaluate probability calibration quality.
    Returns dict of metrics and optionally plots calibration curve.
    """
    probs = model.predict_proba(X)[:, 1]

    ll    = log_loss(y, probs)
    brier = brier_score_loss(y, probs)

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(y, probs, n_bins=n_bins)

    # ECE (expected calibration error)
    ece = _expected_calibration_error(y.values, probs, n_bins)

    print(f"\n── {model_name} Calibration ──")
    print(f"  Log loss:  {ll:.4f}")
    print(f"  Brier:     {brier:.4f}")
    print(f"  ECE:       {ece:.4f}")
    print(f"  Accuracy:  {(model.predict(X) == y).mean():.4f}")

    if plot:
        _plot_calibration_curve(fraction_pos, mean_pred, probs, model_name)

    return {
        "log_loss": ll,
        "brier":    brier,
        "ece":      ece,
        "accuracy": (model.predict(X) == y).mean(),
    }


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int) -> float:
    """Calculate expected calibration error (ECE)."""
    bins  = np.linspace(0, 1, n_bins + 1)
    ece   = 0.0
    n     = len(y_true)

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc  = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return ece


def _plot_calibration_curve(
    fraction_pos: np.ndarray,
    mean_pred: np.ndarray,
    probs: np.ndarray,
    model_name: str,
):
    """Plot calibration curve and probability distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.plot(mean_pred, fraction_pos, "s-", label=model_name, color="#E63946")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(f"{model_name} — Calibration Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Probability distribution
    ax2.hist(probs, bins=30, color="#457B9D", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title(f"{model_name} — Probability Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("models/plots", exist_ok=True)
    path = f"models/plots/{model_name.lower().replace(' ', '_')}_calibration.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved calibration plot to {path}")


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Saved model to {path}")


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
