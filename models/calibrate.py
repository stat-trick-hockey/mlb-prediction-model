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
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CalibratedModel:
    """
    A version-safe calibrated classifier wrapper.
    Defined at module level so joblib can pickle it cleanly.
    Wraps any sklearn-compatible classifier with isotonic or sigmoid calibration.
    """

    def __init__(self, base_model, calibrator, method: str):
        self.base_model  = base_model
        self.calibrator  = calibrator
        self.method      = method

    def predict_proba(self, X) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            cal = self.calibrator.predict(raw)
        else:
            cal = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        cal = np.clip(cal, 0, 1)
        return np.column_stack([1 - cal, cal])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.base_model.feature_importances_


def calibrate_classifier(
    model,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    method: str = "isotonic",
) -> CalibratedModel:
    """
    Calibrate a trained classifier on a held-out calibration set.
    Uses isotonic regression (nonlinear, better for larger sets) or
    sigmoid / Platt scaling (linear, better for small sets).

    Args:
        model:  trained XGBClassifier (already fitted)
        X_cal:  held-out calibration features
        y_cal:  held-out calibration targets
        method: "isotonic" or "sigmoid"

    Returns:
        CalibratedModel instance — same predict_proba / predict interface
    """
    raw_probs = model.predict_proba(X_cal)[:, 1]

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, y_cal)
    else:
        calibrator = LogisticRegression()
        calibrator.fit(raw_probs.reshape(-1, 1), y_cal)

    return CalibratedModel(model, calibrator, method)


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
    Returns dict of metrics and optionally saves a calibration plot.
    """
    probs = model.predict_proba(X)[:, 1]

    ll    = log_loss(y, probs)
    brier = brier_score_loss(y, probs)
    ece   = _expected_calibration_error(y.values, probs, n_bins)

    fraction_pos, mean_pred = calibration_curve(y, probs, n_bins=n_bins)

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
        "accuracy": float((model.predict(X) == y).mean()),
    }


def _expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int,
) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(probs[mask].mean() - y_true[mask].mean())
    return ece


def _plot_calibration_curve(
    fraction_pos: np.ndarray,
    mean_pred: np.ndarray,
    probs: np.ndarray,
    model_name: str,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.plot(mean_pred, fraction_pos, "s-", label=model_name, color="#E63946")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(f"{model_name} — Calibration Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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
