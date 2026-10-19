"""Threshold-free metrics, plus F1 at a threshold chosen on validation only."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y, scores) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _safe_ap(y, scores) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def best_f1_threshold(y_true, scores, grid: np.ndarray | None = None) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if grid is None:
        # Quantiles cover decision-function scores that are not probabilities.
        grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_t = float(np.median(scores))
    best_f = -1.0
    for t in grid:
        pred = scores >= t
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f:
            best_f = float(f)
            best_t = float(t)
    return best_t


def classification_report_dict(y_true, scores, threshold: float, probabilities: bool = True) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_noshow": float(precision_score(y_true, pred, zero_division=0)),
        "recall_noshow": float(recall_score(y_true, pred, zero_division=0)),
        "f1_noshow": float(f1_score(y_true, pred, zero_division=0)),
        "specificity": float(spec),
        "roc_auc": _safe_auc(y_true, scores),
        "pr_auc": _safe_ap(y_true, scores),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    if probabilities:
        clipped = np.clip(scores, 0.0, 1.0)
        out["brier"] = float(brier_score_loss(y_true, clipped))
    else:
        out["brier"] = float("nan")
    return out


def metrics_at_half_and_tuned(y_val, s_val, y_test, s_test, probabilities: bool = True) -> dict:
    tuned = best_f1_threshold(y_val, s_val)
    half = 0.5 if probabilities else tuned
    val_tuned = classification_report_dict(y_val, s_val, tuned, probabilities)
    test_tuned = classification_report_dict(y_test, s_test, tuned, probabilities)
    test_half = classification_report_dict(y_test, s_test, half, probabilities)
    return {
        "threshold_tuned": tuned,
        "val": val_tuned,
        "test": test_tuned,
        "test_at_0_5": test_half,
    }
