"""Figures for the report. One idea per figure, labelled axes, no chartjunk."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

INK = "#1f2a2e"
TEAL = "#0f6f6a"
CLAY = "#c2643a"
SAND = "#e6e0d4"
BLUE = "#2f5d8c"


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titleweight": "medium",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d5d5d5",
            "grid.linewidth": 0.6,
            "font.size": 11,
            "savefig.bbox": "tight",
        }
    )


def _lead_bins(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[-0.1, 0, 2, 7, 14, 30, 10_000],
        labels=["0 (same day)", "1–2", "3–7", "8–14", "15–30", "31+"],
    )


def plot_eda(df: pd.DataFrame, fig_dir: Path) -> None:
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    tmp = df.copy()
    tmp["lead_bin"] = _lead_bins(tmp["lead_days"])
    rates = tmp.groupby("lead_bin", observed=True)["noshow"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(rates.index.astype(str), rates["mean"], color=TEAL)
    ax.set_ylabel("No-show rate")
    ax.set_xlabel("Days between booking and the appointment")
    ax.set_ylim(0, max(0.45, rates["mean"].max() + 0.05))
    ax.set_title("Missed visits rise with the booking lead time")
    for i, (m, n) in enumerate(zip(rates["mean"], rates["size"])):
        ax.text(i, m + 0.01, f"n={n:,}", ha="center", va="bottom", fontsize=8, color=INK)
    fig.savefig(fig_dir / "eda_lead_time.png")
    plt.close(fig)

    age_bin = pd.cut(
        tmp["age"],
        bins=[-0.1, 17, 34, 49, 64, 200],
        labels=["0–17", "18–34", "35–49", "50–64", "65+"],
    )
    age_rates = tmp.groupby(age_bin, observed=True)["noshow"].mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(age_rates.index.astype(str), age_rates.values, color=BLUE)
    ax.set_ylabel("No-show rate")
    ax.set_xlabel("Age")
    ax.set_title("Younger patients miss appointments more often")
    fig.savefig(fig_dir / "eda_age.png")
    plt.close(fig)

    sms = (
        tmp.groupby(["lead_bin", "sms_received"], observed=True)["noshow"]
        .mean()
        .unstack("sms_received")
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = np.arange(len(sms.index))
    width = 0.38
    if 0 in sms.columns:
        ax.bar(x - width / 2, sms[0], width, label="No SMS", color=SAND, edgecolor=INK, linewidth=0.4)
    if 1 in sms.columns:
        ax.bar(x + width / 2, sms[1], width, label="SMS sent", color=CLAY)
    ax.set_xticks(x, [str(i) for i in sms.index], rotation=15)
    ax.set_ylabel("No-show rate")
    ax.set_xlabel("Lead time (days)")
    ax.set_title("SMS looks harmful until lead time is held constant")
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "eda_sms_by_lead.png")
    plt.close(fig)

    hist = tmp.groupby("has_prior")["noshow"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    labels = ["No earlier visit\non record", "At least one visit\nbefore this booking"]
    ax.bar(labels, hist["mean"].reindex([0, 1]).values, color=[BLUE, TEAL])
    ax.set_ylabel("No-show rate")
    ax.set_title("A prior visit is already informative")
    fig.savefig(fig_dir / "eda_prior_visit.png")
    plt.close(fig)


def plot_learning_curve(rows: list[dict], path: Path, title: str) -> None:
    _style()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(frame["train_rows"], frame["train_auc"], marker="o", color=CLAY, label="Train AUC")
    ax.plot(frame["train_rows"], frame["val_auc"], marker="o", color=TEAL, label="Validation AUC")
    ax.set_xlabel("Training appointments (patient subsample)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(title)
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def plot_dl_curves(curve_dir: Path, runs: list[tuple[str, str]], path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for run_id, label in runs:
        csv_path = curve_dir / f"{run_id}.csv"
        if not csv_path.exists():
            continue
        h = pd.read_csv(csv_path)
        axes[0].plot(h["epoch"], h["loss"], label=f"{label} train")
        axes[0].plot(h["epoch"], h["val_loss"], linestyle="--", label=f"{label} val")
        axes[1].plot(h["epoch"], h["auc"], label=f"{label} train")
        axes[1].plot(h["epoch"], h["val_auc"], linestyle="--", label=f"{label} val")
    axes[0].set_title("Cross-entropy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("ROC-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0.5, 1.0)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Deep models: training against the patient validation split", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def plot_roc_pr(y_true, score_map: dict[str, np.ndarray], fig_dir: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for name, scores in score_map.items():
        RocCurveDisplay.from_predictions(y_true, scores, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], color="#888888", linewidth=0.8, linestyle=":")
    ax.set_title("Test ROC curves")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(fig_dir / "roc_test.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    prevalence = float(np.mean(y_true))
    for name, scores in score_map.items():
        PrecisionRecallDisplay.from_predictions(y_true, scores, name=name, ax=ax)
    ax.axhline(prevalence, color="#888888", linewidth=0.8, linestyle=":", label=f"Prevalence {prevalence:.2f}")
    ax.set_title("Test precision–recall curves")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(fig_dir / "pr_test.png")
    plt.close(fig)


def plot_confusion(y_true, scores, threshold: float, path: Path, title: str) -> None:
    _style()
    pred = (np.asarray(scores) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred. show", "Pred. no-show"])
    ax.set_yticks([0, 1], ["Attended", "No-show"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=INK, fontsize=12)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path)
    plt.close(fig)


def plot_importance(names: list[str], values: np.ndarray, path: Path, title: str, xlabel: str) -> None:
    _style()
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.barh(np.array(names)[order], np.array(values)[order], color=TEAL)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def plot_slices(slice_df: pd.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = np.arange(len(slice_df))
    ax.plot(x, slice_df["prevalence"], marker="o", color=BLUE, label="Actual no-show rate")
    ax.plot(x, slice_df["recall"], marker="o", color=CLAY, label="Recall of no-shows")
    ax.plot(x, slice_df["precision"], marker="o", color=TEAL, label="Precision of alerts")
    ax.set_xticks(x, slice_df["bin"].astype(str), rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Where the booked-alert threshold works, by lead time")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(y_true, scores, path: Path, title: str) -> None:
    _style()
    frac, mean_pred = calibration_curve(y_true, np.clip(scores, 0, 1), n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], linestyle=":", color="#888888")
    ax.plot(mean_pred, frac, marker="o", color=TEAL)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed no-show rate")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def lead_slice_table(df: pd.DataFrame, y, scores, threshold: float) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    rows = []
    tmp = df.reset_index(drop=True)
    tmp_bins = _lead_bins(tmp["lead_days"])
    for name in tmp_bins.cat.categories:
        mask = (tmp_bins == name).to_numpy()
        if mask.sum() == 0:
            continue
        yt = y[mask]
        pr = pred[mask]
        tp = int(((pr == 1) & (yt == 1)).sum())
        fp = int(((pr == 1) & (yt == 0)).sum())
        fn = int(((pr == 0) & (yt == 1)).sum())
        rows.append(
            {
                "bin": str(name),
                "n": int(mask.sum()),
                "prevalence": float(yt.mean()),
                "alert_rate": float(pr.mean()),
                "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
                "recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                "fn": fn,
                "fp": fp,
            }
        )
    return pd.DataFrame(rows)
