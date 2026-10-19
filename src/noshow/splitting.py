"""Patient-level splits.

Appointments are not independent. A random row split puts the same person
on both sides, and with a history feature that is almost a copy of the
patient's habit the test score looks better than a new patient would justify.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from noshow.config import SEED


def _patient_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("patient_id")["noshow"].agg(n="size", rate="mean")
    g = g.reset_index()
    g["stratum"] = np.where(g["rate"] <= 0, "none", np.where(g["rate"] >= 0.5, "high", "mid"))
    return g


def _assign_stratum(ids: np.ndarray, counts: dict, test_size: float, val_size: float) -> dict[str, str]:
    """Give each stratum the same row mix, so the no-show rate does not drift.

    Filling the test set first, and walking strata in turn, used up the
    patients who sometimes miss visits. Train was then left with the
    people who always attend. Splitting inside each stratum avoids that.
    """
    total = float(sum(int(counts[i]) for i in ids))
    targets = {
        "test": test_size * total,
        "val": (1.0 - test_size) * val_size * total,
    }
    targets["train"] = total - targets["test"] - targets["val"]
    filled = {"train": 0.0, "val": 0.0, "test": 0.0}
    assignment = {}
    for pid in ids:
        def deficit(split, filled=filled, targets=targets):
            target = targets[split]
            if target <= 0:
                return -1e9
            return (target - filled[split]) / target

        split = max(("train", "val", "test"), key=deficit)
        assignment[str(pid)] = split
        filled[split] += float(counts[pid])
    return assignment


def split_by_patient(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.15,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, val, test integer positions.

    val_size is the share of the non-test rows kept for validation.
    """
    rng = np.random.RandomState(seed)
    patients = _patient_table(df)
    counts = dict(zip(patients["patient_id"], patients["n"].astype(int)))
    assignment: dict[str, str] = {}
    for _, part in patients.groupby("stratum"):
        ids = part["patient_id"].to_numpy().copy()
        rng.shuffle(ids)
        assignment.update(_assign_stratum(ids, counts, test_size, val_size))

    pid = df["patient_id"].to_numpy()
    split_of = np.array([assignment[str(p)] for p in pid])
    train_idx = np.flatnonzero(split_of == "train")
    val_idx = np.flatnonzero(split_of == "val")
    test_idx = np.flatnonzero(split_of == "test")

    # Nobody should appear twice.
    overlap = (
        len(set(pid[train_idx]) & set(pid[val_idx]))
        + len(set(pid[train_idx]) & set(pid[test_idx]))
        + len(set(pid[val_idx]) & set(pid[test_idx]))
    )
    if overlap:
        raise RuntimeError(f"patient overlap across splits: {overlap}")
    if len(train_idx) + len(val_idx) + len(test_idx) != len(df):
        raise RuntimeError("split did not cover every row")
    return train_idx, val_idx, test_idx


def split_summary(df: pd.DataFrame, train_idx, val_idx, test_idx) -> dict:
    def pack(name, idx):
        part = df.iloc[idx]
        return {
            "split": name,
            "rows": int(len(part)),
            "patients": int(part["patient_id"].nunique()),
            "noshow_rate": float(part["noshow"].mean()),
            "has_prior_rate": float(part["has_prior"].mean()) if "has_prior" in part else None,
        }

    return {
        "train": pack("train", train_idx),
        "val": pack("val", val_idx),
        "test": pack("test", test_idx),
    }
