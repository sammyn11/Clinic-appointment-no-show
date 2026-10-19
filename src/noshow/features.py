"""Features that would be known when the clerk confirms the booking.

History is the part that is easy to get wrong. Sorting by the appointment
date and shifting the previous label looks reasonable, but it uses visits
that had not happened yet on the day the current appointment was booked.
A later booking for an earlier slot would then leak into an earlier booking
for a later slot.

Rule used here: a past visit counts only when its appointment date is
strictly before the calendar day on which the current appointment was
scheduled. Same-day visits are not treated as already observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from noshow.config import SEED, TE_SMOOTHING

BASE_COLS = [
    "age",
    "lead_days",
    "log_lead",
    "same_day",
    "female",
    "scholarship",
    "hipertension",
    "diabetes",
    "alcoholism",
    "handcap",
    "comorbidity_count",
    "scheduled_hour",
    "dow_sin",
    "dow_cos",
    "neighbourhood_freq",
    "neighbourhood_te",
]

HISTORY_COLS = [
    "prior_appointments",
    "log_prior_appointments",
    "prior_noshows",
    "prior_noshow_rate",
    "has_prior",
    "log_days_since_last",
]

SMS_COLS = ["sms_received"]

# Numeric inputs for the embedding network. Neighbourhood is an index, not a rate.
EMBED_NUM_COLS = [
    c for c in (BASE_COLS + HISTORY_COLS) if c not in ("neighbourhood_freq", "neighbourhood_te")
]


def add_booking_time_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["patient_id", "appt_date", "appointment_id"]).reset_index(drop=True)
    n = len(out)
    prior_n = np.zeros(n, dtype=np.int32)
    prior_ns = np.zeros(n, dtype=np.int32)
    days_since = np.zeros(n, dtype=np.float64)
    has_prior = np.zeros(n, dtype=np.int8)

    patient = out["patient_id"].to_numpy()
    appt = out["appt_date"].to_numpy().astype("datetime64[D]")
    sched = out["sched_date"].to_numpy().astype("datetime64[D]")
    y = out["noshow"].to_numpy().astype(np.int32)

    if len(patient) == 0:
        raise ValueError("empty frame")
    change = np.flatnonzero(patient[1:] != patient[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]

    for s, e in zip(starts, ends):
        days = appt[s:e]
        sched_d = sched[s:e]
        labels = y[s:e]
        csum = np.cumsum(labels)
        # last visit whose day is strictly before the booking date
        pos = np.searchsorted(days, sched_d, side="left") - 1
        valid = pos >= 0
        if not np.any(valid):
            continue
        local = np.arange(s, e)[valid]
        p = pos[valid]
        has_prior[local] = 1
        prior_n[local] = p + 1
        prior_ns[local] = csum[p]
        gap = (sched_d[valid] - days[p]).astype("timedelta64[D]").astype(np.int32)
        days_since[local] = gap

    out["prior_appointments"] = prior_n
    out["prior_noshows"] = prior_ns
    out["has_prior"] = has_prior
    out["days_since_last"] = days_since
    rate = np.zeros(n, dtype=np.float64)
    np.divide(prior_ns, prior_n, out=rate, where=prior_n > 0)
    out["prior_noshow_rate"] = rate
    out["log_prior_appointments"] = np.log1p(prior_n.astype(np.float64))
    out["log_days_since_last"] = np.log1p(days_since)
    return out


def _smooth_means(neighbourhood: pd.Series, y: np.ndarray, global_mean: float, m: float) -> pd.Series:
    tmp = pd.DataFrame({"neighbourhood": neighbourhood.to_numpy(), "y": y})
    agg = tmp.groupby("neighbourhood")["y"].agg(["sum", "count"])
    te = (agg["sum"] + m * global_mean) / (agg["count"] + m)
    return te


def fit_neighbourhood_encoder(train_df: pd.DataFrame, y_train: np.ndarray, m: float = TE_SMOOTHING) -> dict:
    global_mean = float(np.mean(y_train))
    freq = train_df["neighbourhood"].value_counts(normalize=True)
    te = _smooth_means(train_df["neighbourhood"], y_train, global_mean, m)
    index = {name: i + 1 for i, name in enumerate(sorted(freq.index))}
    return {
        "freq": freq,
        "te": te,
        "global_mean": global_mean,
        "index": index,
        "m": m,
    }


def oof_target_encode(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    m: float = TE_SMOOTHING,
    n_splits: int = 5,
    seed: int = SEED,
) -> np.ndarray:
    """Out-of-fold neighbourhood rates so a row's own label is not in its feature."""
    oof = np.empty(len(train_df), dtype=np.float64)
    global_mean = float(np.mean(y_train))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in splitter.split(train_df, y_train):
        stats = _smooth_means(train_df.iloc[tr]["neighbourhood"], y_train[tr], global_mean, m)
        mapped = train_df.iloc[va]["neighbourhood"].map(stats)
        oof[va] = mapped.fillna(global_mean).to_numpy(dtype=np.float64)
    return oof


def apply_neighbourhood(
    df: pd.DataFrame,
    encoder: dict,
    te_values: np.ndarray | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["neighbourhood_freq"] = out["neighbourhood"].map(encoder["freq"]).fillna(0.0).astype(float)
    if te_values is None:
        out["neighbourhood_te"] = (
            out["neighbourhood"].map(encoder["te"]).fillna(encoder["global_mean"]).astype(float)
        )
    else:
        out["neighbourhood_te"] = te_values
    out["neighbourhood_idx"] = out["neighbourhood"].map(encoder["index"]).fillna(0).astype(int)
    return out


def matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return df[columns].to_numpy(dtype=np.float64)
