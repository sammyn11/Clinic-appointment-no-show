"""Load the public appointment file and drop rows that are not usable."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from noshow.config import DATA_DIR, DATA_URL, EXP_DIR, RAW_CSV

EXPECTED_ROWS = 110_527


def ensure_raw_csv(path: Path = RAW_CSV) -> Path:
    """Download the CSV if it is not already on disk.

    The file is about 10 MB. It stays in data/ and is gitignored.
    """
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading dataset -> {path}")
    urllib.request.urlretrieve(DATA_URL, path)
    return path


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    path = ensure_raw_csv(path)
    # PatientId is a large integer. Reading it as float would merge distinct people.
    df = pd.read_csv(path, dtype={"PatientId": str, "AppointmentID": str})
    df["PatientId"] = df["PatientId"].str.replace(r"\.0$", "", regex=True)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a modelling frame and a short audit of what was removed.

    Lead time is the difference between the appointment *date* and the
    scheduling *date*. The appointment timestamp in this file is always
    midnight, so subtracting the raw timestamps makes ordinary same-day
    afternoon bookings look like negative waits. I compare calendar dates.
    """
    report = {"rows_raw": int(len(df))}
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]

    required = [
        "PatientId",
        "AppointmentID",
        "Gender",
        "ScheduledDay",
        "AppointmentDay",
        "Age",
        "Neighbourhood",
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "SMS_received",
        "No-show",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"unexpected columns, missing {missing}")

    out["patient_id"] = out["PatientId"].astype(str)
    out["appointment_id"] = out["AppointmentID"].astype(str)
    out["scheduled_day"] = pd.to_datetime(out["ScheduledDay"], utc=True)
    out["appointment_day"] = pd.to_datetime(out["AppointmentDay"], utc=True)

    sched_date = out["scheduled_day"].dt.tz_convert("UTC").dt.normalize()
    appt_date = out["appointment_day"].dt.tz_convert("UTC").dt.normalize()
    out["lead_days"] = (appt_date - sched_date).dt.days.astype(int)
    out["scheduled_hour"] = out["scheduled_day"].dt.hour.astype(int)
    out["appt_dow"] = appt_date.dt.dayofweek.astype(int)
    out["appt_date"] = appt_date.dt.tz_localize(None)
    out["sched_date"] = sched_date.dt.tz_localize(None)

    out["age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["female"] = (out["Gender"].str.upper() == "F").astype(int)
    out["neighbourhood"] = out["Neighbourhood"].str.strip()
    for col, src in [
        ("scholarship", "Scholarship"),
        ("hipertension", "Hipertension"),
        ("diabetes", "Diabetes"),
        ("alcoholism", "Alcoholism"),
        ("sms_received", "SMS_received"),
    ]:
        out[col] = pd.to_numeric(out[src], errors="coerce").fillna(0).astype(int)
    out["handcap"] = pd.to_numeric(out["Handcap"], errors="coerce").fillna(0).astype(int)
    # Source coding: "Yes" means the patient did not attend.
    out["noshow"] = (out["No-show"].astype(str).str.strip().str.lower() == "yes").astype(int)

    out["same_day"] = (out["lead_days"] == 0).astype(int)
    out["comorbidity_count"] = (
        out["hipertension"] + out["diabetes"] + out["alcoholism"]
    ).astype(int)

    n_before = len(out)
    bad_age = out["age"].isna() | (out["age"] < 0) | (out["age"] > 100)
    bad_lead = out["lead_days"] < 0
    bad_gender = ~out["Gender"].str.upper().isin(["F", "M"])
    dup_appt = out["appointment_id"].duplicated(keep="first")
    drop_mask = bad_age | bad_lead | bad_gender | dup_appt
    report["dropped_age"] = int(bad_age.sum())
    report["dropped_negative_lead"] = int(bad_lead.sum())
    report["dropped_gender"] = int(bad_gender.sum())
    report["dropped_duplicate_appointment"] = int(dup_appt.sum())
    out = out.loc[~drop_mask].copy()
    report["rows_after_row_filters"] = int(len(out))
    report["rows_removed"] = int(n_before - len(out))

    out["age"] = out["age"].astype(int)
    out["log_lead"] = np.log1p(out["lead_days"].to_numpy(dtype=float))
    # day-of-week as a circle, so Monday sits next to Sunday
    out["dow_sin"] = np.sin(2 * np.pi * out["appt_dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["appt_dow"] / 7.0)

    report["patients"] = int(out["patient_id"].nunique())
    report["noshow_rate"] = float(out["noshow"].mean())
    report["noshow_count"] = int(out["noshow"].sum())
    report["same_day_rate"] = float(out["same_day"].mean())
    report["date_min"] = str(out["appt_date"].min().date())
    report["date_max"] = str(out["appt_date"].max().date())
    report["neighbourhoods"] = int(out["neighbourhood"].nunique())
    report["rows_raw_expected"] = EXPECTED_ROWS
    return out.reset_index(drop=True), report


def save_cleaning_report(report: dict, path: Path | None = None) -> Path:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (EXP_DIR / "cleaning_report.json")
    path.write_text(json.dumps(report, indent=2))
    return path
