"""Paths and constants shared by the notebook and the experiment script."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"
EXP_DIR = ROOT / "experiments"
CURVE_DIR = EXP_DIR / "training_curves"
TB_DIR = EXP_DIR / "tensorboard"

RAW_CSV = DATA_DIR / "noshowappointments-kagglev2-may-2016.csv"
# Public mirror of the Kaggle file published by Joni Hoppen (CC BY-NC-SA 4.0).
# The raw file is downloaded locally and is not committed.
DATA_URL = (
    "https://raw.githubusercontent.com/ksatola/Medical-Appointments-No-Shows/"
    "master/noshowappointments-kagglev2-may-2016.csv"
)

SEED = 42
# Smoothing constant for neighbourhood target encoding.
TE_SMOOTHING = 30

# Positive class is a missed appointment ("No-show" == Yes in the source file).
POSITIVE_LABEL = 1
