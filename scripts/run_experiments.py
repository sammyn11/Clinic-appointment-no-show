#!/usr/bin/env python3
"""Train the no-show experiments.

Usage (from the repository root):

    PYTHONPATH=src .venv/bin/python scripts/run_experiments.py
    PYTHONPATH=src .venv/bin/python scripts/run_experiments.py --fast
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noshow.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
