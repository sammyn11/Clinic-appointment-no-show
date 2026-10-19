"""Write one JSON file per run, and mirror the same numbers to W&B offline."""

from __future__ import annotations

import json
import os
from pathlib import Path

from noshow.config import EXP_DIR

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_DIR", str(EXP_DIR / "wandb"))


def log_run(run_id: str, config: dict, metrics: dict, notes: str = "") -> Path:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    runs = EXP_DIR / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    payload = {"id": run_id, "config": config, "metrics": metrics, "notes": notes}
    path = runs / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))

    try:
        import wandb

        wandb.init(
            project="appointment-noshow",
            name=run_id,
            config=config,
            mode="offline",
            reinit=True,
            dir=str(EXP_DIR / "wandb"),
        )
        flat = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub, subval in value.items():
                    if isinstance(subval, (int, float)):
                        flat[f"{key}/{sub}"] = subval
            elif isinstance(value, (int, float)):
                flat[key] = value
        if flat:
            wandb.log(flat)
        wandb.finish()
    except Exception as exc:  # logging should not wipe a finished fit
        (runs / f"{run_id}.wandb_error.txt").write_text(str(exc))
    return path
