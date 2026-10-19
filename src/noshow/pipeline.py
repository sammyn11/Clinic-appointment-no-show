"""Fit the experiment list and write metrics, figures, and model files.

The test patients are not used to choose a hyperparameter. Search trials
report an inner patient holdout. The test columns in the results table are
computed after that choice, once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from noshow.config import (
    CURVE_DIR,
    EXP_DIR,
    FIG_DIR,
    MODEL_DIR,
    SEED,
    TB_DIR,
)
from noshow.data import clean, load_raw, save_cleaning_report
from noshow.features import (
    BASE_COLS,
    EMBED_NUM_COLS,
    HISTORY_COLS,
    SMS_COLS,
    add_booking_time_history,
    apply_neighbourhood,
    fit_neighbourhood_encoder,
    oof_target_encode,
)
from noshow.metrics import classification_report_dict, metrics_at_half_and_tuned
from noshow.models import (
    build_mlp_functional,
    build_mlp_sequential,
    class_weight_dict,
    compile_binary,
    hist_gbm,
    linear_svm,
    logistic,
    make_tfdata,
    make_tfdata_embed,
    random_forest,
    sample_weight_balanced,
    set_seeds,
)
from noshow.plots import (
    lead_slice_table,
    plot_calibration,
    plot_confusion,
    plot_dl_curves,
    plot_eda,
    plot_importance,
    plot_learning_curve,
    plot_roc_pr,
    plot_slices,
)
from noshow.splitting import split_by_patient, split_summary
from noshow.tracking import log_run

FULL_COLS = BASE_COLS + HISTORY_COLS
FULL_SMS_COLS = FULL_COLS + SMS_COLS
SCALE_COLS = FULL_SMS_COLS


def _auc(y, scores) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _positive_scores(model, X) -> tuple[np.ndarray, bool]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1], True
    return model.decision_function(X), False


def _eda_numbers(df: pd.DataFrame) -> dict:
    lead = pd.cut(
        df["lead_days"],
        bins=[-0.1, 0, 2, 7, 14, 30, 10_000],
        labels=["0", "1-2", "3-7", "8-14", "15-30", "31+"],
    )
    by_lead = df.groupby(lead, observed=True)["noshow"].agg(["mean", "size"])
    longer = df["lead_days"] >= 3
    return {
        "rows": int(len(df)),
        "patients": int(df["patient_id"].nunique()),
        "noshow_rate": float(df["noshow"].mean()),
        "same_day_share": float(df["same_day"].mean()),
        "same_day_noshow": float(df.loc[df["same_day"] == 1, "noshow"].mean()),
        "later_noshow": float(df.loc[df["same_day"] == 0, "noshow"].mean()),
        "female_share": float(df["female"].mean()),
        "female_noshow": float(df.loc[df["female"] == 1, "noshow"].mean()),
        "male_noshow": float(df.loc[df["female"] == 0, "noshow"].mean()),
        "scholarship_share": float(df["scholarship"].mean()),
        "scholarship_noshow": float(df.loc[df["scholarship"] == 1, "noshow"].mean()),
        "no_scholarship_noshow": float(df.loc[df["scholarship"] == 0, "noshow"].mean()),
        "sms_share": float(df["sms_received"].mean()),
        "sms_noshow": float(df.loc[df["sms_received"] == 1, "noshow"].mean()),
        "no_sms_noshow": float(df.loc[df["sms_received"] == 0, "noshow"].mean()),
        "sms_share_same_day": float(df.loc[df["same_day"] == 1, "sms_received"].mean()),
        "sms_noshow_lead_ge3": float(df.loc[longer & (df["sms_received"] == 1), "noshow"].mean()),
        "no_sms_noshow_lead_ge3": float(df.loc[longer & (df["sms_received"] == 0), "noshow"].mean()),
        "has_prior_share": float(df["has_prior"].mean()),
        "noshow_if_prior": float(df.loc[df["has_prior"] == 1, "noshow"].mean()),
        "noshow_if_no_prior": float(df.loc[df["has_prior"] == 0, "noshow"].mean()),
        "mean_lead_show": float(df.loc[df["noshow"] == 0, "lead_days"].mean()),
        "mean_lead_noshow": float(df.loc[df["noshow"] == 1, "lead_days"].mean()),
        "mean_prior_rate_show": float(df.loc[df["noshow"] == 0, "prior_noshow_rate"].mean()),
        "mean_prior_rate_noshow": float(df.loc[df["noshow"] == 1, "prior_noshow_rate"].mean()),
        "by_lead": {
            str(i): {"rate": float(r["mean"]), "n": int(r["size"])} for i, r in by_lead.iterrows()
        },
    }


def _prepare(fast: bool):
    raw = load_raw()
    cleaned, report = clean(raw)
    save_cleaning_report(report)
    featured = add_booking_time_history(cleaned)
    if fast:
        rng = np.random.RandomState(SEED)
        pids = np.asarray(featured["patient_id"].unique())
        rng.shuffle(pids)
        keep = set(pids[:6000])
        featured = featured.loc[featured["patient_id"].isin(keep)].reset_index(drop=True)
    eda = _eda_numbers(featured if not fast else add_booking_time_history(cleaned))
    # Descriptive stats always refer to the full cleaned cohort, not the fast sample.
    if fast:
        full = add_booking_time_history(cleaned)
        eda = _eda_numbers(full)
    else:
        eda = _eda_numbers(featured)
    (EXP_DIR).mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "eda_stats.json").write_text(json.dumps(eda, indent=2))
    plot_eda(featured if not fast else add_booking_time_history(cleaned), FIG_DIR)

    train_idx, val_idx, test_idx = split_by_patient(featured, seed=SEED)
    summary = split_summary(featured, train_idx, val_idx, test_idx)
    (EXP_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))

    train_df = featured.iloc[train_idx].reset_index(drop=True)
    val_df = featured.iloc[val_idx].reset_index(drop=True)
    test_df = featured.iloc[test_idx].reset_index(drop=True)
    y_train = train_df["noshow"].to_numpy(dtype=int)
    encoder = fit_neighbourhood_encoder(train_df, y_train)
    oof = oof_target_encode(train_df, y_train)
    train_df = apply_neighbourhood(train_df, encoder, te_values=oof)
    val_df = apply_neighbourhood(val_df, encoder)
    test_df = apply_neighbourhood(test_df, encoder)

    scaler = StandardScaler()
    scaler.fit(train_df[SCALE_COLS])

    bundle = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "y_train": y_train,
        "y_val": val_df["noshow"].to_numpy(dtype=int),
        "y_test": test_df["noshow"].to_numpy(dtype=int),
        "scaler": scaler,
        "encoder": encoder,
        "summary": summary,
        "cleaning": report,
        "eda": eda,
    }
    return bundle


def _X(df: pd.DataFrame, cols: list[str], scaler: StandardScaler | None) -> np.ndarray:
    if scaler is None:
        return df[cols].to_numpy(dtype=np.float64)
    transformed = scaler.transform(df[SCALE_COLS])
    frame = pd.DataFrame(transformed, columns=SCALE_COLS)
    return frame[cols].to_numpy(dtype=np.float32)


def _local_matrices(tr_df, ho_df, cols, scale: bool):
    y_tr = tr_df["noshow"].to_numpy(dtype=int)
    enc = fit_neighbourhood_encoder(tr_df, y_tr)
    oof = oof_target_encode(tr_df, y_tr, n_splits=3)
    tr_df = apply_neighbourhood(tr_df, enc, te_values=oof)
    ho_df = apply_neighbourhood(ho_df, enc)
    if not scale:
        return tr_df[cols].to_numpy(dtype=np.float64), ho_df[cols].to_numpy(dtype=np.float64), y_tr, ho_df["noshow"].to_numpy(dtype=int)
    sc = StandardScaler().fit(tr_df[SCALE_COLS])
    def take(frame):
        full = pd.DataFrame(sc.transform(frame[SCALE_COLS]), columns=SCALE_COLS)
        return full[cols].to_numpy(dtype=np.float32)
    return take(tr_df), take(ho_df), y_tr, ho_df["noshow"].to_numpy(dtype=int)


def _evaluate_fitted(model, X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    s_train, proba = _positive_scores(model, X_train)
    s_val, _ = _positive_scores(model, X_val)
    s_test, _ = _positive_scores(model, X_test)
    packed = metrics_at_half_and_tuned(y_val, s_val, y_test, s_test, probabilities=proba)
    if not proba:
        packed["test_at_default"] = classification_report_dict(y_test, s_test, 0.0, probabilities=False)
    packed["train_roc_auc"] = _auc(y_train, s_train)
    packed["scores_val"] = s_val
    packed["scores_test"] = s_test
    packed["probabilities"] = proba
    return packed


def _row_from_pack(exp_id, paradigm, algorithm, features, split, params, bundle, packed, seconds, notes, stage="heldout_test"):
    test = packed["test"]
    val = packed["val"]
    half = packed["test_at_0_5"]
    return {
        "experiment_id": exp_id,
        "stage": stage,
        "paradigm": paradigm,
        "algorithm": algorithm,
        "features": features,
        "split": split,
        "hyperparameters": json.dumps(params),
        "train_rows": int(len(bundle["y_train"])) if stage == "heldout_test" else params.get("inner_train_rows"),
        "val_rows": int(len(bundle["y_val"])) if stage == "heldout_test" else params.get("inner_val_rows"),
        "test_rows": int(len(bundle["y_test"])) if stage == "heldout_test" else None,
        "train_roc_auc": packed.get("train_roc_auc"),
        "val_roc_auc": val["roc_auc"],
        "val_pr_auc": val["pr_auc"],
        "threshold": packed["threshold_tuned"],
        "roc_auc": test["roc_auc"] if stage == "heldout_test" else None,
        "pr_auc": test["pr_auc"] if stage == "heldout_test" else None,
        "f1_noshow": test["f1_noshow"] if stage == "heldout_test" else None,
        "precision_noshow": test["precision_noshow"] if stage == "heldout_test" else None,
        "recall_noshow": test["recall_noshow"] if stage == "heldout_test" else None,
        "specificity": test["specificity"] if stage == "heldout_test" else None,
        "accuracy": test["accuracy"] if stage == "heldout_test" else None,
        "balanced_accuracy": test["balanced_accuracy"] if stage == "heldout_test" else None,
        "brier": test["brier"] if stage == "heldout_test" else None,
        "f1_at_0_5": half["f1_noshow"] if stage == "heldout_test" else None,
        "recall_at_0_5": half["recall_noshow"] if stage == "heldout_test" else None,
        "precision_at_0_5": half["precision_noshow"] if stage == "heldout_test" else None,
        "tp": test["tp"] if stage == "heldout_test" else None,
        "fp": test["fp"] if stage == "heldout_test" else None,
        "tn": test["tn"] if stage == "heldout_test" else None,
        "fn": test["fn"] if stage == "heldout_test" else None,
        "seconds": round(seconds, 1),
        "notes": notes,
    }


def _persist(rows: list[dict]) -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(EXP_DIR / "results.csv", index=False)


def _fit_sklearn(exp_id, algorithm, features, cols, model, bundle, scale, notes, params):
    X_train = _X(bundle["train_df"], cols, bundle["scaler"] if scale else None)
    X_val = _X(bundle["val_df"], cols, bundle["scaler"] if scale else None)
    X_test = _X(bundle["test_df"], cols, bundle["scaler"] if scale else None)
    y_train, y_val, y_test = bundle["y_train"], bundle["y_val"], bundle["y_test"]
    start = time.perf_counter()
    if algorithm.startswith("hist_gbm"):
        model.fit(X_train, y_train, sample_weight=sample_weight_balanced(y_train))
    else:
        model.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    packed = _evaluate_fitted(model, X_train, y_train, X_val, y_val, X_test, y_test)
    row = _row_from_pack(
        exp_id, "traditional", algorithm, features, "patient", params, bundle, packed, seconds, notes
    )
    # Logistic regression and the linear SVM also expose n_iter_, and that number
    # is a solver iteration count. Only boosting uses it as an early-stopping point.
    if algorithm.startswith("hist_gbm") and hasattr(model, "n_iter_"):
        n_iter = int(np.max(model.n_iter_))
        row["notes"] = notes + f" Early stopping stopped at {n_iter} boosting iterations."
        params = dict(params)
        params["n_iter_kept"] = n_iter
        row["hyperparameters"] = json.dumps(params)
    log_run(exp_id, params, {"val_roc_auc": packed["val"]["roc_auc"], "test": packed["test"], "train_roc_auc": packed["train_roc_auc"]}, notes)
    return row, model, packed


def _inner_indices(train_df: pd.DataFrame):
    tr, _, ho = split_by_patient(train_df, test_size=0.25, val_size=0.0, seed=SEED + 7)
    return tr, ho


def run(fast: bool = False) -> pd.DataFrame:
    set_seeds(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print("preparing data")
    bundle = _prepare(fast=fast)
    rows: list[dict] = []
    fitted = {}

    y_train, y_val, y_test = bundle["y_train"], bundle["y_val"], bundle["y_test"]
    print(
        "split",
        {k: (v["rows"], round(v["noshow_rate"], 3)) for k, v in bundle["summary"].items()},
    )

    # E00 majority: score everyone with the training prevalence.
    prev = float(y_train.mean())
    s_val = np.full(len(y_val), prev)
    s_test = np.full(len(y_test), prev)
    s_train = np.full(len(y_train), prev)
    packed = metrics_at_half_and_tuned(y_val, s_val, y_test, s_test, probabilities=True)
    packed["train_roc_auc"] = _auc(y_train, s_train)
    packed["scores_val"] = s_val
    packed["scores_test"] = s_test
    packed["probabilities"] = True
    rows.append(
        _row_from_pack(
            "E00",
            "baseline",
            "majority_prevalence",
            "none",
            "patient",
            {"constant_score": prev},
            bundle,
            packed,
            0.0,
            "Every row is scored with the training no-show rate. Accuracy will look high because most people attend. AUC stays at 0.5.",
        )
    )
    _persist(rows)

    specs = [
        ("E01", "logreg", "base_booking", BASE_COLS, logistic(C=1.0, class_weight=None), True,
         "Unweighted logistic regression on booking-time fields, no personal history. Linear baseline."),
        ("E02", "logreg", "full_booking", FULL_COLS, logistic(C=1.0, class_weight=None), True,
         "Same model as E01 after adding booking-time history. The comparison isolates the history features."),
    ]
    for exp_id, algo, feat, cols, model, scale, notes in specs:
        print("fitting", exp_id)
        row, model, packed = _fit_sklearn(exp_id, algo, feat, cols, model, bundle, scale, notes, {"C": 1.0, "class_weight": None})
        rows.append(row)
        fitted[exp_id] = (model, packed, cols, scale)
        _persist(rows)

    # E03 class weight and C, chosen on an inner patient holdout.
    print("searching logistic C")
    tr_idx, ho_idx = _inner_indices(bundle["train_df"])
    tr_df = bundle["train_df"].iloc[tr_idx].reset_index(drop=True)
    ho_df = bundle["train_df"].iloc[ho_idx].reset_index(drop=True)
    best_C = None
    best_inner = -1.0
    C_grid = [0.01, 0.1, 1.0, 10.0] if not fast else [0.1, 1.0]
    for C in C_grid:
        Xtr, Xho, ytr, yho = _local_matrices(tr_df, ho_df, FULL_COLS, scale=True)
        start = time.perf_counter()
        model = logistic(C=C, class_weight="balanced")
        model.fit(Xtr, ytr)
        seconds = time.perf_counter() - start
        scores = model.predict_proba(Xho)[:, 1]
        inner_auc = _auc(yho, scores)
        inner_pack = {
            "val": classification_report_dict(yho, scores, 0.5, True),
            "test": classification_report_dict(yho, scores, 0.5, True),
            "test_at_0_5": classification_report_dict(yho, scores, 0.5, True),
            "threshold_tuned": 0.5,
            "train_roc_auc": _auc(ytr, model.predict_proba(Xtr)[:, 1]),
        }
        # The helper expects val/test. For search rows I overwrite stage afterwards.
        params = {
            "C": C,
            "class_weight": "balanced",
            "inner_train_rows": int(len(ytr)),
            "inner_val_rows": int(len(yho)),
        }
        row = _row_from_pack(
            f"E03_C{C}",
            "traditional",
            "logreg",
            "full_booking",
            "inner_patient_holdout",
            params,
            bundle,
            inner_pack,
            seconds,
            "Inner holdout only. Not a test score. Used to pick C.",
            stage="inner_search",
        )
        row["val_roc_auc"] = inner_auc
        rows.append(row)
        log_run(f"E03_C{C}", params, {"inner_roc_auc": inner_auc}, row["notes"])
        if inner_auc > best_inner:
            best_inner = inner_auc
            best_C = C
    print("selected C", best_C, "inner auc", round(best_inner, 4))
    row, model, packed = _fit_sklearn(
        "E03",
        "logreg_balanced",
        "full_booking",
        FULL_COLS,
        logistic(C=best_C, class_weight="balanced"),
        bundle,
        True,
        f"Class-weighted logistic regression. C={best_C} won the inner holdout (inner ROC-AUC {best_inner:.3f}). Refit on all training patients.",
        {"C": best_C, "class_weight": "balanced", "inner_roc_auc": best_inner},
    )
    rows.append(row)
    fitted["E03"] = (model, packed, FULL_COLS, True)
    _persist(rows)

    # Forests.
    rf_trees = 40 if fast else 200
    print("fitting E04 random forest")
    row, model, packed = _fit_sklearn(
        "E04",
        "random_forest",
        "full_booking",
        FULL_COLS,
        random_forest(n_estimators=rf_trees, max_depth=12, min_samples_leaf=5, max_features="sqrt"),
        bundle,
        False,
        "Preset forest. Depth is capped so a few noisy patients do not grow pure leaves.",
        {"n_estimators": rf_trees, "max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": "balanced_subsample"},
    )
    rows.append(row)
    fitted["E04"] = (model, packed, FULL_COLS, False)
    _persist(rows)

    rf_grid = [
        {"max_depth": 8, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"max_depth": 16, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"max_depth": 16, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"max_depth": 12, "min_samples_leaf": 5, "max_features": 0.5},
        {"max_depth": None, "min_samples_leaf": 10, "max_features": "sqrt"},
    ]
    if fast:
        rf_grid = rf_grid[:2]
    search_trees = 25 if fast else 120
    best_rf = None
    best_rf_auc = -1.0
    print("searching random forest")
    for i, params in enumerate(rf_grid, start=1):
        Xtr, Xho, ytr, yho = _local_matrices(tr_df, ho_df, FULL_COLS, scale=False)
        start = time.perf_counter()
        model = random_forest(n_estimators=search_trees, **params)
        model.fit(Xtr, ytr)
        seconds = time.perf_counter() - start
        inner_auc = _auc(yho, model.predict_proba(Xho)[:, 1])
        inner_pack = {
            "val": classification_report_dict(yho, model.predict_proba(Xho)[:, 1], 0.5, True),
            "test": classification_report_dict(yho, model.predict_proba(Xho)[:, 1], 0.5, True),
            "test_at_0_5": classification_report_dict(yho, model.predict_proba(Xho)[:, 1], 0.5, True),
            "threshold_tuned": 0.5,
            "train_roc_auc": _auc(ytr, model.predict_proba(Xtr)[:, 1]),
        }
        full_params = dict(params)
        full_params.update({"n_estimators": search_trees, "inner_train_rows": int(len(ytr)), "inner_val_rows": int(len(yho))})
        row = _row_from_pack(
            f"E05_trial{i}",
            "traditional",
            "random_forest",
            "full_booking",
            "inner_patient_holdout",
            full_params,
            bundle,
            inner_pack,
            seconds,
            "Inner patient holdout used to choose depth, leaf size, and feature subsample.",
            stage="inner_search",
        )
        row["val_roc_auc"] = inner_auc
        rows.append(row)
        log_run(f"E05_trial{i}", full_params, {"inner_roc_auc": inner_auc}, row["notes"])
        print("  trial", i, "auc", round(inner_auc, 4), params)
        if inner_auc > best_rf_auc:
            best_rf_auc = inner_auc
            best_rf = params
    final_trees = 50 if fast else 260
    row, model, packed = _fit_sklearn(
        "E05",
        "random_forest_tuned",
        "full_booking",
        FULL_COLS,
        random_forest(n_estimators=final_trees, **best_rf),
        bundle,
        False,
        f"Forest refit with the inner-holdout winner (inner ROC-AUC {best_rf_auc:.3f}) and more trees.",
        {"n_estimators": final_trees, **best_rf, "inner_roc_auc": best_rf_auc, "class_weight": "balanced_subsample"},
    )
    rows.append(row)
    fitted["E05"] = (model, packed, FULL_COLS, False)
    _persist(rows)

    hgb_iter = 40 if fast else 250
    print("fitting gradient boosting")
    for exp_id, lr, depth, l2, note in [
        ("E06", 0.08, 6, 1.0, "Slow learning rate, depth 6, L2 penalty 1. Early stopping watches an internal slice of the training patients' rows."),
        ("E07", 0.2, 3, 0.1, "Faster learning rate and shallower trees. This is the bias end of the same algorithm, against E06."),
    ]:
        row, model, packed = _fit_sklearn(
            exp_id,
            "hist_gradient_boosting",
            "full_booking",
            FULL_COLS,
            hist_gbm(lr, depth, l2, hgb_iter),
            bundle,
            False,
            note,
            {"learning_rate": lr, "max_depth": depth, "l2_regularization": l2, "max_iter": hgb_iter, "sample_weight": "balanced"},
        )
        rows.append(row)
        fitted[exp_id] = (model, packed, FULL_COLS, False)
        _persist(rows)

    print("fitting linear SVM")
    row, model, packed = _fit_sklearn(
        "E08",
        "linear_svc",
        "full_booking",
        FULL_COLS,
        linear_svm(C=1.0),
        bundle,
        True,
        "Linear SVM with a balanced class weight. Scores are margins, not probabilities, so the default alert rule is the sign of the margin (threshold 0) and the F1 threshold is chosen on validation margins.",
        {"C": 1.0, "class_weight": "balanced", "dual": False},
    )
    rows.append(row)
    fitted["E08"] = (model, packed, FULL_COLS, True)
    _persist(rows)

    print("fitting SMS ablation")
    row, model, packed = _fit_sklearn(
        "E09",
        "logreg_balanced",
        "full_booking_plus_sms",
        FULL_SMS_COLS,
        logistic(C=best_C, class_weight="balanced"),
        bundle,
        True,
        "Same balanced logistic model as E03 with the SMS flag added. SMS is an intervention, not a patient trait, so it is not in the main feature set.",
        {"C": best_C, "class_weight": "balanced", "sms": True},
    )
    rows.append(row)
    fitted["E09"] = (model, packed, FULL_SMS_COLS, True)
    _persist(rows)

    # Row split sensitivity. Same honest features, but patients can sit on both sides.
    print("fitting row-split sensitivity")
    idx = np.arange(len(bundle["train_df"]) + len(bundle["val_df"]) + len(bundle["test_df"]))
    # Rebuild from the three frames stacked in original feature space is messy because
    # encodings differ. Restack the unencoded columns from the three frames.
    # History columns are already booking-time safe; only the neighbourhood encoding
    # must be refit inside the row split.
    stacked = pd.concat([bundle["train_df"], bundle["val_df"], bundle["test_df"]], ignore_index=True)
    y_all = stacked["noshow"].to_numpy(dtype=int)
    tr_i, te_i = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=SEED)
    tr_i, va_i = train_test_split(tr_i, test_size=0.15, stratify=y_all[tr_i], random_state=SEED)
    overlap = len(set(stacked.loc[tr_i, "patient_id"]) & set(stacked.loc[te_i, "patient_id"]))
    enc = fit_neighbourhood_encoder(stacked.iloc[tr_i], y_all[tr_i])
    oof = oof_target_encode(stacked.iloc[tr_i], y_all[tr_i])
    tr_e = apply_neighbourhood(stacked.iloc[tr_i], enc, te_values=oof)
    va_e = apply_neighbourhood(stacked.iloc[va_i], enc)
    te_e = apply_neighbourhood(stacked.iloc[te_i], enc)
    sc = StandardScaler().fit(tr_e[SCALE_COLS])
    def take(frame):
        full = pd.DataFrame(sc.transform(frame[SCALE_COLS]), columns=SCALE_COLS)
        return full[FULL_COLS].to_numpy(dtype=np.float32)
    model = logistic(C=best_C, class_weight="balanced")
    start = time.perf_counter()
    model.fit(take(tr_e), tr_e["noshow"])
    seconds = time.perf_counter() - start
    leak_bundle = {
        "y_train": tr_e["noshow"].to_numpy(dtype=int),
        "y_val": va_e["noshow"].to_numpy(dtype=int),
        "y_test": te_e["noshow"].to_numpy(dtype=int),
    }
    packed = _evaluate_fitted(model, take(tr_e), leak_bundle["y_train"], take(va_e), leak_bundle["y_val"], take(te_e), leak_bundle["y_test"])
    # _row_from_pack uses bundle sizes for heldout_test. Pass leak_bundle merged carefully.
    fake = dict(bundle)
    fake["y_train"] = leak_bundle["y_train"]
    fake["y_val"] = leak_bundle["y_val"]
    fake["y_test"] = leak_bundle["y_test"]
    row = _row_from_pack(
        "E10",
        "traditional",
        "logreg_balanced",
        "full_booking",
        "random_appointment",
        {"C": best_C, "class_weight": "balanced", "patients_in_train_and_test": overlap},
        fake,
        packed,
        seconds,
        f"Same features and model class as E03, but appointments are split at random. {overlap} patients appear in both the training and test portions. This is the optimistic number a row split produces.",
    )
    rows.append(row)
    log_run("E10", {"patients_overlap": overlap}, {"test": packed["test"]}, row["notes"])
    _persist(rows)

    # Deep models.
    print("fitting neural nets")
    dl_records = _run_deep(bundle, fast=fast)
    for row, model, packed, cols, kind in dl_records:
        rows.append(row)
        fitted[row["experiment_id"]] = (model, packed, cols, True, kind)
        _persist(rows)

    results = pd.DataFrame(rows)
    results.to_csv(EXP_DIR / "results.csv", index=False)

    official = results.loc[results["stage"] == "heldout_test"].copy()
    traditional_ids = ["E02", "E03", "E04", "E05", "E06", "E07", "E08"]
    trad = official.loc[official["experiment_id"].isin(traditional_ids)].sort_values("val_roc_auc", ascending=False)
    deep = official.loc[official["paradigm"] == "deep"].sort_values("val_roc_auc", ascending=False)
    best_trad_id = trad.iloc[0]["experiment_id"]
    best_deep_id = deep.iloc[0]["experiment_id"]
    # The networks sit on top of each other. A fresh TensorFlow run once
    # saved E13 because validation AUC moved by less than 0.001. That is
    # not a different model. If E11 is within 0.002 of the best validation
    # AUC, keep the small Sequential net.
    deep_tie = 0.002
    best_deep_val = float(deep.iloc[0]["val_roc_auc"])
    if best_deep_id != "E11" and (deep["experiment_id"] == "E11").any():
        e11_val = float(deep.loc[deep["experiment_id"] == "E11", "val_roc_auc"].iloc[0])
        if best_deep_val - e11_val <= deep_tie:
            print(
                f"deep models tied within {deep_tie:.3f}",
                f"(best {best_deep_id} {best_deep_val:.4f}, E11 {e11_val:.4f}); keeping E11",
            )
            best_deep_id = "E11"
    print("selected", best_trad_id, best_deep_id)

    _save_traditional(best_trad_id, fitted, bundle)
    _save_deep(best_deep_id, fitted, bundle)
    _analysis_figures(best_trad_id, best_deep_id, fitted, bundle, results, fast=fast)

    choice = {
        "best_traditional": best_trad_id,
        "best_deep": best_deep_id,
        "selection_rule": (
            "highest validation ROC-AUC within paradigm; test labels were not used to choose; "
            "if E11 is within 0.002 validation AUC of the best network, keep E11"
        ),
        "traditional_val_ranking": trad[["experiment_id", "val_roc_auc", "roc_auc"]].to_dict(orient="records"),
        "deep_val_ranking": deep[["experiment_id", "val_roc_auc", "roc_auc"]].to_dict(orient="records"),
    }
    (EXP_DIR / "model_choice.json").write_text(json.dumps(choice, indent=2, default=str))
    print(results.loc[results["stage"] == "heldout_test", ["experiment_id", "roc_auc", "pr_auc", "f1_noshow", "recall_noshow"]].to_string(index=False))
    return results


def _run_deep(bundle, fast: bool):
    import tensorflow as tf
    from tensorflow import keras

    epochs = 3 if fast else 25
    patience = 1 if fast else 5
    batch = 512
    y_train, y_val, y_test = bundle["y_train"], bundle["y_val"], bundle["y_test"]
    X_train = _X(bundle["train_df"], FULL_COLS, bundle["scaler"])
    X_val = _X(bundle["val_df"], FULL_COLS, bundle["scaler"])
    X_test = _X(bundle["test_df"], FULL_COLS, bundle["scaler"])
    cw = class_weight_dict(y_train)

    configs = [
        ("E11", "mlp_small", [64, 32], 0.2, False, 1e-3, True, "Small Sequential network, dropout 0.2, class weights, Adam at 1e-3."),
        ("E12", "mlp_deep", [128, 64, 32], 0.35, True, 1e-3, True, "Deeper Sequential network with batch norm and heavier dropout. Same learning rate as E11, so the difference is capacity and regularisation."),
        ("E13", "mlp_deep_high_lr", [128, 64, 32], 0.35, True, 1e-2, True, "Same network as E12 with the learning rate raised to 1e-2. Checks whether Adam stays stable."),
        ("E14", "mlp_deep_no_weight", [128, 64, 32], 0.35, True, 1e-3, False, "Same network as E12 with no class weights. The loss then mostly describes the majority class."),
    ]
    if fast:
        configs = configs[:2]

    records = []
    for exp_id, name, widths, drop, bn, lr, use_cw, notes in configs:
        set_seeds(SEED)
        model = build_mlp_sequential(X_train.shape[1], widths, drop, bn)
        compile_binary(model, lr)
        ds_tr = make_tfdata(X_train, y_train, batch, shuffle=True)
        ds_va = make_tfdata(X_val, y_val, batch, shuffle=False)
        ds_te = make_tfdata(X_test, y_test, batch, shuffle=False)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=patience, restore_best_weights=True
            ),
            keras.callbacks.TensorBoard(log_dir=str(TB_DIR / exp_id), histogram_freq=0),
            keras.callbacks.CSVLogger(str(CURVE_DIR / f"{exp_id}.csv")),
        ]
        start = time.perf_counter()
        history = model.fit(
            ds_tr,
            validation_data=ds_va,
            epochs=epochs,
            class_weight=cw if use_cw else None,
            callbacks=callbacks,
            verbose=0,
            shuffle=False,
        )
        seconds = time.perf_counter() - start
        # Training used tf.data. Prediction uses arrays so the row order cannot get shuffled.
        s_train = model.predict(X_train, batch_size=batch, verbose=0).ravel()
        s_val = model.predict(X_val, batch_size=batch, verbose=0).ravel()
        s_test = model.predict(X_test, batch_size=batch, verbose=0).ravel()
        packed = metrics_at_half_and_tuned(y_val, s_val, y_test, s_test, probabilities=True)
        packed["train_roc_auc"] = _auc(y_train, s_train)
        packed["scores_val"] = s_val
        packed["scores_test"] = s_test
        packed["probabilities"] = True
        params = {
            "widths": widths,
            "dropout": drop,
            "batch_norm": bn,
            "learning_rate": lr,
            "class_weight": use_cw,
            "batch_size": batch,
            "epochs_ran": len(history.history["loss"]),
            "api": "sequential",
            "input": "tf.data",
        }
        row = _row_from_pack(exp_id, "deep", name, "full_booking_scaled", "patient", params, bundle, packed, seconds, notes)
        log_run(exp_id, params, {"val_roc_auc": packed["val"]["roc_auc"], "test": packed["test"]}, notes)
        records.append((row, model, packed, FULL_COLS, "sequential"))

    # Functional embedding model.
    if not fast or True:
        set_seeds(SEED)
        cols = EMBED_NUM_COLS
        # Scaler was fit on SCALE_COLS; slice after transform.
        Xtr = _X(bundle["train_df"], cols, bundle["scaler"])
        Xva = _X(bundle["val_df"], cols, bundle["scaler"])
        Xte = _X(bundle["test_df"], cols, bundle["scaler"])
        ntr = bundle["train_df"]["neighbourhood_idx"].to_numpy(dtype=np.int32)
        nva = bundle["val_df"]["neighbourhood_idx"].to_numpy(dtype=np.int32)
        nte = bundle["test_df"]["neighbourhood_idx"].to_numpy(dtype=np.int32)
        n_emb = int(max(bundle["encoder"]["index"].values()) + 1)
        model = build_mlp_functional(Xtr.shape[1], n_neighbourhoods=n_emb, embed_dim=8)
        compile_binary(model, 1e-3)
        ds_tr = make_tfdata_embed(Xtr, ntr, y_train, batch, True)
        ds_va = make_tfdata_embed(Xva, nva, y_val, batch, False)
        ds_te = make_tfdata_embed(Xte, nte, y_test, batch, False)
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience, restore_best_weights=True),
            keras.callbacks.TensorBoard(log_dir=str(TB_DIR / "E15"), histogram_freq=0),
            keras.callbacks.CSVLogger(str(CURVE_DIR / "E15.csv")),
        ]
        start = time.perf_counter()
        history = model.fit(
            ds_tr,
            validation_data=ds_va,
            epochs=epochs,
            class_weight=cw,
            callbacks=callbacks,
            verbose=0,
            shuffle=False,
        )
        seconds = time.perf_counter() - start
        embed_inputs = lambda X, n: {"numeric": X, "neighbourhood": n}
        s_train = model.predict(embed_inputs(Xtr, ntr), batch_size=batch, verbose=0).ravel()
        s_val = model.predict(embed_inputs(Xva, nva), batch_size=batch, verbose=0).ravel()
        s_test = model.predict(embed_inputs(Xte, nte), batch_size=batch, verbose=0).ravel()
        packed = metrics_at_half_and_tuned(y_val, s_val, y_test, s_test, True)
        packed["train_roc_auc"] = _auc(y_train, s_train)
        packed["scores_val"] = s_val
        packed["scores_test"] = s_test
        packed["probabilities"] = True
        notes = "Functional API. Neighbourhood is an embedding of size 8 instead of a smoothed target rate, so the net has to learn which districts matter."
        params = {
            "embedding_dim": 8,
            "learning_rate": 1e-3,
            "class_weight": True,
            "epochs_ran": len(history.history["loss"]),
            "api": "functional",
            "input": "tf.data",
            "n_neighbourhoods": n_emb,
        }
        row = _row_from_pack("E15", "deep", "mlp_embedding", "history_plus_neighbourhood_index", "patient", params, bundle, packed, seconds, notes)
        log_run("E15", params, {"val_roc_auc": packed["val"]["roc_auc"], "test": packed["test"]}, notes)
        records.append((row, model, packed, cols, "functional"))
    return records


def _save_traditional(exp_id, fitted, bundle):
    model, packed, cols, scale = fitted[exp_id][:4]
    payload = {
        "experiment_id": exp_id,
        "model": model,
        "columns": cols,
        "scale": scale,
        "scaler": bundle["scaler"] if scale else None,
        "threshold": packed["threshold_tuned"],
        "neighbourhood_freq": bundle["encoder"]["freq"],
        "neighbourhood_te": bundle["encoder"]["te"],
        "global_mean": bundle["encoder"]["global_mean"],
        "positive_class": "noshow",
    }
    joblib.dump(payload, MODEL_DIR / "best_traditional.joblib")


def _save_deep(exp_id, fitted, bundle):
    model, packed, cols, scale, kind = fitted[exp_id]
    model.save(MODEL_DIR / "best_deep.keras")
    try:
        model.save(MODEL_DIR / "best_deep.h5")
    except Exception as exc:
        (MODEL_DIR / "best_deep_h5_error.txt").write_text(str(exc))
    sidecar = {
        "experiment_id": exp_id,
        "kind": kind,
        "columns": cols,
        "threshold": packed["threshold_tuned"],
        "scaler_mean": bundle["scaler"].mean_.tolist(),
        "scaler_scale": bundle["scaler"].scale_.tolist(),
        "scaler_columns": SCALE_COLS,
        "neighbourhood_index": bundle["encoder"]["index"],
        "global_mean": bundle["encoder"]["global_mean"],
        "neighbourhood_freq": bundle["encoder"]["freq"].to_dict(),
        "neighbourhood_te": bundle["encoder"]["te"].to_dict(),
    }
    (MODEL_DIR / "best_deep_preprocess.json").write_text(json.dumps(sidecar))


def _analysis_figures(best_trad_id, best_deep_id, fitted, bundle, results, fast: bool):
    trad_model, trad_pack, trad_cols, trad_scale = fitted[best_trad_id][:4]
    deep_model, deep_pack, deep_cols, _, deep_kind = fitted[best_deep_id]

    # Learning curves on validation, for the balanced logistic model and the chosen tree/boosted model.
    print("learning curves")
    _curve_for("E03", fitted, bundle, FIG_DIR / "learning_curve_logreg.png", "Logistic regression (E03): train vs validation AUC", fast)
    tree_id = best_trad_id if best_trad_id != "E03" else "E06"
    if tree_id in fitted and not str(tree_id).startswith("E08"):
        _curve_for(tree_id, fitted, bundle, FIG_DIR / "learning_curve_tree.png", f"{tree_id}: train vs validation AUC", fast)

    plot_dl_curves(
        CURVE_DIR,
        [("E12", "deep 1e-3"), ("E13", "deep 1e-2"), ("E11", "small")],
        FIG_DIR / "dl_learning_curves.png",
    )

    y_test = bundle["y_test"]
    score_map = {}
    for exp_id in ["E03", best_trad_id, best_deep_id]:
        if exp_id in fitted and exp_id not in score_map:
            score_map[exp_id] = fitted[exp_id][1]["scores_test"]
    # unique labels if best trad is E03
    if len(score_map) >= 2:
        plot_roc_pr(y_test, score_map, FIG_DIR)

    plot_confusion(
        y_test,
        trad_pack["scores_test"],
        trad_pack["threshold_tuned"],
        FIG_DIR / "cm_best_traditional.png",
        f"{best_trad_id} test confusion (F1 threshold from validation)",
    )
    plot_confusion(
        y_test,
        deep_pack["scores_test"],
        deep_pack["threshold_tuned"],
        FIG_DIR / "cm_best_deep.png",
        f"{best_deep_id} test confusion (F1 threshold from validation)",
    )
    if trad_pack["probabilities"]:
        plot_calibration(
            y_test,
            trad_pack["scores_test"],
            FIG_DIR / "calibration_traditional.png",
            f"{best_trad_id} reliability on the test patients",
        )

    slices = lead_slice_table(bundle["test_df"], y_test, trad_pack["scores_test"], trad_pack["threshold_tuned"])
    slices.to_csv(EXP_DIR / "slices_lead_test.csv", index=False)
    plot_slices(slices, FIG_DIR / "slices_lead.png")

    # Age slices too, for the discussion.
    age_rows = _age_slices(bundle["test_df"], y_test, trad_pack["scores_test"], trad_pack["threshold_tuned"])
    age_rows.to_csv(EXP_DIR / "slices_age_test.csv", index=False)

    print("permutation importance")
    X_val = _X(bundle["val_df"], trad_cols, bundle["scaler"] if trad_scale else None)
    y_val = bundle["y_val"]
    rng = np.random.RandomState(SEED)
    take = min(len(y_val), 1500 if fast else 6000)
    # Stratify the subsample.
    pos = np.flatnonzero(y_val == 1)
    neg = np.flatnonzero(y_val == 0)
    n_pos = max(1, int(take * y_val.mean()))
    n_neg = take - n_pos
    choose = np.concatenate([
        rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
        rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
    ])
    repeats = 2 if fast else 5
    try:
        imp = permutation_importance(
            trad_model,
            X_val[choose],
            y_val[choose],
            n_repeats=repeats,
            scoring="roc_auc",
            random_state=SEED,
            n_jobs=-1,
        )
        order = pd.DataFrame({"feature": trad_cols, "importance": imp.importances_mean})
        order.to_csv(EXP_DIR / "permutation_importance.csv", index=False)
        plot_importance(
            trad_cols,
            imp.importances_mean,
            FIG_DIR / "permutation_importance.png",
            f"What {best_trad_id} actually uses (validation, permutation)",
            "Drop in ROC-AUC when the column is shuffled",
        )
    except Exception as exc:
        (EXP_DIR / "permutation_error.txt").write_text(str(exc))

    if "E03" in fitted:
        coef_model = fitted["E03"][0]
        if hasattr(coef_model, "coef_"):
            coefs = coef_model.coef_.ravel()
            pd.DataFrame({"feature": FULL_COLS, "coefficient": coefs}).to_csv(
                EXP_DIR / "logreg_coefficients.csv", index=False
            )
            plot_importance(
                FULL_COLS,
                coefs,
                FIG_DIR / "logreg_coefficients.png",
                "Standardised logistic coefficients (E03)",
                "Coefficient on the log-odds of a no-show",
            )

    # Keep test scores for the notebook without refitting.
    payload = {"y_test": y_test}
    for exp_id, item in fitted.items():
        payload[exp_id] = item[1]["scores_test"]
    np.savez_compressed(EXP_DIR / "test_scores.npz", **payload)


def _curve_for(exp_id, fitted, bundle, path: Path, title: str, fast: bool):
    model, _, cols, scale = fitted[exp_id][:4]
    # Refit the same hyperparameters on growing patient subsets.
    params = {}
    # Use the already-built model's get_params when possible.
    if not hasattr(model, "get_params"):
        return
    fractions = [0.25, 1.0] if fast else [0.1, 0.25, 0.5, 0.75, 1.0]
    patients = np.array(bundle["train_df"]["patient_id"].astype(object).unique())
    rng = np.random.RandomState(SEED)
    rng.shuffle(patients)
    y_val = bundle["y_val"]
    rows = []
    for frac in fractions:
        n_keep = max(30, int(len(patients) * frac))
        keep = set(patients[:n_keep])
        part = bundle["train_df"].loc[bundle["train_df"]["patient_id"].isin(keep)]
        y = part["noshow"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        if scale:
            local_scaler = StandardScaler().fit(part[SCALE_COLS])
            def take(frame, sc=local_scaler):
                full = pd.DataFrame(sc.transform(frame[SCALE_COLS]), columns=SCALE_COLS)
                return full[cols].to_numpy(dtype=np.float32)
            X = take(part)
            X_val = take(bundle["val_df"])
        else:
            X = _X(part, cols, None)
            X_val = _X(bundle["val_df"], cols, None)
        clone = model.__class__(**model.get_params())
        if exp_id in {"E06", "E07"}:
            clone.fit(X, y, sample_weight=sample_weight_balanced(y))
        else:
            clone.fit(X, y)
        s_tr, _ = _positive_scores(clone, X)
        s_va, _ = _positive_scores(clone, X_val)
        rows.append({"fraction": frac, "train_rows": int(len(part)), "train_auc": _auc(y, s_tr), "val_auc": _auc(y_val, s_va)})
        print("  curve", exp_id, frac, rows[-1])
    if rows:
        pd.DataFrame(rows).to_csv(EXP_DIR / f"learning_curve_{exp_id}.csv", index=False)
        plot_learning_curve(rows, path, title)


def _age_slices(df, y, scores, threshold) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tmp = df.reset_index(drop=True)
    bins = pd.cut(tmp["age"], bins=[-0.1, 17, 34, 49, 64, 200], labels=["0-17", "18-34", "35-49", "50-64", "65+"])
    rows = []
    for name in bins.cat.categories:
        mask = (bins == name).to_numpy()
        if mask.sum() == 0:
            continue
        yt, pr = y[mask], pred[mask]
        tp = int(((pr == 1) & (yt == 1)).sum())
        fp = int(((pr == 1) & (yt == 0)).sum())
        fn = int(((pr == 0) & (yt == 1)).sum())
        rows.append(
            {
                "bin": str(name),
                "n": int(mask.sum()),
                "prevalence": float(yt.mean()),
                "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
                "recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                "fn": fn,
                "fp": fp,
            }
        )
    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Small sample, for checking that the script runs.")
    args = parser.parse_args()
    run(fast=args.fast)
