#!/usr/bin/env python3
"""Pure probability-calibration functions for Task 10.

Importing this module has no filesystem or project-model side effects.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


METHODS = ("identity", "logistic", "isotonic")
COMPLEXITY = {"identity": 0, "logistic": 1, "isotonic": 2}
EPS = np.finfo(np.float64).eps


class CalibrationContractError(RuntimeError):
    """Raised when a shared calibration contract is violated."""


class RawIdentityPredictionError(CalibrationContractError):
    """Raised when the raw identity probability vector violates its contract."""


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON payload contains NaN or Infinity")
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA or value is None:
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8")); digest.update(b"\n")
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def validate_probabilities(probability: np.ndarray, expected_rows: int | None = None) -> dict[str, Any]:
    values = np.asarray(probability)
    if values.ndim != 1:
        raise CalibrationContractError("Probabilities must be one-dimensional")
    if values.dtype != np.dtype("float64"):
        raise CalibrationContractError(f"Probabilities must be float64, observed {values.dtype}")
    if expected_rows is not None and len(values) != expected_rows:
        raise CalibrationContractError(f"Probability row count {len(values)} != {expected_rows}")
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    out_of_bounds = int(np.count_nonzero((values < 0) | (values > 1)))
    if nonfinite or out_of_bounds:
        raise CalibrationContractError(f"Invalid probabilities: nonfinite={nonfinite}, out_of_bounds={out_of_bounds}")
    return {
        "rows": len(values), "dtype": str(values.dtype), "nonfinite_count": nonfinite,
        "out_of_bounds_count": out_of_bounds, "exact_zero_count": int(np.count_nonzero(values == 0)),
        "exact_one_count": int(np.count_nonzero(values == 1)),
    }


def validate_raw_identity_predictions(probability: np.ndarray, expected_rows: int | None = None) -> dict[str, Any]:
    """Validate raw base probabilities and expose a distinct fatal contract error."""
    try:
        return validate_probabilities(probability, expected_rows)
    except CalibrationContractError as exc:
        raise RawIdentityPredictionError(f"INVALID_RAW_IDENTITY_PREDICTIONS: {exc}") from exc


def make_common_folds(y: np.ndarray, n_splits: int = 5, seed: int = 20260921) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int8)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise CalibrationContractError("Fold labels must be a complete binary vector")
    folds = np.full(len(labels), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_id, (_, heldout) in enumerate(splitter.split(np.zeros(len(labels), dtype=np.int8), labels)):
        if np.any(folds[heldout] != -1):
            raise CalibrationContractError("Cross-fitting held-out folds overlap")
        folds[heldout] = fold_id
    if np.any(folds < 0) or set(np.unique(folds)) != set(range(n_splits)):
        raise CalibrationContractError("Cross-fitting folds do not cover every row exactly once")
    return folds


def fold_summary(folds: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    labels = np.asarray(y, dtype=np.int8); fold_ids = np.asarray(folds, dtype=np.int8)
    rows = []
    for fold_id in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold_id
        rows.append({"fold_id": int(fold_id), "rows": int(mask.sum()), "positive_count": int(labels[mask].sum()),
                     "negative_count": int(mask.sum() - labels[mask].sum())})
    return pd.DataFrame(rows)


def endpoint_protected_logit(probability: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    raw = np.asarray(probability, dtype=np.float64)
    protected = np.clip(raw, EPS, 1.0 - EPS)
    z = np.log(protected) - np.log1p(-protected)
    return z, {
        "exact_zero_count": int(np.count_nonzero(raw == 0)),
        "exact_one_count": int(np.count_nonzero(raw == 1)),
        "protected_endpoint_count": int(np.count_nonzero(protected != raw)),
    }


def logistic_estimator() -> LogisticRegression:
    return LogisticRegression(penalty=None, solver="lbfgs", fit_intercept=True, class_weight=None, max_iter=10000, tol=1e-12)


def fit_logistic_calibrator(probability: np.ndarray, y: np.ndarray) -> tuple[LogisticRegression, dict[str, Any]]:
    z, endpoint = endpoint_protected_logit(probability)
    model = logistic_estimator()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(z.reshape(-1, 1), np.asarray(y, dtype=np.int8))
    convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    alpha = float(model.intercept_[0]); beta = float(model.coef_[0, 0]); iterations = int(model.n_iter_[0])
    converged = not convergence_warnings and iterations < model.max_iter
    valid = converged and math.isfinite(alpha) and math.isfinite(beta) and beta > 0
    metadata = {**endpoint, "alpha": alpha, "beta": beta, "iterations": iterations,
                "converged": converged, "convergence_warnings": convergence_warnings, "valid": valid}
    if not valid:
        raise CalibrationContractError(f"Invalid Logistic calibrator: {metadata}")
    return model, metadata


def predict_logistic_calibrator(model: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    z, _ = endpoint_protected_logit(probability)
    values = model.predict_proba(z.reshape(-1, 1))[:, list(model.classes_).index(1)].astype(np.float64)
    validate_probabilities(values, len(z))
    return values


def fit_isotonic_calibrator(probability: np.ndarray, y: np.ndarray) -> tuple[IsotonicRegression, dict[str, Any]]:
    raw = np.asarray(probability, dtype=np.float64)
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(raw, np.asarray(y, dtype=np.int8))
    x = np.asarray(model.X_thresholds_, dtype=np.float64); y_threshold = np.asarray(model.y_thresholds_, dtype=np.float64)
    valid = (len(x) > 0 and np.isfinite(x).all() and np.isfinite(y_threshold).all() and
             np.all(np.diff(x) > 0) and np.all(np.diff(y_threshold) >= 0) and
             np.all((y_threshold >= 0) & (y_threshold <= 1)))
    if not valid:
        raise CalibrationContractError("Invalid Isotonic thresholds")
    low = float(model.predict(np.array([raw.min() - 1.0]))[0]); high = float(model.predict(np.array([raw.max() + 1.0]))[0])
    clipping_ok = low == float(y_threshold[0]) and high == float(y_threshold[-1])
    if not clipping_ok:
        raise CalibrationContractError("Isotonic out-of-range clipping check failed")
    return model, {"threshold_count": len(x), "x_min": float(x[0]), "x_max": float(x[-1]),
                   "y_min": float(y_threshold[0]), "y_max": float(y_threshold[-1]),
                   "nondecreasing": True, "out_of_range_clipping_ok": clipping_ok, "valid": True}


def predict_isotonic_calibrator(model: IsotonicRegression, probability: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict(np.asarray(probability, dtype=np.float64)), dtype=np.float64)
    validate_probabilities(values, len(probability))
    return values


def apply_candidate(method: str, calibrator: Any, probability: np.ndarray) -> np.ndarray:
    if method == "identity":
        values = np.asarray(probability, dtype=np.float64).copy()
    elif method == "logistic":
        values = predict_logistic_calibrator(calibrator, probability)
    elif method == "isotonic":
        values = predict_isotonic_calibrator(calibrator, probability)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    validate_probabilities(values, len(probability))
    return values


def official_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int8); values = np.asarray(probability, dtype=np.float64)
    audit = validate_probabilities(values, len(labels))
    if set(np.unique(labels)) != {0, 1}:
        raise CalibrationContractError("Metrics require both binary classes")
    return {
        "N": len(labels), "positive_count": int(labels.sum()), "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, values)),
        "average_precision": float(average_precision_score(labels, values)),
        "log_loss": float(log_loss(labels, values, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, values)),
        "mean_probability": float(values.mean()), "min_probability": float(values.min()), "max_probability": float(values.max()),
        "exact_zero_count": audit["exact_zero_count"], "exact_one_count": audit["exact_one_count"],
        "nonfinite_count": audit["nonfinite_count"], "out_of_bounds_count": audit["out_of_bounds_count"],
    }


def fold_metric_rows(model_id: str, method: str, folds: np.ndarray, y: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for fold_id in sorted(np.unique(folds)):
        mask = folds == fold_id
        rows.append({"model_id": model_id, "method": method, "estimate_role": "OOF_CALIBRATION_DEVELOPMENT_ESTIMATE",
                     "fold_id": int(fold_id), **official_metrics(y[mask], probability[mask])})
    return rows


def validate_oof(probability: np.ndarray, assigned: np.ndarray, expected_rows: int) -> dict[str, Any]:
    if len(assigned) != expected_rows or not np.all(np.asarray(assigned) == 1):
        raise CalibrationContractError("Each OOF row must be assigned exactly once")
    return validate_probabilities(np.asarray(probability), expected_rows)


def choose_numeric_best(metrics: Mapping[str, float], tie_tolerance: float = 1e-12) -> str:
    valid = {method: float(value) for method, value in metrics.items() if method in METHODS and math.isfinite(float(value))}
    if not valid:
        raise CalibrationContractError("Numeric-best selection requires at least one finite admissible method")
    minimum = min(valid.values())
    upper = np.nextafter(minimum + tie_tolerance, np.inf)
    tied = [method for method in METHODS if method in valid and valid[method] <= upper]
    return tied[0]


def bootstrap_draws(y: np.ndarray, replicates: int = 2000, seed: int = 20260922) -> Iterable[tuple[int, np.ndarray]]:
    labels = np.asarray(y, dtype=np.int8)
    positive = np.flatnonzero(labels == 1); negative = np.flatnonzero(labels == 0)
    generator = np.random.Generator(np.random.PCG64(seed))
    for replicate_id in range(replicates):
        pos_draw = generator.choice(positive, size=len(positive), replace=True)
        neg_draw = generator.choice(negative, size=len(negative), replace=True)
        yield replicate_id, np.concatenate([pos_draw, neg_draw])


def paired_stratified_bootstrap(
    y: np.ndarray, probabilities: Mapping[tuple[str, str], np.ndarray], numeric_best: Mapping[str, str] | None = None,
    replicates: int = 2000, seed: int = 20260922, batch_size: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = np.asarray(y, dtype=np.int8)
    pairs = list(probabilities)
    if any(len(probabilities[pair]) != len(labels) for pair in pairs):
        raise CalibrationContractError("Bootstrap probability vectors are not row aligned")
    ll = np.column_stack([
        -(labels * np.log(np.clip(probabilities[pair], EPS, 1 - EPS)) + (1 - labels) * np.log1p(-np.clip(probabilities[pair], EPS, 1 - EPS)))
        for pair in pairs
    ]).astype(np.float64)
    brier = np.column_stack([(np.asarray(probabilities[pair], np.float64) - labels) ** 2 for pair in pairs]).astype(np.float64)
    positive = np.flatnonzero(labels == 1); negative = np.flatnonzero(labels == 0)
    generator = np.random.Generator(np.random.PCG64(seed))
    ll_results = np.empty((replicates, len(pairs)), dtype=np.float64)
    brier_results = np.empty_like(ll_results)
    for batch_start in range(0, replicates, batch_size):
        stop = min(replicates, batch_start + batch_size)
        counts = np.zeros((stop - batch_start, len(labels)), dtype=np.float64)
        for row in range(stop - batch_start):
            pos_draw = generator.choice(positive, size=len(positive), replace=True)
            neg_draw = generator.choice(negative, size=len(negative), replace=True)
            counts[row] = np.bincount(np.concatenate([pos_draw, neg_draw]), minlength=len(labels))
        ll_results[batch_start:stop] = counts @ ll / len(labels)
        brier_results[batch_start:stop] = counts @ brier / len(labels)
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    rows: list[dict[str, Any]] = []
    for replicate_id in range(replicates):
        for index, (model_id, method) in enumerate(pairs):
            identity_index = pair_index[(model_id, "identity")]
            rows.append({
                "replicate_id": replicate_id, "model_id": model_id, "method": method,
                "bootstrap_log_loss": float(ll_results[replicate_id, index]),
                "bootstrap_brier_score": float(brier_results[replicate_id, index]),
                "log_loss": float(ll_results[replicate_id, index]),
                "brier_score": float(brier_results[replicate_id, index]),
                "brier_difference_vs_identity": float(brier_results[replicate_id, index] - brier_results[replicate_id, identity_index]),
                "admissible_numeric_best_reference": None if numeric_best is None else numeric_best[model_id],
                "log_loss_difference_vs_admissible_numeric_best": None if numeric_best is None else float(
                    ll_results[replicate_id, index] - ll_results[replicate_id, pair_index[(model_id, numeric_best[model_id])]]
                ),
                "estimate_role": "OOF_CALIBRATION_DEVELOPMENT_ESTIMATE",
            })
    metadata = {
        "generator": "np.random.Generator(np.random.PCG64(20260922))", "algorithm": "PCG64",
        "seed": seed, "replicates": replicates, "positive_count": len(positive), "negative_count": len(negative),
        "positive_position_fingerprint": sha256_lines(positive), "negative_position_fingerprint": sha256_lines(negative),
        "pair_order": [list(pair) for pair in pairs], "batch_size_for_linear_algebra_only": batch_size,
    }
    return pd.DataFrame(rows), metadata


def annotate_bootstrap_references(frame: pd.DataFrame, references: Mapping[str, str]) -> pd.DataFrame:
    """Attach the post-hard-admissibility Log-loss reference to absolute bootstrap metrics."""
    required = {"replicate_id", "model_id", "method", "bootstrap_log_loss", "bootstrap_brier_score", "brier_difference_vs_identity"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CalibrationContractError(f"Bootstrap table missing required columns: {missing}")
    result = frame.copy()
    result["admissible_numeric_best_reference"] = result["model_id"].map(references)
    if result["admissible_numeric_best_reference"].isna().any():
        raise CalibrationContractError("Missing admissible numeric-best bootstrap reference")
    reference = result[["replicate_id", "model_id", "method", "bootstrap_log_loss"]].rename(
        columns={"method": "admissible_numeric_best_reference", "bootstrap_log_loss": "reference_bootstrap_log_loss"}
    )
    result = result.drop(columns=["log_loss_difference_vs_admissible_numeric_best"], errors="ignore").merge(
        reference,
        on=["replicate_id", "model_id", "admissible_numeric_best_reference"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if result["reference_bootstrap_log_loss"].isna().any():
        raise CalibrationContractError("Bootstrap reference rows are incomplete")
    result["log_loss_difference_vs_admissible_numeric_best"] = (
        result["bootstrap_log_loss"] - result["reference_bootstrap_log_loss"]
    )
    result = result.drop(columns="reference_bootstrap_log_loss")
    return result.sort_values(["replicate_id", "model_id", "method"], kind="stable").reset_index(drop=True)


def paired_standard_error(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2 or not np.isfinite(array).all():
        raise CalibrationContractError("Paired standard error requires at least two finite replicates")
    return float(np.std(array, ddof=1))


def log_loss_one_se_pass(delta: float, paired_se: float) -> bool:
    return bool(float(delta) <= float(paired_se) + 1e-12)


def brier_guardrail_pass(delta: float, paired_se: float) -> bool:
    return bool(float(delta) <= float(paired_se))


def logistic_ranking_guardrail(raw_auc: float, raw_ap: float, mapped_auc: float, mapped_ap: float, beta: float, tolerance: float = 1e-10) -> tuple[bool, float, float]:
    auc_change = float(mapped_auc - raw_auc); ap_change = float(mapped_ap - raw_ap)
    auc_lower = np.nextafter(raw_auc - tolerance, -np.inf); auc_upper = np.nextafter(raw_auc + tolerance, np.inf)
    ap_lower = np.nextafter(raw_ap - tolerance, -np.inf); ap_upper = np.nextafter(raw_ap + tolerance, np.inf)
    passed = beta > 0 and auc_lower <= mapped_auc <= auc_upper and ap_lower <= mapped_ap <= ap_upper
    return bool(passed), auc_change, ap_change


def isotonic_ranking_guardrail(raw_auc: float, raw_ap: float, mapped_auc: float, mapped_ap: float, max_drop: float = 0.0005) -> tuple[bool, float, float]:
    auc_change = float(mapped_auc - raw_auc); ap_change = float(mapped_ap - raw_ap)
    # Decimal round-trip construction makes the specified decimal boundary
    # inclusive while the next lower float remains a strict failure.
    auc_floor = float(Decimal(str(raw_auc)) - Decimal(str(max_drop)))
    ap_floor = float(Decimal(str(raw_ap)) - Decimal(str(max_drop)))
    passed = mapped_auc >= auc_floor and mapped_ap >= ap_floor
    return bool(passed), auc_change, ap_change


def stability_warning(fold_rows: pd.DataFrame, method: str) -> str:
    method_rows = fold_rows[fold_rows.method.eq(method)].sort_values("fold_id")
    identity = fold_rows[fold_rows.method.eq("identity")].sort_values("fold_id")
    if len(method_rows) != len(identity) or len(method_rows) == 0:
        return "NOT_CHECKABLE"
    improvements = int(np.count_nonzero(method_rows.log_loss.to_numpy() < identity.log_loss.to_numpy()))
    return "STABILITY_WARNING" if improvements <= 1 else ""


def mechanical_selection(
    rows: list[dict[str, Any]], bootstrap: pd.DataFrame | None = None, tie_tolerance: float = 1e-12,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply the v2 constrained selector in its fixed, non-circular order."""
    by_method = {row["method"]: dict(row) for row in rows}
    if "identity" not in by_method or not by_method["identity"].get("fit_and_probability_valid", False):
        raise RawIdentityPredictionError("INVALID_RAW_IDENTITY_PREDICTIONS: identity is not fit/probability valid")
    annotated: list[dict[str, Any]] = []
    for method in METHODS:
        row = by_method.get(method)
        if row is None:
            continue
        row["hard_admissible"] = bool(
            row.get("fit_and_probability_valid", False)
            and row.get("passes_brier_guardrail", False)
            and row.get("passes_ranking_guardrail", False)
        )
        row["eligible"] = False
        row["selected"] = False
        row["selection_reason"] = "PENDING_V2_SELECTION"
        annotated.append(row)
    hard = [row for row in annotated if row["hard_admissible"]]
    if not hard:
        raise CalibrationContractError("EMPTY_HARD_ADMISSIBLE_SET selector implementation defect")
    reference = choose_numeric_best(
        {row["method"]: row["oof_log_loss"] for row in hard}, tie_tolerance
    )
    selected = None
    for row in annotated:
        method = row["method"]
        row["admissible_numeric_best_reference"] = reference
        row["numeric_best_reference"] = reference
        if not row["hard_admissible"]:
            row["delta_log_loss_vs_admissible_numeric_best"] = None
            row["paired_se_log_loss_vs_admissible_numeric_best"] = None
            row["passes_log_loss_one_se"] = False
            if not row.get("fit_and_probability_valid", False):
                row["selection_reason"] = row.get("fit_failure_reason") or "INVALID_FIT_OR_PROBABILITY"
            else:
                row["selection_reason"] = "INELIGIBLE_HARD_CONSTRAINT"
            continue
        delta = float(row["oof_log_loss"]) - float(by_method[reference]["oof_log_loss"])
        if method == reference:
            se = 0.0
        elif bootstrap is not None:
            candidate = bootstrap[(bootstrap["method"] == method)].sort_values("replicate_id", kind="stable")
            reference_rows = bootstrap[(bootstrap["method"] == reference)].sort_values("replicate_id", kind="stable")
            if len(candidate) != len(reference_rows) or not np.array_equal(
                candidate["replicate_id"].to_numpy(), reference_rows["replicate_id"].to_numpy()
            ):
                raise CalibrationContractError("Bootstrap replicate pairing mismatch")
            se = paired_standard_error(
                candidate["bootstrap_log_loss"].to_numpy(np.float64)
                - reference_rows["bootstrap_log_loss"].to_numpy(np.float64)
            )
        else:
            value = row.get("paired_se_log_loss_vs_admissible_numeric_best")
            if value is None:
                raise CalibrationContractError("V2 selection requires paired Log-loss SE evidence")
            se = float(value)
        row["delta_log_loss_vs_admissible_numeric_best"] = delta
        row["paired_se_log_loss_vs_admissible_numeric_best"] = se
        row["delta_log_loss_vs_numeric_best"] = delta
        row["paired_se_log_loss_vs_numeric_best"] = se
        row["passes_log_loss_one_se"] = log_loss_one_se_pass(delta, se)
        row["eligible"] = bool(row["passes_log_loss_one_se"])
        row["selection_reason"] = "ELIGIBLE_NOT_SIMPLEST" if row["eligible"] else "OUTSIDE_ONE_SE_BAND"
        if row["eligible"] and selected is None:
            selected = method
            row["selected"] = True
            row["selection_reason"] = "SELECTED_SIMPLEST_ELIGIBLE"
    if selected is None:
        raise CalibrationContractError("NO_ONE_SE_ELIGIBLE_METHOD selector implementation defect")
    return selected, annotated


def verify_v2_selection_table(selection: pd.DataFrame, bootstrap: pd.DataFrame) -> dict[str, str]:
    """Independently reconstruct v2 references, one-SE eligibility, and selections."""
    required = {
        "model_id", "method", "fit_and_probability_valid", "oof_log_loss", "passes_brier_guardrail",
        "passes_ranking_guardrail", "hard_admissible", "admissible_numeric_best_reference",
        "delta_log_loss_vs_admissible_numeric_best", "paired_se_log_loss_vs_admissible_numeric_best",
        "passes_log_loss_one_se", "eligible", "selected",
    }
    missing = sorted(required - set(selection.columns))
    if missing:
        raise CalibrationContractError(f"Selection table missing v2 columns: {missing}")
    verified: dict[str, str] = {}
    for model_id in selection.model_id.drop_duplicates().tolist():
        saved_rows = selection[selection.model_id.eq(model_id)]
        if set(saved_rows.method) != set(METHODS) or len(saved_rows) != len(METHODS):
            raise CalibrationContractError(f"Candidate row count mismatch: {model_id}")
        if int(saved_rows.selected.astype(bool).sum()) != 1:
            raise CalibrationContractError(f"Selected method count mismatch: {model_id}")
        input_rows = []
        for saved in saved_rows.itertuples(index=False):
            input_rows.append({
                "model_id": model_id, "method": saved.method,
                "fit_and_probability_valid": bool(saved.fit_and_probability_valid),
                "oof_log_loss": saved.oof_log_loss,
                "passes_brier_guardrail": bool(saved.passes_brier_guardrail),
                "passes_ranking_guardrail": bool(saved.passes_ranking_guardrail),
                "stability_warning": getattr(saved, "stability_warning", ""),
            })
        model_bootstrap = bootstrap[bootstrap.model_id.eq(model_id)]
        selected, reconstructed = mechanical_selection(input_rows, model_bootstrap)
        reconstructed_by_method = {row["method"]: row for row in reconstructed}
        for saved in saved_rows.itertuples(index=False):
            expected = reconstructed_by_method[saved.method]
            for field in ("hard_admissible", "passes_log_loss_one_se", "eligible", "selected"):
                if bool(getattr(saved, field)) != bool(expected[field]):
                    raise CalibrationContractError(f"Inconsistent method-selection row: {model_id}/{saved.method}/{field}")
            if saved.admissible_numeric_best_reference != expected["admissible_numeric_best_reference"]:
                raise CalibrationContractError(f"Ineligible or inconsistent numerical reference: {model_id}/{saved.method}")
            for field in ("delta_log_loss_vs_admissible_numeric_best", "paired_se_log_loss_vs_admissible_numeric_best"):
                saved_value = getattr(saved, field)
                expected_value = expected[field]
                if expected_value is None and pd.isna(saved_value):
                    continue
                if expected_value is None or pd.isna(saved_value) or abs(float(saved_value) - float(expected_value)) > 1e-15:
                    raise CalibrationContractError(f"Inconsistent method-selection statistic: {model_id}/{saved.method}/{field}")
        verified[model_id] = selected
    return verified


def safe_sigmoid(value: np.ndarray | float) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64); result = np.empty_like(x)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive]); result[~positive] = exp_x / (1.0 + exp_x)
    return result


def calibration_in_the_large(y: np.ndarray, probability: np.ndarray) -> float:
    labels = np.asarray(y, dtype=np.float64); z, _ = endpoint_protected_logit(probability)
    def function(intercept: float) -> float:
        return float(np.sum(safe_sigmoid(intercept + z) - labels, dtype=np.float64))
    lower, upper = -1.0, 1.0; f_lower, f_upper = function(lower), function(upper)
    while f_lower * f_upper > 0 and max(abs(lower), abs(upper)) < 128:
        lower *= 2; upper *= 2; f_lower, f_upper = function(lower), function(upper)
    if f_lower * f_upper > 0:
        raise CalibrationContractError("Calibration intercept failed to bracket a root")
    for _ in range(500):
        midpoint = (lower + upper) / 2; f_mid = function(midpoint)
        if abs(f_mid) <= 1e-10 or upper - lower <= 1e-12:
            return float(midpoint)
        if f_lower * f_mid <= 0:
            upper, f_upper = midpoint, f_mid
        else:
            lower, f_lower = midpoint, f_mid
    raise CalibrationContractError("Calibration intercept bisection did not converge")


def calibration_joint_intercept_slope(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    z, _ = endpoint_protected_logit(probability)
    model = logistic_estimator()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(z.reshape(-1, 1), np.asarray(y, dtype=np.int8))
    if any(issubclass(item.category, ConvergenceWarning) for item in caught) or int(model.n_iter_[0]) >= model.max_iter:
        raise CalibrationContractError("Joint calibration intercept/slope fit did not converge")
    intercept = float(model.intercept_[0]); slope = float(model.coef_[0, 0])
    if not math.isfinite(intercept) or not math.isfinite(slope):
        raise CalibrationContractError("Joint calibration intercept/slope is nonfinite")
    return intercept, slope


def tie_preserving_bins(y: np.ndarray, probability: np.ndarray, requested_bins: int) -> pd.DataFrame:
    labels = np.asarray(y, dtype=np.int8); values = np.asarray(probability, dtype=np.float64)
    validate_probabilities(values, len(labels))
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    candidates = cumulative[:-1]
    selected: list[int] = []
    for boundary in range(1, requested_bins):
        target = boundary * len(values) / requested_bins
        if not len(candidates):
            break
        distances = np.abs(candidates - target)
        minimum = distances.min()
        chosen = int(candidates[np.flatnonzero(distances == minimum)[0]])
        selected.append(chosen)
    selected = sorted(set(selected))
    unique_group = np.searchsorted(np.asarray(selected, dtype=np.int64), cumulative - counts, side="right")
    row_group = unique_group[inverse]
    rows = []
    actual_bins = int(row_group.max() + 1)
    for group_id in range(actual_bins):
        mask = row_group == group_id
        observed = float(labels[mask].mean()); mean_probability = float(values[mask].mean())
        rows.append({"requested_bin_count": requested_bins, "actual_bin_count": actual_bins, "bin_id": group_id + 1,
                     "rows": int(mask.sum()), "positive_count": int(labels[mask].sum()), "observed_rate": observed,
                     "mean_probability": mean_probability, "min_probability": float(values[mask].min()),
                     "max_probability": float(values[mask].max()), "prediction_minus_observed_gap": mean_probability - observed})
    return pd.DataFrame(rows)


def bin_diagnostics(bins: pd.DataFrame) -> tuple[float, float]:
    total = float(bins.rows.sum()); gaps = bins.prediction_minus_observed_gap.abs().to_numpy(np.float64)
    ece = float(np.sum(bins.rows.to_numpy(np.float64) / total * gaps))
    return ece, float(gaps.max())


def top_risk_summary(y: np.ndarray, probability: np.ndarray, shares: Iterable[float] = (0.01, 0.05, 0.10)) -> list[dict[str, Any]]:
    labels = np.asarray(y, dtype=np.int8); values = np.asarray(probability, dtype=np.float64)
    order = np.argsort(-values, kind="stable"); sorted_values = values[order]
    rows = []
    for share in shares:
        k = int(math.ceil(float(share) * len(values))); cutoff = float(sorted_values[k - 1]); mask = values >= cutoff
        rows.append({"target_share": float(share), "actual_rows": int(mask.sum()), "actual_share": float(mask.mean()),
                     "cutoff_probability": cutoff, "mean_probability": float(values[mask].mean()),
                     "observed_rate": float(labels[mask].mean()), "positive_count": int(labels[mask].sum())})
    return rows


def metric_with_calibration_diagnostics(y: np.ndarray, probability: np.ndarray, protected_endpoint_count: int = 0) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    metrics = official_metrics(y, probability)
    metrics["protected_endpoint_count"] = int(protected_endpoint_count)
    metrics["calibration_in_the_large_intercept"] = calibration_in_the_large(y, probability)
    intercept, slope = calibration_joint_intercept_slope(y, probability)
    metrics["calibration_joint_intercept"] = intercept; metrics["calibration_slope"] = slope
    bins10 = tie_preserving_bins(y, probability, 10); bins20 = tie_preserving_bins(y, probability, 20)
    metrics["ece_10"], metrics["max_abs_bin_gap_10"] = bin_diagnostics(bins10)
    metrics["ece_20"], metrics["max_abs_bin_gap_20"] = bin_diagnostics(bins20)
    return metrics, bins10, bins20


def completion_check(status_path: Path, config_hash: str, required_artifacts: Iterable[Path]) -> tuple[bool, str]:
    if not status_path.is_file():
        return False, "status file missing"
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"status unreadable: {type(exc).__name__}"
    if payload.get("status") != "COMPLETE" or payload.get("config_hash") != config_hash:
        return False, "status/config mismatch"
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False, "artifact manifest missing or empty"
    by_path = {item.get("path"): item for item in artifacts if isinstance(item, dict)}
    for path in required_artifacts:
        item = by_path.get(str(path))
        if not item or not path.is_file() or item.get("sha256") != sha256_file(path):
            return False, f"missing or hash-invalid artifact: {path}"
    return True, "complete hash-valid artifacts"
