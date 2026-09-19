#!/usr/bin/env python3
"""Strict Task 09 validation and state-transition helpers.

These helpers are deliberately independent of model fitting so they can be
exercised with small fixtures and reused by post-training verification.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


class SharedContractError(RuntimeError):
    """A data or membership contract failure that blocks dependent work."""


class FamilyImplementationError(RuntimeError):
    """A diagnosed family-wide implementation failure."""


class VerificationError(RuntimeError):
    """A saved artifact or inference verification failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_epoch_mean(batch_mean_losses: Iterable[float], batch_sizes: Iterable[int]) -> float:
    losses = np.asarray(list(batch_mean_losses), dtype=np.float64)
    sizes = np.asarray(list(batch_sizes), dtype=np.int64)
    if losses.ndim != 1 or sizes.ndim != 1 or len(losses) != len(sizes) or not len(losses):
        raise ValueError("Batch losses and sizes must be nonempty aligned vectors")
    if not np.isfinite(losses).all() or np.any(sizes <= 0):
        raise FloatingPointError("Batch losses must be finite and batch sizes positive")
    result = float(np.sum(losses * sizes, dtype=np.float64) / np.sum(sizes, dtype=np.int64))
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite sample-weighted epoch loss")
    return result


@dataclass
class EarlyStoppingTracker:
    patience_limit: int = 5
    min_reset_delta: float = 0.0001
    best_ap: float = -math.inf
    best_log_loss: float = math.inf
    best_epoch: int = 0
    patience_reference: float = -math.inf
    patience_count: int = 0

    def update(self, *, epoch: int, average_precision: float, log_loss: float) -> dict[str, Any]:
        if epoch <= 0 or not math.isfinite(average_precision) or not math.isfinite(log_loss):
            raise ValueError("Finite metrics and a positive epoch are required")
        is_best = average_precision > self.best_ap or (
            average_precision == self.best_ap
            and (log_loss < self.best_log_loss or (log_loss == self.best_log_loss and epoch < self.best_epoch))
        )
        if is_best:
            self.best_ap = average_precision
            self.best_log_loss = log_loss
            self.best_epoch = epoch
        reset = average_precision >= self.patience_reference + self.min_reset_delta
        if reset:
            self.patience_reference = average_precision
            self.patience_count = 0
        else:
            self.patience_count += 1
        return {
            "is_best": is_best,
            "patience_reset": reset,
            "patience_count": self.patience_count,
            "should_stop": self.patience_count >= self.patience_limit,
            "best_epoch": self.best_epoch,
            "best_ap": self.best_ap,
            "patience_reference": self.patience_reference,
        }


def validate_prediction_frame(
    predictions: pd.DataFrame,
    authoritative_tuning: pd.DataFrame,
    tuning_labels: np.ndarray,
    expected_model_ids: list[str],
    expected_probability_columns: dict[str, str],
) -> list[dict[str, Any]]:
    expected_columns = [
        "case_id", "base_order", "target",
        *[expected_probability_columns[model_id] for model_id in expected_model_ids],
    ]
    rows: list[dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, expected: Any, reason: str) -> None:
        rows.append({
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "actual": actual,
            "expected": expected,
            "reason": reason,
        })
        if not passed:
            raise VerificationError(f"{name}: {reason}; actual={actual!r}, expected={expected!r}")

    check("prediction_schema", list(predictions.columns) == expected_columns, list(predictions.columns), expected_columns,
          "Merged tuning predictions must have the exact ordered schema")
    check("prediction_row_count", len(predictions) == len(authoritative_tuning), len(predictions), len(authoritative_tuning),
          "Every authoritative tuning application must appear exactly once")
    check("prediction_unique_case_id", predictions["case_id"].notna().all() and predictions["case_id"].is_unique,
          int(predictions["case_id"].nunique(dropna=True)), len(predictions), "case_id must be complete and unique")
    check("prediction_case_id_order", np.array_equal(predictions["case_id"].to_numpy(), authoritative_tuning["case_id"].to_numpy()),
          "ordered comparison", "exact match", "case_id order must match authoritative tuning membership")
    check("prediction_base_order", np.array_equal(predictions["base_order"].to_numpy(), authoritative_tuning["base_order"].to_numpy()),
          "ordered comparison", "exact match", "base_order must match authoritative tuning membership")
    labels = predictions["target"].to_numpy()
    check("prediction_target_dtype", pd.api.types.is_integer_dtype(predictions["target"].dtype), str(predictions["target"].dtype),
          "integer", "The stored target must be binary integer data")
    check("prediction_target_alignment", np.array_equal(labels.astype(np.int8), np.asarray(tuning_labels, dtype=np.int8)),
          "ordered comparison", "exact match", "Stored tuning labels must match the authorized source")
    check("prediction_target_binary", set(np.unique(labels)) == {0, 1}, sorted(pd.unique(labels).tolist()), [0, 1],
          "Only binary tuning labels are valid")
    for model_id in expected_model_ids:
        column = expected_probability_columns[model_id]
        values = predictions[column].to_numpy()
        check(f"{model_id}__probability_dtype", predictions[column].dtype == np.dtype("float64"),
              str(predictions[column].dtype), "float64", "Saved selected-model probabilities must be float64")
        check(f"{model_id}__probability_complete", predictions[column].notna().all(),
              int(predictions[column].isna().sum()), 0, "Probability column must have no missing rows")
        check(f"{model_id}__probability_finite", np.isfinite(values).all(), int(np.count_nonzero(~np.isfinite(values))), 0,
              "Probability column must be finite")
        check(f"{model_id}__probability_bounds", bool(np.all((values >= 0) & (values <= 1))),
              f"min={np.nanmin(values)}, max={np.nanmax(values)}", "[0, 1]", "Probabilities must be inside [0, 1]")
    return rows


def validate_registry(registry: dict[str, Any], expected_model_ids: list[str]) -> list[dict[str, Any]]:
    if not isinstance(registry, dict) or not isinstance(registry.get("models"), list):
        raise VerificationError("Registry must contain a models list")
    models = registry["models"]
    ids = [item.get("model_id") for item in models if isinstance(item, dict)]
    if len(models) != len(expected_model_ids) or len(ids) != len(set(ids)) or ids != expected_model_ids:
        raise VerificationError(f"Registry model IDs are incomplete, duplicated, or unordered: {ids}")
    rows: list[dict[str, Any]] = []
    for item in models:
        required = {
            "model_id", "family", "feature_set", "ordered_predictors", "model_path", "model_sha256",
            "config", "selected_iteration_or_epoch", "class_order", "representation",
            "preprocessor_reference", "preprocessor_sha256", "fit_membership_fingerprint",
            "tuning_membership_fingerprint", "calibration_status", "decision_threshold_status",
        }
        missing = sorted(required - set(item))
        if missing:
            raise VerificationError(f"Registry item {item.get('model_id')} is missing fields: {missing}")
        if not item["ordered_predictors"] or len(item["ordered_predictors"]) != len(set(item["ordered_predictors"])):
            raise VerificationError(f"Registry predictors are empty or duplicated: {item['model_id']}")
        if item["class_order"] != [0, 1]:
            raise VerificationError(f"Unexpected class order: {item['model_id']}={item['class_order']}")
        if item["calibration_status"] != "NOT_FITTED" or item["decision_threshold_status"] != "NOT_SELECTED":
            raise VerificationError(f"Unexpected downstream status in registry: {item['model_id']}")
        rows.append({"check": f"registry__{item['model_id']}", "status": "PASS", "actual": "complete", "expected": "complete",
                     "reason": "Required model metadata and ordered predictors are present"})
    return rows


def completion_status_check(
    status_path: Path,
    config_hash: str,
    expected_model_id: str | None = None,
) -> tuple[bool, str]:
    if not status_path.is_file():
        return False, "status file missing"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"status unreadable: {type(exc).__name__}"
    required_top = {"status", "config_hash", "comparison", "candidates", "history", "registry", "artifacts"}
    missing_top = sorted(required_top - set(status))
    if missing_top:
        return False, f"missing status fields: {missing_top}"
    if status["status"] != "COMPLETE" or status["config_hash"] != config_hash:
        return False, "status/config hash mismatch"
    registry = status["registry"]
    if not isinstance(registry, dict):
        return False, "registry is not an object"
    model_id = registry.get("model_id")
    if expected_model_id is not None and model_id != expected_model_id:
        return False, f"wrong model_id: {model_id}"
    if not isinstance(registry.get("ordered_predictors"), list) or not registry["ordered_predictors"]:
        return False, "ordered predictors missing"
    if registry.get("representation") not in {"linear_nn", "gbdt"}:
        return False, "invalid representation"
    if not registry.get("preprocessor_sha256"):
        return False, "preprocessor hash missing"
    artifacts = status["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        return False, "artifact manifest missing or empty"
    if any(not isinstance(item, dict) for item in artifacts):
        return False, "artifact manifest contains a non-object"
    kinds = [item.get("kind") for item in artifacts]
    required_kinds = {"model", "prediction_part", "reload_sample"}
    if registry.get("family") == "mlp":
        required_kinds.add("architecture")
    if not required_kinds.issubset(kinds) or len(kinds) != len(set(kinds)):
        return False, f"artifact kinds missing or duplicated: {kinds}"
    for item in artifacts:
        if set(item) < {"kind", "path", "sha256"} or not item.get("path") or not item.get("sha256"):
            return False, "artifact entry lacks kind/path/sha256"
        path = Path(item["path"])
        if not path.is_file():
            return False, f"artifact missing: {item['kind']}"
        if sha256_file(path) != item["sha256"]:
            return False, f"artifact checksum mismatch: {item['kind']}"
    return True, "complete schema and mandatory artifacts verified"


def run_isolated_combinations(
    combinations: list[str],
    family_of: Callable[[str], str],
    run_one: Callable[[str], Any],
    record_failure: Callable[[str, str, str], None],
) -> dict[str, Any]:
    """Execute combinations with explicit shared/local/family failure semantics."""
    results: dict[str, Any] = {}
    blocked_families: dict[str, str] = {}
    for combination in combinations:
        family = family_of(combination)
        if family in blocked_families:
            reason = blocked_families[family]
            record_failure(combination, "BLOCKED_BY_FAMILY_FAILURE", reason)
            results[combination] = {"status": "BLOCKED_BY_FAMILY_FAILURE", "reason": reason}
            continue
        try:
            results[combination] = {"status": "COMPLETE", "result": run_one(combination)}
        except KeyboardInterrupt as exc:
            reason = f"KeyboardInterrupt: {exc or 'execution interrupted'}"
            record_failure(combination, "INTERRUPTED", reason)
            raise
        except SharedContractError:
            raise
        except FamilyImplementationError as exc:
            reason = f"{type(exc).__name__}: {exc}"
            blocked_families[family] = reason
            record_failure(combination, "FAILED_FAMILY", reason)
            results[combination] = {"status": "FAILED_FAMILY", "reason": reason}
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            record_failure(combination, "FAILED_LOCAL", reason)
            results[combination] = {"status": "FAILED_LOCAL", "reason": reason}
    return results


def finalize_after_verification(
    write_pending: Callable[[], None],
    verify: Callable[[], Any],
    write_complete: Callable[[Any], None],
    write_failed: Callable[[Exception], None],
) -> Any:
    write_pending()
    try:
        result = verify()
    except Exception as exc:
        write_failed(exc)
        raise
    write_complete(result)
    return result


def test_execution_status(evidence: dict[str, Any]) -> tuple[str, str]:
    if not evidence:
        return "NOT_CHECKED", "No machine-readable execution evidence"
    required = {"command", "cwd", "started_at", "ended_at", "exit_code", "collected_tests", "stdout_path", "stderr_path"}
    missing = sorted(required - set(evidence))
    if missing:
        return "FAIL", f"Missing test evidence fields: {missing}"
    if evidence["exit_code"] == 0 and int(evidence["collected_tests"]) > 0:
        return "PASS", "Exit code zero and at least one test collected"
    return "FAIL", f"exit_code={evidence['exit_code']}, collected_tests={evidence['collected_tests']}"
