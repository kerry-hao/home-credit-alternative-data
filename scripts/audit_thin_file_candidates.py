#!/usr/bin/env python3
"""Task 11 Part 1: target-blind TRAIN-only thin-file candidate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq


VERSION = "1.0.0"
TASK_NAME = "TASK_11_PART_1_THIN_FILE_CANDIDATE_AUDIT"
EXPECTED_TOTAL = 1_526_659
EXPECTED_SPLITS = {"train": 1_068_661, "validation": 228_999, "evaluation": 228_999}
EXPECTED_ROLES = {"validation_tuning": 114_499, "validation_calibration": 114_500}
EXPECTED_MEMBERSHIP_FP = "8bfb238774655f46a4f21c2668b2815f321c9298cb6d38a8da33f35a1ab91714"
MODEL_ORDER = ["logit_T", "logit_T_plus_AD", "lightgbm_T", "lightgbm_T_plus_AD", "mlp_T", "mlp_T_plus_AD"]
EXPECTED_METHODS = {
    "logit_T": "identity",
    "logit_T_plus_AD": "identity",
    "lightgbm_T": "identity",
    "lightgbm_T_plus_AD": "identity",
    "mlp_T": "logistic",
    "mlp_T_plus_AD": "identity",
}
ACCOUNT_CANDIDATES = ["credquantity_1099L", "t__observed_active_credit_count"]
QUANTILES = [("p01", 0.01), ("p05", 0.05), ("p10", 0.10), ("p20", 0.20),
             ("p25", 0.25), ("p33", 0.33), ("p40", 0.40), ("p50", 0.50),
             ("p60", 0.60), ("p67", 0.67), ("p75", 0.75), ("p80", 0.80),
             ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]
OUTPUT_NAMES = {
    "foundation_verification.csv", "account_count_candidate_audit.csv", "account_count_top_values.csv",
    "bureau_history_audit.csv", "thin_joint_coverage.csv", "case_id_integrity.csv",
    "input_hashes_before_after.csv", "validation_results.csv", "run_config.json", "task11_part1_report.md",
}


class AuditBlocked(RuntimeError):
    """A protected foundation or source contract did not pass."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def json_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return "" if not math.isfinite(float(value)) else repr(float(value))
    return str(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temp, index=False, lineterminator="\n")
    os.replace(temp, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_text(path: Path, value: str) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(value, encoding="utf-8")
    os.replace(temp, path)


def foundation_row(check_id: str, component: str, expected: Any, observed: Any, passed: bool,
                   evidence: Path, notes: str = "") -> dict[str, Any]:
    return {
        "check_id": check_id, "component": component, "expected": json_value(expected),
        "observed": json_value(observed), "status": "PASS" if passed else "FAIL",
        "evidence_path": str(evidence), "notes": notes,
    }


def validation_row(validation_id: str, description: str, observed: Any, expected: Any,
                   status: str, notes: str = "") -> dict[str, Any]:
    return {
        "validation_id": validation_id, "description": description, "observed_result": json_value(observed),
        "expected_result": json_value(expected), "status": status, "notes": notes,
    }


def resolve_inputs(data_root: Path) -> dict[str, Any]:
    train_dir = data_root / "raw/parquet_files/train"
    active = [train_dir / "train_credit_bureau_b_1.parquet"]
    closed = sorted(train_dir.glob("train_credit_bureau_a_1_*.parquet"))
    values: dict[str, Any] = {
        "application_features": data_root / "interim/task08/application_features.parquet",
        "manifest": data_root / "interim/task08_followup/application_manifest.parquet",
        "task08_feature_registry": data_root / "audits/task08/feature_registry.csv",
        "task08_followup_registry": data_root / "audits/task08_followup/feature_registry.csv",
        "feature_definitions": data_root / "raw/feature_definitions.csv",
        "task07_credit_review": data_root / "audits/task07_batch2/credit_count_review.csv",
        "task09_registry": data_root / "audits/task09/first_full/selected_model_registry.json",
        "task09_config": data_root / "audits/task09/first_full/run_config.json",
        "task10_registry": data_root / "audits/task10/first_full_v3/calibration_registry.json",
        "task10_config": data_root / "audits/task10/first_full_v3/run_config.json",
        "task10_status": data_root / "audits/task10/first_full_v3/task_status.json",
        "task10_validation": data_root / "audits/task10/first_full_v3/validation_results.csv",
        "task10_folds": data_root / "audits/task10/first_full_v3/fold_assignments_summary.csv",
        "active_files": active,
        "closed_files": closed,
    }
    required = [value for value in values.values() if isinstance(value, Path)] + active + closed
    if len(closed) != 4:
        raise AuditBlocked(f"Expected four credit_bureau_a_1 shards, found {len(closed)}")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AuditBlocked(f"Required inputs missing: {missing}")
    return values


def protected_paths(inputs: dict[str, Any]) -> list[tuple[str, str, Path]]:
    data_root = inputs["application_features"].parents[2]
    task09 = json.loads(inputs["task09_registry"].read_text(encoding="utf-8"))
    task10 = json.loads(inputs["task10_registry"].read_text(encoding="utf-8"))
    paths: list[tuple[str, str, Path]] = [
        ("task08_application_features", "Task 8", inputs["application_features"]),
        ("task08_application_manifest", "Task 8 split registry", inputs["manifest"]),
        ("task08_feature_registry", "Task 8", inputs["task08_feature_registry"]),
        ("task08_followup_feature_registry", "Task 8 split/preprocessing", inputs["task08_followup_registry"]),
        ("task08_preprocessor", "Task 8 split/preprocessing", data_root / "interim/task08_followup/preprocessing/preprocessor.json"),
        ("task08_feature_sets", "Task 8 split/preprocessing", data_root / "interim/task08_followup/preprocessing/feature_sets.json"),
        ("task09_run_config", "Task 9", inputs["task09_config"]),
        ("task09_selected_model_registry", "Task 9", inputs["task09_registry"]),
        ("task10_run_config", "Task 10", inputs["task10_config"]),
        ("task10_calibration_registry", "Task 10", inputs["task10_registry"]),
        ("task10_task_status", "Task 10", inputs["task10_status"]),
    ]
    for item in task09.get("models", []):
        paths.append((f"task09_model::{item['model_id']}", "Task 9", Path(item["model_path"])))
        if item.get("architecture_path"):
            paths.append((f"task09_architecture::{item['model_id']}", "Task 9", Path(item["architecture_path"])))
    for item in task10.get("models", []):
        calibrator = Path(item["selected_calibrator_path"])
        paths.append((f"task10_selected_calibrator::{item['model_id']}", "Task 10", calibrator))
        paths.append((f"task10_selected_calibrator_metadata::{item['model_id']}", "Task 10", calibrator.with_name("calibrator_metadata.json")))
    unique: dict[str, tuple[str, str, Path]] = {}
    for item in paths:
        unique[str(item[2])] = item
    missing = [str(path) for _, _, path in unique.values() if not path.is_file()]
    if missing:
        raise AuditBlocked(f"Protected artifact missing: {missing}")
    return list(unique.values())


def hash_protected(paths: list[tuple[str, str, Path]]) -> dict[str, dict[str, Any]]:
    return {str(path): {"protected_path": str(path), "artifact": name, "component_task": component,
                        "sha256": sha256_file(path)} for name, component, path in paths}


@dataclass
class Foundation:
    inputs: dict[str, Any]
    manifest: pd.DataFrame
    train_manifest: pd.DataFrame
    rows: list[dict[str, Any]]
    before_hashes: dict[str, dict[str, Any]]


def verify_foundation(data_root: Path) -> Foundation:
    inputs = resolve_inputs(data_root)
    before_hashes = hash_protected(protected_paths(inputs))
    rows: list[dict[str, Any]] = []
    manifest = pd.read_parquet(inputs["manifest"], columns=["case_id", "base_order", "outer_split", "validation_role"])
    split_counts = manifest["outer_split"].value_counts().to_dict()
    role_counts = manifest.loc[manifest["validation_role"].ne(""), "validation_role"].value_counts().to_dict()
    membership_fp = sha256_lines(
        f"{case}\t{outer}\t{role}" for case, outer, role in
        zip(manifest.case_id, manifest.outer_split, manifest.validation_role)
    )
    rows.append(foundation_row("population_count", "Frozen split", EXPECTED_TOTAL, len(manifest), len(manifest) == EXPECTED_TOTAL, inputs["manifest"]))
    for split, expected in EXPECTED_SPLITS.items():
        observed = int(split_counts.get(split, 0))
        rows.append(foundation_row(f"split_count::{split}", "Frozen split", expected, observed, observed == expected, inputs["manifest"]))
    for role, expected in EXPECTED_ROLES.items():
        observed = int(role_counts.get(role, 0))
        rows.append(foundation_row(f"role_count::{role}", "Frozen split", expected, observed, observed == expected, inputs["manifest"]))
    rows.append(foundation_row("membership_fingerprint", "Frozen split", EXPECTED_MEMBERSHIP_FP, membership_fp,
                               membership_fp == EXPECTED_MEMBERSHIP_FP, inputs["manifest"]))
    base_order_ok = np.array_equal(manifest.base_order.to_numpy(np.int64), np.arange(EXPECTED_TOTAL, dtype=np.int64))
    key_ok = manifest.case_id.notna().all() and manifest.case_id.is_unique and base_order_ok
    rows.append(foundation_row("split_key_integrity", "Frozen split", "unique case_id and canonical base_order",
                               {"unique": bool(manifest.case_id.is_unique), "nonnull": bool(manifest.case_id.notna().all()),
                                "base_order": base_order_ok}, key_ok, inputs["manifest"]))

    feature_file = pq.ParquetFile(inputs["application_features"])
    feature_rows = feature_file.metadata.num_rows
    feature_keys = pd.read_parquet(inputs["application_features"], columns=["case_id"])["case_id"]
    rows.append(foundation_row("task08_application_rows", "Task 8", EXPECTED_TOTAL, feature_rows,
                               feature_rows == EXPECTED_TOTAL, inputs["application_features"]))
    feature_key_ok = feature_keys.notna().all() and feature_keys.is_unique
    rows.append(foundation_row("task08_unique_case_id", "Task 8", "unique/non-null", {"unique": bool(feature_keys.is_unique),
                               "null_count": int(feature_keys.isna().sum())}, feature_key_ok, inputs["application_features"]))
    train_manifest = manifest.loc[manifest.outer_split.eq("train"), ["case_id", "base_order"]].copy()
    rows.append(foundation_row("train_membership_available", "Task 8", EXPECTED_SPLITS["train"], len(train_manifest),
                               len(train_manifest) == EXPECTED_SPLITS["train"], inputs["manifest"],
                               "Used saved outer_split; no split was reconstructed."))

    task09 = json.loads(inputs["task09_registry"].read_text(encoding="utf-8"))
    model_ids = [item.get("model_id") for item in task09.get("models", [])]
    task09_ok = task09.get("status") == "COMPLETE" and model_ids == MODEL_ORDER
    rows.append(foundation_row("task09_six_models", "Task 9", {"status": "COMPLETE", "models": MODEL_ORDER},
                               {"status": task09.get("status"), "models": model_ids}, task09_ok, inputs["task09_registry"]))
    model_artifacts_ok = all(Path(item.get("model_path", "")).is_file() and
                             sha256_file(Path(item["model_path"])) == item.get("model_sha256") for item in task09.get("models", []))
    rows.append(foundation_row("task09_selected_artifacts", "Task 9", "six registered model hashes match", model_artifacts_ok,
                               model_artifacts_ok, inputs["task09_registry"]))

    task10 = json.loads(inputs["task10_registry"].read_text(encoding="utf-8"))
    task10_status = json.loads(inputs["task10_status"].read_text(encoding="utf-8"))
    config = json.loads(inputs["task10_config"].read_text(encoding="utf-8"))
    rows.append(foundation_row("task10_first_full_v3_complete", "Task 10", "COMPLETE",
                               {"registry": task10.get("status"), "task_status": task10_status.get("status")},
                               task10.get("status") == "COMPLETE" and task10_status.get("status") == "COMPLETE",
                               inputs["task10_status"]))
    cal_models = task10.get("models", [])
    cal_ids = [item.get("model_id") for item in cal_models]
    cal_paths_ok = len(cal_models) == 6 and cal_ids == MODEL_ORDER and all(
        Path(item.get("selected_calibrator_path", "")).is_file() and
        sha256_file(Path(item["selected_calibrator_path"])) == item.get("selected_calibrator_sha256")
        for item in cal_models
    )
    rows.append(foundation_row("task10_six_selected_calibrators", "Task 10", MODEL_ORDER, cal_ids,
                               cal_paths_ok, inputs["task10_registry"]))
    for item in cal_models:
        model_id, observed = item.get("model_id"), item.get("selected_method")
        expected = EXPECTED_METHODS.get(model_id)
        rows.append(foundation_row(f"task10_method::{model_id}", "Task 10", expected, observed,
                                   observed == expected, inputs["task10_registry"]))
    verification = task10_status.get("verification", {})
    rows.append(foundation_row("task10_calibration_rows", "Task 10", 114_500, verification.get("rows"),
                               verification.get("rows") == 114_500, inputs["task10_status"]))
    rows.append(foundation_row("task10_calibration_positive", "Task 10", 3_600, verification.get("positives"),
                               verification.get("positives") == 3_600, inputs["task10_status"]))
    negatives = verification.get("rows", 0) - verification.get("positives", 0)
    rows.append(foundation_row("task10_calibration_negative", "Task 10", 110_900, negatives,
                               negatives == 110_900, inputs["task10_status"]))
    folds = pd.read_csv(inputs["task10_folds"])
    fold_observed = folds[["rows", "positive_count", "negative_count"]].to_dict("records")
    fold_ok = len(folds) == 5 and (folds.rows == 22_900).all() and (folds.positive_count == 720).all() and (folds.negative_count == 22_180).all()
    rows.append(foundation_row("task10_fold_counts", "Task 10", "5 folds; 22900 rows/720 positive/22180 negative each",
                               fold_observed, bool(fold_ok), inputs["task10_folds"]))
    validations = pd.read_csv(inputs["task10_validation"])
    validation_ok = not validations.status.eq("FAIL").any() and validations.loc[validations.check.ne("final_evaluation"), "status"].eq("PASS").all()
    rows.append(foundation_row("task10_existing_validation", "Task 10", "all applicable checks PASS",
                               validations.status.value_counts().to_dict(), bool(validation_ok), inputs["task10_validation"]))
    interface_ok = config.get("run_id") == "first_full_v3" and config.get("base_run_id") == "first_full" and cal_paths_ok
    rows.append(foundation_row("task10_documented_interface", "Task 10", "first_full/first_full_v3 registry resolves six artifacts",
                               {"base_run_id": config.get("base_run_id"), "run_id": config.get("run_id"), "resolved": cal_paths_ok},
                               interface_ok, inputs["task10_config"], "Consumed via saved registry; no file was rewritten."))
    failed = [row for row in rows if row["status"] != "PASS"]
    if failed:
        raise AuditBlocked(f"Protected foundation contradiction: {json.dumps(failed, ensure_ascii=False)}")
    return Foundation(inputs, manifest, train_manifest, rows, before_hashes)


def clean_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.where(np.isfinite(numeric), np.nan)


def consistent_application_count(train_case_ids: pd.Series, source: pd.DataFrame, column: str) -> tuple[pd.Series, dict[str, int]]:
    """Reuse Task 7's rule: a value exists only when all finite row values agree."""
    index = pd.Index(train_case_ids.to_numpy(), name="case_id")
    matched = source.loc[source.case_id.isin(index), ["case_id", column]].copy()
    matched[column] = clean_numeric(matched[column])
    finite = matched.loc[matched[column].notna()]
    grouped = finite.groupby("case_id", sort=False)[column].agg(["min", "max", "count"])
    consistent = grouped["min"].eq(grouped["max"])
    values = grouped["min"].where(consistent).reindex(index)
    source_counts = matched.groupby("case_id", sort=False).size()
    details = {
        "source_rows": int(len(matched)), "unique_source_case_ids": int(matched.case_id.nunique()),
        "duplicate_source_case_ids": int((source_counts > 1).sum()), "duplicate_source_rows": int((source_counts - 1).clip(lower=0).sum()),
        "source_present_no_finite": int((source_counts.reindex(index, fill_value=0).gt(0) & grouped["count"].reindex(index, fill_value=0).eq(0)).sum()),
        "conflicting_finite_values": int((~consistent).sum()),
    }
    values.name = column
    return values.reset_index(drop=True), details


@dataclass
class DateAggregate:
    source_name: str
    earliest_parseable_ns: np.ndarray
    earliest_valid_ns: np.ndarray
    source_present: np.ndarray
    parseable_present: np.ndarray
    valid_present: np.ndarray
    after_present: np.ndarray
    invalid_present: np.ndarray
    source_rows: int
    unique_source_case_ids: int
    duplicate_source_case_ids: int
    duplicate_source_rows: int
    native_missing_rows: int
    invalid_rows: int
    after_rows: int


def empty_date_aggregate(source_name: str, n: int) -> DateAggregate:
    sentinel = np.iinfo(np.int64).max
    return DateAggregate(source_name, np.full(n, sentinel, dtype=np.int64), np.full(n, sentinel, dtype=np.int64),
                         np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool),
                         0, 0, 0, 0, 0, 0, 0)


def update_date_aggregate(aggregate: DateAggregate, frame: pd.DataFrame, date_column: str,
                          train_index: pd.Index, decision_ns: np.ndarray) -> None:
    positions = train_index.get_indexer(frame["case_id"].to_numpy())
    matched = positions >= 0
    if not matched.any():
        return
    positions = positions[matched]
    raw = frame.loc[matched, date_column].reset_index(drop=True)
    aggregate.source_rows += len(positions)
    aggregate.source_present[positions] = True
    native_missing = raw.isna().to_numpy()
    parsed = pd.to_datetime(raw, errors="coerce")
    parsed_missing = parsed.isna().to_numpy()
    invalid = ~native_missing & parsed_missing
    aggregate.native_missing_rows += int(native_missing.sum())
    aggregate.invalid_rows += int(invalid.sum())
    if invalid.any():
        aggregate.invalid_present[positions[invalid]] = True
    usable = ~parsed_missing
    if usable.any():
        pos = positions[usable]
        date_ns = parsed.loc[usable].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        aggregate.parseable_present[pos] = True
        np.minimum.at(aggregate.earliest_parseable_ns, pos, date_ns)
        after = date_ns > decision_ns[pos]
        aggregate.after_rows += int(after.sum())
        if after.any():
            aggregate.after_present[pos[after]] = True
        valid_pos, valid_ns = pos[~after], date_ns[~after]
        if len(valid_pos):
            aggregate.valid_present[valid_pos] = True
            np.minimum.at(aggregate.earliest_valid_ns, valid_pos, valid_ns)


def finalize_date_aggregate(aggregate: DateAggregate) -> None:
    counts = pd.Series(np.flatnonzero(aggregate.source_present)).value_counts()
    # Presence arrays cannot retain multiplicity; multiplicity is derived separately by callers when needed.
    aggregate.unique_source_case_ids = int(aggregate.source_present.sum())


def iter_parquet_batches(paths: list[Path], columns: list[str], batch_size: int = 500_000) -> Iterator[pd.DataFrame]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            yield batch.to_pandas()


def aggregate_date_source(paths: list[Path], source_name: str, date_column: str,
                          train_index: pd.Index, decision_ns: np.ndarray) -> DateAggregate:
    aggregate = empty_date_aggregate(source_name, len(train_index))
    row_counts = np.zeros(len(train_index), dtype=np.int32)
    for frame in iter_parquet_batches(paths, ["case_id", date_column]):
        positions = train_index.get_indexer(frame.case_id.to_numpy())
        matched_positions = positions[positions >= 0]
        if len(matched_positions):
            np.add.at(row_counts, matched_positions, 1)
        update_date_aggregate(aggregate, frame, date_column, train_index, decision_ns)
    aggregate.unique_source_case_ids = int(np.count_nonzero(row_counts))
    aggregate.duplicate_source_case_ids = int(np.count_nonzero(row_counts > 1))
    aggregate.duplicate_source_rows = int(np.maximum(row_counts - 1, 0).sum())
    return aggregate


def combine_bureau_history(decision_ns: np.ndarray, active: DateAggregate, closed: DateAggregate) -> dict[str, Any]:
    sentinel = np.iinfo(np.int64).max
    earliest_unrestricted = np.minimum(active.earliest_parseable_ns, closed.earliest_parseable_ns)
    unrestricted_available = earliest_unrestricted != sentinel
    unrestricted_days = np.full(len(decision_ns), np.nan)
    unrestricted_days[unrestricted_available] = (
        decision_ns[unrestricted_available] - earliest_unrestricted[unrestricted_available]
    ) / 86_400_000_000_000
    earliest_valid = np.minimum(active.earliest_valid_ns, closed.earliest_valid_ns)
    valid = earliest_valid != sentinel
    days = np.full(len(decision_ns), np.nan)
    days[valid] = (decision_ns[valid] - earliest_valid[valid]) / 86_400_000_000_000
    return {
        "earliest_valid_ns": earliest_valid, "valid": valid, "days": days, "years": days / 365.25,
        "unrestricted_days": unrestricted_days,
        "negative_before_exclusion": unrestricted_available & (unrestricted_days < 0),
    }


def percent(count: int, denominator: int) -> float | None:
    return (100.0 * count / denominator) if denominator else None


def account_audit_row(candidate: str, values: pd.Series, source_definition: str,
                      zero_missing_note: str, total: int) -> dict[str, Any]:
    numeric = clean_numeric(values)
    observed = numeric.dropna().to_numpy(float)
    non_missing = len(observed)
    row: dict[str, Any] = {
        "candidate_name": candidate, "exact_source_or_engineered_definition": source_definition,
        "source_level_availability_status": "AVAILABLE", "total_train_cases": total,
        "non_missing_count": non_missing, "non_missing_percentage": percent(non_missing, total),
        "missing_count": total - non_missing, "missing_percentage": percent(total - non_missing, total),
        "zero_count": int(np.count_nonzero(observed == 0)),
        "zero_percentage_all_train": percent(int(np.count_nonzero(observed == 0)), total),
        "zero_percentage_non_missing": percent(int(np.count_nonzero(observed == 0)), non_missing),
        "positive_count": int(np.count_nonzero(observed > 0)),
        "positive_percentage_all_train": percent(int(np.count_nonzero(observed > 0)), total),
        "negative_count": int(np.count_nonzero(observed < 0)),
        "negative_percentage_all_train": percent(int(np.count_nonzero(observed < 0)), total),
        "unique_non_missing_values": int(pd.Series(observed).nunique()),
        "minimum": float(observed.min()) if non_missing else None,
        "maximum": float(observed.max()) if non_missing else None,
        "zero_and_missing_note": zero_missing_note,
    }
    quantiles = np.quantile(observed, [q for _, q in QUANTILES], method="linear") if non_missing else [None] * len(QUANTILES)
    for (name, _), value in zip(QUANTILES, quantiles):
        row[name] = float(value) if value is not None else None
    return row


def top_values(candidate: str, values: pd.Series, total: int) -> list[dict[str, Any]]:
    observed = clean_numeric(values).dropna()
    counts = observed.value_counts(dropna=False).rename_axis("value").reset_index(name="count")
    counts = counts.sort_values(["count", "value"], ascending=[False, True], kind="stable").head(10)
    return [{"candidate_name": candidate, "value": row.value, "count": int(row.count),
             "percentage_all_train": percent(int(row.count), total),
             "percentage_non_missing_train": percent(int(row.count), len(observed)), "frequency_rank": rank}
            for rank, row in enumerate(counts.itertuples(index=False), start=1)]


def history_metric(metric: str, value: Any, unit: str, denominator_name: str,
                   denominator_value: int | None = None, notes: str = "") -> dict[str, Any]:
    count_units = {"applications", "rows"}
    percentage = percent(int(value), int(denominator_value)) if unit in count_units and denominator_value is not None else None
    return {"metric": metric, "value": value, "unit": unit, "denominator_name": denominator_name,
            "denominator_value": denominator_value, "percentage": percentage, "notes": notes}


def build_history_audit(active: DateAggregate, closed: DateAggregate, history: dict[str, Any], total: int) -> pd.DataFrame:
    active_available = active.parseable_present
    closed_available = closed.parseable_present
    valid = history["valid"]
    days = history["days"][valid]
    years = history["years"][valid]
    rows = [
        history_metric("total_train_cases", total, "applications", "outer_train", total),
        history_metric("active_bureau_date_available", int(active_available.sum()), "applications", "outer_train", total),
        history_metric("closed_bureau_start_date_available", int(closed_available.sum()), "applications", "outer_train", total),
        history_metric("both_sources_available", int((active_available & closed_available).sum()), "applications", "outer_train", total),
        history_metric("at_least_one_source_available", int((active_available | closed_available).sum()), "applications", "outer_train", total),
        history_metric("neither_source_available", int((~active_available & ~closed_available).sum()), "applications", "outer_train", total),
        history_metric("valid_earliest_bureau_date_constructed", int(valid.sum()), "applications", "outer_train", total),
        history_metric("source_dates_after_decision_rows", active.after_rows + closed.after_rows, "rows", "all_matched_source_rows", active.source_rows + closed.source_rows),
        history_metric("applications_with_source_date_after_decision", int((active.after_present | closed.after_present).sum()), "applications", "outer_train", total),
        history_metric("invalid_or_unparsable_source_date_rows", active.invalid_rows + closed.invalid_rows, "rows", "all_matched_source_rows", active.source_rows + closed.source_rows),
        history_metric("applications_with_invalid_or_unparsable_source_date", int((active.invalid_present | closed.invalid_present).sum()), "applications", "outer_train", total),
        history_metric("native_missing_source_date_rows", active.native_missing_rows + closed.native_missing_rows, "rows", "all_matched_source_rows", active.source_rows + closed.source_rows),
        history_metric("zero_day_history", int(np.count_nonzero(valid & (history["days"] == 0))), "applications", "outer_train", total),
        history_metric("negative_history_before_invalid_exclusion", int(history["negative_before_exclusion"].sum()), "applications", "outer_train", total,
                       "Earliest parseable source date was after decision; excluded from valid history."),
        history_metric("valid_bureau_history_days", int(valid.sum()), "applications", "outer_train", total),
        history_metric("valid_bureau_history_years", int(valid.sum()), "applications", "outer_train", total),
    ]
    for source in (active, closed):
        rows.extend([
            history_metric(f"{source.source_name}_matched_source_rows", source.source_rows, "rows", f"{source.source_name}_matched_source_rows", source.source_rows),
            history_metric(f"{source.source_name}_unique_source_case_ids", source.unique_source_case_ids, "applications", "outer_train", total),
            history_metric(f"{source.source_name}_duplicate_source_case_ids", source.duplicate_source_case_ids, "applications", f"{source.source_name}_unique_source_case_ids", source.unique_source_case_ids),
            history_metric(f"{source.source_name}_duplicate_source_rows", source.duplicate_source_rows, "rows", f"{source.source_name}_matched_source_rows", source.source_rows),
        ])
    if len(days):
        rows.append(history_metric("bureau_history_days_minimum", float(days.min()), "days", "valid_bureau_history", len(days)))
        rows.append(history_metric("bureau_history_years_minimum", float(years.min()), "years", "valid_bureau_history", len(years)))
        q_days = np.quantile(days, [q for _, q in QUANTILES], method="linear")
        q_years = np.quantile(years, [q for _, q in QUANTILES], method="linear")
        for (name, _), day_value, year_value in zip(QUANTILES, q_days, q_years):
            rows.append(history_metric(f"bureau_history_days_{name}", float(day_value), "days", "valid_bureau_history", len(days)))
            rows.append(history_metric(f"bureau_history_years_{name}", float(year_value), "years", "valid_bureau_history", len(years)))
        rows.append(history_metric("bureau_history_days_maximum", float(days.max()), "days", "valid_bureau_history", len(days)))
        rows.append(history_metric("bureau_history_years_maximum", float(years.max()), "years", "valid_bureau_history", len(years)))
    return pd.DataFrame(rows)


def joint_coverage(candidate: str, count_values: pd.Series, history_valid: np.ndarray, total: int) -> list[dict[str, Any]]:
    values = clean_numeric(count_values).to_numpy(float)
    count_valid = np.isfinite(values)
    categories = [
        ("both_valid", count_valid & history_valid),
        ("count_valid_history_missing", count_valid & ~history_valid),
        ("count_missing_history_valid", ~count_valid & history_valid),
        ("both_missing", ~count_valid & ~history_valid),
        ("count_zero_history_valid", count_valid & (values == 0) & history_valid),
        ("count_zero_history_missing", count_valid & (values == 0) & ~history_valid),
        ("count_positive_history_valid", count_valid & (values > 0) & history_valid),
        ("count_positive_history_missing", count_valid & (values > 0) & ~history_valid),
    ]
    return [{"count_candidate": candidate, "history_candidate": "bureau_history_years", "coverage_category": name,
             "count": int(mask.sum()), "percentage": percent(int(mask.sum()), total),
             "denominator_name": "outer_train", "denominator_value": total} for name, mask in categories]


def case_integrity_rows(train: pd.DataFrame, feature_rows: int, feature_unique: int,
                        cred_details: dict[str, int], active: DateAggregate, closed: DateAggregate) -> list[dict[str, Any]]:
    total = len(train)
    return [
        {"source": "task08_application_features", "metric": "source_row_count", "value": feature_rows, "expected_or_note": str(EXPECTED_TOTAL)},
        {"source": "task08_application_features", "metric": "unique_source_case_id_count", "value": feature_unique, "expected_or_note": str(EXPECTED_TOTAL)},
        {"source": "credit_bureau_b_1", "metric": "source_row_count_before_aggregation", "value": cred_details["source_rows"], "expected_or_note": "TRAIN-matched rows"},
        {"source": "credit_bureau_b_1", "metric": "unique_source_case_id_count", "value": cred_details["unique_source_case_ids"], "expected_or_note": "TRAIN cases with source rows"},
        {"source": "credit_bureau_b_1", "metric": "duplicated_source_case_id_count", "value": cred_details["duplicate_source_case_ids"], "expected_or_note": "Expected one-to-many source"},
        {"source": "credit_bureau_b_1", "metric": "duplicate_source_row_count", "value": cred_details["duplicate_source_rows"], "expected_or_note": "Rows beyond first per case"},
        {"source": "credit_bureau_b_1", "metric": "conflicting_finite_application_count", "value": cred_details["conflicting_finite_values"], "expected_or_note": "Preserved as missing"},
        {"source": "credit_bureau_b_1", "metric": "source_present_no_finite_application_count", "value": cred_details["source_present_no_finite"], "expected_or_note": "Preserved as missing"},
        {"source": active.source_name, "metric": "source_row_count_before_aggregation", "value": active.source_rows, "expected_or_note": "TRAIN-matched rows"},
        {"source": active.source_name, "metric": "unique_source_case_id_count", "value": active.unique_source_case_ids, "expected_or_note": "TRAIN cases with source rows"},
        {"source": active.source_name, "metric": "duplicated_source_case_id_count", "value": active.duplicate_source_case_ids, "expected_or_note": "Expected one-to-many source"},
        {"source": closed.source_name, "metric": "source_row_count_before_aggregation", "value": closed.source_rows, "expected_or_note": "TRAIN-matched rows"},
        {"source": closed.source_name, "metric": "unique_source_case_id_count", "value": closed.unique_source_case_ids, "expected_or_note": "TRAIN cases with source rows"},
        {"source": closed.source_name, "metric": "duplicated_source_case_id_count", "value": closed.duplicate_source_case_ids, "expected_or_note": "Expected one-to-many source"},
        {"source": "final_audit_table", "metric": "row_count", "value": total, "expected_or_note": str(EXPECTED_SPLITS["train"])},
        {"source": "final_audit_table", "metric": "unique_case_id_count", "value": int(train.case_id.nunique()), "expected_or_note": str(EXPECTED_SPLITS["train"])},
        {"source": "final_audit_table", "metric": "missing_train_case_ids_after_joins", "value": 0, "expected_or_note": "0"},
        {"source": "final_audit_table", "metric": "unexpected_extra_case_ids", "value": 0, "expected_or_note": "0"},
        {"source": "final_audit_table", "metric": "join_expanded_one_row_per_application", "value": False, "expected_or_note": "False"},
    ]


def get_git_state(repo_root: Path) -> tuple[str, list[str]]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.splitlines()
    return commit, status


def run(data_root: Path, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    data_root, output_dir = data_root.resolve(), output_dir.resolve()
    protected_roots = [data_root / name for name in ("audits/task08", "audits/task08_followup", "audits/task09", "audits/task10",
                                                       "interim/task08", "interim/task08_followup", "interim/task09", "interim/task10",
                                                       "models/task09", "models/task10")]
    if any(output_dir == root or root in output_dir.parents for root in protected_roots):
        raise AuditBlocked(f"Output directory is inside a protected Task 8-10 path: {output_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Task 11 Part 1 output: {output_dir}")

    foundation = verify_foundation(data_root)
    inputs, manifest = foundation.inputs, foundation.manifest
    features = pd.read_parquet(inputs["application_features"], columns=["case_id", "date_decision", "t__observed_active_credit_count"])
    if not np.array_equal(features.case_id.to_numpy(), manifest.case_id.to_numpy()):
        raise AuditBlocked("Task 8 feature keys do not preserve frozen manifest order")
    train_positions = foundation.train_manifest.base_order.to_numpy(np.int64)
    train = features.iloc[train_positions].reset_index(drop=True)
    if not np.array_equal(train.case_id.to_numpy(), foundation.train_manifest.case_id.to_numpy()):
        raise AuditBlocked("TRAIN feature membership does not align with frozen manifest")
    if len(train) != EXPECTED_SPLITS["train"] or not train.case_id.is_unique:
        raise AuditBlocked("Final audit population is not one row per frozen TRAIN application")
    decisions = pd.to_datetime(train.date_decision, errors="coerce")
    if decisions.isna().any():
        raise AuditBlocked(f"TRAIN decision dates missing/unparsable: {int(decisions.isna().sum())}")
    decision_ns = decisions.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    train_index = pd.Index(train.case_id.to_numpy(), name="case_id")

    bureau_b = pd.concat([batch for batch in iter_parquet_batches(inputs["active_files"], ["case_id", "credquantity_1099L"])], ignore_index=True)
    cred_values, cred_details = consistent_application_count(train.case_id, bureau_b, "credquantity_1099L")
    active = aggregate_date_source(inputs["active_files"], "credit_bureau_b_1_active", "contractdate_551D", train_index, decision_ns)
    closed = aggregate_date_source(inputs["closed_files"], "credit_bureau_a_1_closed", "dateofcredstart_739D", train_index, decision_ns)
    history = combine_bureau_history(decision_ns, active, closed)
    history_valid = history["valid"]

    account_rows = [
        account_audit_row("credquantity_1099L", cred_values,
                          "credit_bureau_b_1.credquantity_1099L; Task 7 application value only when all finite linked row values agree",
                          "Zero would be an observed consistent source value; no source, no finite value, or conflicting finite values remain missing.", len(train)),
        account_audit_row("t__observed_active_credit_count", train["t__observed_active_credit_count"],
                          "Task 8 direct value from static_0.numactivecreds_622L before model imputation/scaling",
                          "Zero is a valid observed active-credit count, not absence of all past credit history; missing remains missing.", len(train)),
    ]
    account_frame = pd.DataFrame(account_rows)
    top_frame = pd.DataFrame(top_values("credquantity_1099L", cred_values, len(train)) +
                             top_values("t__observed_active_credit_count", train["t__observed_active_credit_count"], len(train)))
    history_frame = build_history_audit(active, closed, history, len(train))
    joint_frame = pd.DataFrame(joint_coverage("credquantity_1099L", cred_values, history_valid, len(train)) +
                               joint_coverage("t__observed_active_credit_count", train["t__observed_active_credit_count"], history_valid, len(train)))
    integrity_frame = pd.DataFrame(case_integrity_rows(train, len(features), int(features.case_id.nunique()), cred_details, active, closed))

    after_hashes = hash_protected(protected_paths(inputs))
    hash_rows = []
    for path, before in foundation.before_hashes.items():
        after = after_hashes.get(path, {})
        hash_rows.append({"protected_path": path, "component_task": before["component_task"],
                          "sha256_before": before["sha256"], "sha256_after": after.get("sha256"),
                          "unchanged_flag": before["sha256"] == after.get("sha256")})
    hash_frame = pd.DataFrame(hash_rows)
    hashes_ok = bool(hash_frame.unchanged_flag.all())
    foundation.rows.append(foundation_row("protected_hashes_unchanged", "Tasks 8-10", True, hashes_ok, hashes_ok,
                                          output_dir / "input_hashes_before_after.csv", f"{len(hash_frame)} protected files hashed before and after."))
    if not hashes_ok:
        changed = hash_frame.loc[~hash_frame.unchanged_flag, "protected_path"].tolist()
        raise AuditBlocked(f"Protected files changed during audit: {changed}")
    foundation_frame = pd.DataFrame(foundation.rows)

    validations = [
        validation_row("candidate_list_exact", "Only the two authorized account-count candidates were audited",
                       account_frame.candidate_name.tolist(), ACCOUNT_CANDIDATES,
                       "PASS" if account_frame.candidate_name.tolist() == ACCOUNT_CANDIDATES else "FAIL"),
        validation_row("frozen_train_membership", "Saved outer TRAIN membership used", len(train), EXPECTED_SPLITS["train"],
                       "PASS" if len(train) == EXPECTED_SPLITS["train"] else "FAIL"),
        validation_row("target_independence", "No target column was loaded", False, False, "PASS"),
        validation_row("final_evaluation_not_loaded", "No final-evaluation features or labels were loaded into audit calculations", False, False, "PASS"),
        validation_row("history_nonnegative", "All retained bureau history is nonnegative", int(np.count_nonzero(history["days"][history_valid] < 0)), 0,
                       "PASS" if not np.count_nonzero(history["days"][history_valid] < 0) else "FAIL"),
        validation_row("history_year_conversion", "Years equal days divided by 365.25",
                       bool(np.allclose(history["years"][history_valid], history["days"][history_valid] / 365.25)), True,
                       "PASS" if np.allclose(history["years"][history_valid], history["days"][history_valid] / 365.25) else "FAIL"),
        validation_row("zero_missing_preserved", "Zero and missing remain distinct for both count candidates",
                       {row["candidate_name"]: {"zero": row["zero_count"], "missing": row["missing_count"]} for row in account_rows},
                       "separate reported counts", "PASS"),
        validation_row("one_row_per_train_case", "Final audit population remains one row per TRAIN case",
                       {"rows": len(train), "unique": int(train.case_id.nunique())},
                       {"rows": EXPECTED_SPLITS["train"], "unique": EXPECTED_SPLITS["train"]},
                       "PASS" if len(train) == train.case_id.nunique() == EXPECTED_SPLITS["train"] else "FAIL"),
        validation_row("no_join_expansion", "Source aggregation did not expand application rows", len(train), EXPECTED_SPLITS["train"], "PASS"),
        validation_row("joint_primary_reconciliation", "Four primary joint states reconcile to TRAIN total",
                       {candidate: int(joint_frame[(joint_frame.count_candidate == candidate) & joint_frame.coverage_category.isin(
                           ["both_valid", "count_valid_history_missing", "count_missing_history_valid", "both_missing"])]["count"].sum())
                        for candidate in ACCOUNT_CANDIDATES}, EXPECTED_SPLITS["train"],
                       "PASS" if all(int(joint_frame[(joint_frame.count_candidate == candidate) & joint_frame.coverage_category.isin(
                           ["both_valid", "count_valid_history_missing", "count_missing_history_valid", "both_missing"])]["count"].sum()) == len(train)
                                     for candidate in ACCOUNT_CANDIDATES) else "FAIL"),
        validation_row("protected_output_boundary", "Output is outside protected Task 8-10 directories", str(output_dir), "Task 11 directory", "PASS"),
        validation_row("protected_hashes", "Protected hashes unchanged", hashes_ok, True, "PASS" if hashes_ok else "FAIL"),
        validation_row("no_chart_outputs", "No chart/HTML/notebook files are specified", sorted(OUTPUT_NAMES), "CSV/JSON/Markdown only", "PASS"),
    ]
    validation_frame = pd.DataFrame(validations)
    if validation_frame.status.eq("FAIL").any():
        raise AuditBlocked(f"Task 11 validation failed: {validation_frame.loc[validation_frame.status.eq('FAIL')].to_dict('records')}")

    git_commit, git_status = get_git_state(repo_root)
    package_versions = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                        "pyarrow": pyarrow.__version__}
    raw_paths = [str(path) for path in [inputs["application_features"], inputs["manifest"], inputs["feature_definitions"],
                                        *inputs["active_files"], *inputs["closed_files"]]]
    config = {
        "task_name": TASK_NAME, "part": 1, "version": VERSION, "timestamp_utc": utc_now(),
        "repository_root": str(repo_root), "data_root": str(data_root), "python_executable": sys.executable,
        "package_versions": package_versions, "frozen_split_fingerprint": EXPECTED_MEMBERSHIP_FP,
        "canonical_task09_base_run": "first_full", "canonical_task10_run": "first_full_v3",
        "candidate_variables": [*ACCOUNT_CANDIDATES, "bureau_history_years"], "raw_source_paths_used": raw_paths,
        "existing_linkage_helpers_reused": [
            "Frozen application_manifest.outer_split membership and case_id linkage",
            "Task 7 credit_candidate_state semantics: finite per-application min=max required",
            "Task 7 date parsing and source-date-minus-decision-date validity convention",
        ],
        "output_directory": str(output_dir), "target_loaded": False,
        "final_evaluation_data_or_labels_loaded": False, "random_seeds": None,
        "git_commit": git_commit, "pre_existing_worktree_status": git_status,
    }

    account_lookup = account_frame.set_index("candidate_name")
    history_lookup = history_frame.set_index("metric")["value"].to_dict()
    joint_lookup = joint_frame.set_index(["count_candidate", "coverage_category"])[["count", "percentage"]]
    report = f"""# Task 11 Part 1 — thin-file candidate audit

Status: **COMPLETE**

## Protected foundation verification

Task 8 application and frozen-split interfaces, all six Task 9 selected models, and canonical Task 10 run `first_full_v3` passed read-only checks. All {len(hash_frame)} protected hashes were unchanged.

## Candidates audited on frozen outer TRAIN

- `credquantity_1099L`: {int(account_lookup.loc['credquantity_1099L','non_missing_count']):,} non-missing ({account_lookup.loc['credquantity_1099L','non_missing_percentage']:.6f}%); {int(account_lookup.loc['credquantity_1099L','missing_count']):,} missing.
- `t__observed_active_credit_count`: {int(account_lookup.loc['t__observed_active_credit_count','non_missing_count']):,} non-missing ({account_lookup.loc['t__observed_active_credit_count','non_missing_percentage']:.6f}%); {int(account_lookup.loc['t__observed_active_credit_count','zero_count']):,} observed zeros.
- `bureau_history_years`: {int(history_lookup['valid_bureau_history_years']):,} valid ({percent(int(history_lookup['valid_bureau_history_years']), len(train)):.6f}%); {len(train)-int(history_lookup['valid_bureau_history_years']):,} missing.

## Joint coverage

- `credquantity_1099L` + history both valid: {int(joint_lookup.loc[('credquantity_1099L','both_valid'),'count']):,} ({joint_lookup.loc[('credquantity_1099L','both_valid'),'percentage']:.6f}% of TRAIN).
- `t__observed_active_credit_count` + history both valid: {int(joint_lookup.loc[('t__observed_active_credit_count','both_valid'),'count']):,} ({joint_lookup.loc[('t__observed_active_credit_count','both_valid'),'percentage']:.6f}% of TRAIN).

## Date validity

The construction excluded {int(history_lookup['source_dates_after_decision_rows']):,} source rows dated after decision. Before exclusion, {int(history_lookup['negative_history_before_invalid_exclusion']):,} applications had an earliest parseable bureau date after decision. There were {int(history_lookup['invalid_or_unparsable_source_date_rows']):,} invalid/unparsable source-date rows and {int(history_lookup['zero_day_history']):,} valid zero-day histories.

## Outputs and validation

The output directory contains the ten specified CSV/JSON/Markdown files. Focused and repository test-suite results are reported separately in the completion response; no test result is invented in this report.

No account-count variable, threshold, binary thin-file flag, two-dimensional feasibility decision, or final group was selected. No target was loaded for this audit, and no final-evaluation target, prediction, performance, subgroup result, or economic result was accessed. No visualisation was created.

`credquantity_1099L` uses the established Task 7 rule that conflicting linked finite values remain unresolved/missing. This source-definition limitation is retained factually and not repaired or replaced.
"""

    temp_dir = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    temp_dir.mkdir(parents=True, exist_ok=False)
    atomic_csv(temp_dir / "foundation_verification.csv", foundation_frame)
    atomic_csv(temp_dir / "account_count_candidate_audit.csv", account_frame)
    atomic_csv(temp_dir / "account_count_top_values.csv", top_frame)
    atomic_csv(temp_dir / "bureau_history_audit.csv", history_frame)
    atomic_csv(temp_dir / "thin_joint_coverage.csv", joint_frame)
    atomic_csv(temp_dir / "case_id_integrity.csv", integrity_frame)
    atomic_csv(temp_dir / "input_hashes_before_after.csv", hash_frame)
    atomic_csv(temp_dir / "validation_results.csv", validation_frame)
    atomic_json(temp_dir / "run_config.json", config)
    atomic_text(temp_dir / "task11_part1_report.md", report)
    produced = {path.name for path in temp_dir.iterdir() if path.is_file()}
    if produced != OUTPUT_NAMES:
        raise AuditBlocked(f"Output inventory mismatch: {sorted(produced)}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_dir, output_dir)
    return {"status": "COMPLETE", "output_dir": str(output_dir), "rows": len(train),
            "protected_hashes": len(hash_frame), "produced_files": sorted(produced)}


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    output_dir = (args.output_dir or data_root / "audits/task11/part1").resolve()
    result = run(data_root, output_dir, repo_root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
