#!/usr/bin/env python3
"""Run and independently verify the frozen Task 12 final evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, Iterable

for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
):
    os.environ.setdefault(_thread_variable, "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

import audit_thin_file_candidates as part1
import calibrate_baseline_models as task10
import freeze_task11_analysis_contract as part3
from task10_calibration import apply_candidate
from train_baseline_models import (
    TabularMLP,
    class_one_probability,
    feature_contract_for,
    load_matrix_rows,
    predict_mlp,
    torch_load_state,
)


VERSION = "1.0.0"
TASK_NAME = "TASK_12_FORMAL_FINAL_EVALUATION"
EXPECTED_TOTAL = 1_526_659
EXPECTED_FINAL = 228_999
EXPECTED_FINGERPRINT = part3.EXPECTED_FINGERPRINT
# The initial console capture used a literal ``\\n`` delimiter and produced
# f8273e4e...; the canonical newline-delimited digest below covers the same 75
# paths and identical individual SHA-256 values.
PRE_PHASE_A_PROTECTED_REGISTRY_DIGEST = "f5087d325ef11936d2eeab42723c65ed56c7d5a35a07868b75040ec9c3489d97"
PRE_PHASE_A_LEGACY_REGISTRY_DIGEST = "f8273e4eab0432f84fce862c79633e32a9913286c156522031e5ccea5ff95f3f"
MODEL_ORDER = list(part3.MODEL_ORDER)
SELECTED_METHODS = dict(part3.SELECTED_METHODS)
ALGORITHM_PAIRS = dict(part3.ALGORITHM_PAIRS)
PRIMARY_COMPARISON = list(part3.PRIMARY_COMPARISON)
SECONDARY_COMPARISON = list(part3.SECONDARY_COMPARISON)
STABLE_TIE_BREAK_RULE = part3.STABLE_TIE_BREAK_RULE
ACCOUNT_GROUPS = list(part3.ACCOUNT_GROUPS)
HISTORY_GROUPS = list(part3.HISTORY_GROUPS)
AD_GROUPS = list(part3.AD_GROUPS)
AD_INDICATORS = list(part3.AD_INDICATORS)
CORE_OUTPUT_NAMES = set(part3.FUTURE_OUTPUT_NAMES)
SUPPLEMENTAL_OUTPUT_NAMES = {
    "overall_model_metrics.csv",
    "overall_incremental_gains.csv",
    "pd_reassessment_summary.csv",
    "risk_quintile_migration.csv",
    "approval_switch_summary.csv",
    "subgroup_integrity.csv",
    "economic_complexity_comparisons.csv",
}
AUDIT_OUTPUT_NAMES = {
    "foundation_verification.csv",
    "protected_hashes_before_after.csv",
    "validation_results.csv",
    "run_config.json",
    "task12_final_evaluation_report.md",
}
OUTPUT_NAMES = CORE_OUTPUT_NAMES | SUPPLEMENTAL_OUTPUT_NAMES | AUDIT_OUTPUT_NAMES
README_START = "<!-- TASK12_FINAL_EVALUATION_START -->"
README_END = "<!-- TASK12_FINAL_EVALUATION_END -->"
QUANTILE_CONVENTION = "numpy.quantile(method='linear')"


class Task12Blocked(RuntimeError):
    """The final evaluation cannot safely continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--focused-test-result", default="not supplied")
    parser.add_argument("--full-test-result", default="not supplied")
    parser.add_argument("--verify-completed", action="store_true")
    parser.add_argument("--record-post-run", action="store_true")
    parser.add_argument("--repair-economic-output", action="store_true")
    parser.add_argument("--post-focused-test-result", default="not supplied")
    parser.add_argument("--post-full-test-result", default="not supplied")
    parser.add_argument("--independent-verifier-result", default="not supplied")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def atomic_text(path: Path, value: str) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(value, encoding="utf-8")
    os.replace(temp, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(json_safe(value), indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temp, index=False, lineterminator="\n")
    os.replace(temp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temp, compression="zstd")
    os.replace(temp, path)


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
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA or value is None:
        return None
    return value


def information_set(model_id: str) -> str:
    return "T_plus_AD" if model_id.endswith("_plus_AD") else "T"


def algorithm_family(model_id: str) -> str:
    return "lightgbm" if model_id.startswith("lightgbm") else model_id.split("_", 1)[0]


def protected_paths(data_root: Path, repo_root: Path) -> list[tuple[str, str, Path]]:
    records = list(part3.protected_paths(data_root, repo_root))
    records.extend([
        ("task11_part3_script", "Task 11 Part 3", repo_root / "scripts/freeze_task11_analysis_contract.py"),
        ("task11_part3_test", "Task 11 Part 3", repo_root / "tests/test_task11_part3.py"),
    ])
    part3_dir = data_root / "audits/task11/part3"
    if not part3_dir.is_dir():
        raise Task12Blocked(f"Missing Task 11 Part 3 output directory: {part3_dir}")
    part3_inventory = {path.name for path in part3_dir.iterdir() if path.is_file()}
    if part3_inventory != part3.OUTPUT_NAMES:
        raise Task12Blocked(f"Task 11 Part 3 inventory mismatch: {sorted(part3_inventory)}")
    for path in sorted(part3_dir.iterdir()):
        if path.is_file():
            records.append((f"task11_part3_output::{path.name}", "Task 11 Part 3", path))
    unique = {str(path.resolve()): (artifact, component, path.resolve()) for artifact, component, path in records}
    missing = [path for path in unique if not Path(path).is_file()]
    if missing:
        raise Task12Blocked(f"Protected files missing: {missing}")
    return [unique[path] for path in sorted(unique)]


def hash_paths(records: Iterable[tuple[str, str, Path]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for artifact, component, path in records:
        resolved = str(path.resolve())
        result[resolved] = {
            "artifact": artifact,
            "component_task": component,
            "protected_path": resolved,
            "sha256": sha256_file(path),
        }
    return result


def protected_registry_digest(hashes: dict[str, dict[str, str]]) -> str:
    lines = [f"{path}\t{hashes[path]['sha256']}" for path in sorted(hashes)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def verify_phase_a_readme(readme: str) -> list[dict[str, Any]]:
    requirements = {
        "phase_a_status": "Status: Final-evaluation ready." in readme,
        "six_models_frozen": "Six model systems are trained and frozen" in readme,
        "calibration_complete": "Calibration is complete" in readme,
        "task11_contract": "Task 11 freezes the final" in readme,
        "subgroups": all(token in readme for token in ("account count", "bureau history", "observed AD-module richness")),
        "primary_comparison": "`lightgbm_T` versus `lightgbm_T_plus_AD`" in readme,
        "approval_grid": "40%–80%" in readme and "0.25-point" in readme,
        "risk_grid": "0.75%–2.00%" in readme and "0.01-point" in readme,
        "economic_grid": "ratios are 2, 5, 10, 20, 30, and 50" in readme,
        "binary_target_boundary": "provided binary default target" in readme and "month-specific default horizon" in readme,
        "final_not_run": "Task 12 has not yet been executed" in readme,
        "no_private_absolute_path": "/Users/" not in readme,
        "no_chronological_plan": "chronological" not in readme.lower(),
        "bounded_section": readme.count(README_START) == 1 and readme.count(README_END) == 1,
    }
    rows = []
    for check, passed in requirements.items():
        rows.append({
            "check_id": f"readme::{check}", "component": "Phase A README", "observed": bool(passed),
            "expected": True, "status": "PASS" if passed else "FAIL", "notes": "Verified before final-data loading.",
        })
    if not all(requirements.values()):
        raise Task12Blocked(f"Phase A README requirements failed: {[key for key, value in requirements.items() if not value]}")
    return rows


def verify_foundation(data_root: Path, repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, component: str, observed: Any, expected: Any, passed: bool, notes: str = "") -> None:
        rows.append({
            "check_id": check_id, "component": component, "observed": json.dumps(json_safe(observed), sort_keys=True),
            "expected": json.dumps(json_safe(expected), sort_keys=True), "status": "PASS" if passed else "FAIL", "notes": notes,
        })

    part3_result = part3.verify_completed(data_root, data_root / "audits/task11/part3", repo_root)
    add("task11_part3", "Task 11", part3_result["status"], "PASS", part3_result["status"] == "PASS")
    foundation = part1.verify_foundation(data_root)
    foundation_ok = all(item["status"] == "PASS" for item in foundation.rows)
    add("task8_10_foundation", "Tasks 8-10", len(foundation.rows), 27, foundation_ok and len(foundation.rows) == 27)
    task10_saved = task10.verify_saved_run(data_root, "first_full", "first_full_v3", MODEL_ORDER)
    methods = {item["model_id"]: item["selected_method"] for item in task10_saved["models"]}
    add("task10_saved_run", "Task 10", task10_saved["status"], "PASS", task10_saved["status"] == "PASS")
    add("task10_not_previously_predicted", "Task 10", task10_saved["final_evaluation_status"], "NOT_PREDICTED", task10_saved["final_evaluation_status"] == "NOT_PREDICTED")
    add("model_order", "Task 9", [item["model_id"] for item in task10_saved["models"]], MODEL_ORDER, [item["model_id"] for item in task10_saved["models"]] == MODEL_ORDER)
    add("selected_methods", "Task 10", methods, SELECTED_METHODS, methods == SELECTED_METHODS)
    part3_contract = json.loads((data_root / "audits/task11/part3/analysis_contract.json").read_text(encoding="utf-8"))
    add("part3_fingerprint", "Task 11 Part 3", part3_contract["frozen_foundation"]["membership_fingerprint"], EXPECTED_FINGERPRINT, part3_contract["frozen_foundation"]["membership_fingerprint"] == EXPECTED_FINGERPRINT)
    add("part3_models", "Task 11 Part 3", part3_contract["frozen_foundation"]["model_order"], MODEL_ORDER, part3_contract["frozen_foundation"]["model_order"] == MODEL_ORDER)
    add("part3_methods", "Task 11 Part 3", part3_contract["frozen_foundation"]["selected_calibration_methods"], SELECTED_METHODS, part3_contract["frozen_foundation"]["selected_calibration_methods"] == SELECTED_METHODS)
    if any(item["status"] != "PASS" for item in rows):
        raise Task12Blocked(f"Foundation verification failed: {[item for item in rows if item['status'] != 'PASS']}")
    return rows, {"foundation": foundation, "task10_saved": task10_saved, "part3_contract": part3_contract}


def load_final_membership(data_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    path = data_root / "interim/task08_followup/application_manifest.parquet"
    metadata = pd.read_parquet(path, columns=["case_id", "base_order", "outer_split", "validation_role"])
    if len(metadata) != EXPECTED_TOTAL or metadata.case_id.isna().any() or not metadata.case_id.is_unique:
        raise Task12Blocked("Frozen manifest population/key contract failed")
    base_order = metadata.base_order.to_numpy(np.int64)
    if not np.array_equal(base_order, np.arange(EXPECTED_TOTAL, dtype=np.int64)):
        raise Task12Blocked("Frozen manifest base_order is not canonical")
    fingerprint = sha256_lines(
        f"{case}\t{outer}\t{role}" for case, outer, role in
        zip(metadata.case_id, metadata.outer_split, metadata.validation_role)
    )
    if fingerprint != EXPECTED_FINGERPRINT:
        raise Task12Blocked(f"Frozen membership fingerprint mismatch: {fingerprint}")
    split_counts = metadata.outer_split.value_counts().to_dict()
    expected_splits = {"train": 1_068_661, "validation": 228_999, "evaluation": 228_999}
    if split_counts != expected_splits:
        raise Task12Blocked(f"Split-count contradiction: {split_counts}")
    final_positions = np.flatnonzero(metadata.outer_split.eq("evaluation").to_numpy())
    train_positions = np.flatnonzero(metadata.outer_split.eq("train").to_numpy())
    validation_positions = np.flatnonzero(metadata.outer_split.eq("validation").to_numpy())
    if len(final_positions) != EXPECTED_FINAL or np.intersect1d(final_positions, train_positions).size or np.intersect1d(final_positions, validation_positions).size:
        raise Task12Blocked("Final membership count or split separation failed")
    final = metadata.iloc[final_positions][["case_id", "base_order"]].reset_index(drop=True)
    if final.case_id.duplicated().any() or not np.array_equal(final.base_order.to_numpy(np.int64), final_positions):
        raise Task12Blocked("Final case/order integrity failed")
    return final, final_positions, metadata.case_id.to_numpy(), {
        "manifest_path": str(path), "membership_fingerprint": fingerprint, "split_counts": split_counts,
        "final_count": len(final), "final_unique_cases": int(final.case_id.nunique()),
        "train_final_overlap": 0, "validation_final_overlap": 0,
    }


def build_subgroups(data_root: Path, final: pd.DataFrame, final_positions: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_path = data_root / "interim/task08/application_features.parquet"
    columns = ["case_id", "date_decision", "t__observed_active_credit_count", *AD_INDICATORS]
    features = pd.read_parquet(feature_path, columns=columns)
    selected = features.iloc[final_positions].reset_index(drop=True)
    if not np.array_equal(selected.case_id.to_numpy(), final.case_id.to_numpy()):
        raise Task12Blocked("Pre-model feature rows do not align with final membership/order")
    account_numeric = pd.to_numeric(selected["t__observed_active_credit_count"], errors="coerce").astype(float)
    account = pd.Series(pd.NA, index=selected.index, dtype="object")
    account.loc[account_numeric.eq(0)] = "0"
    account.loc[account_numeric.eq(1)] = "1"
    account.loc[account_numeric.eq(2)] = "2"
    account.loc[account_numeric.ge(3) & np.isfinite(account_numeric)] = "3_plus"
    invalid_account = account_numeric.notna() & (~np.isfinite(account_numeric) | account_numeric.lt(0) | ~np.isclose(account_numeric, np.round(account_numeric)))
    account.loc[invalid_account] = pd.NA
    module_values = selected[AD_INDICATORS].apply(pd.to_numeric, errors="coerce")
    if module_values.isna().any().any() or not module_values.isin([0, 1]).all().all():
        raise Task12Blocked("Final AD-richness indicators are not complete binary pre-model values")
    richness = module_values.astype(np.int8).sum(axis=1)
    if not richness.isin([0, 1, 2, 3]).all():
        raise Task12Blocked("Final observed AD-module count is outside 0-3")
    decisions = pd.to_datetime(selected.date_decision, errors="coerce")
    if decisions.isna().any():
        raise Task12Blocked("Final decision date parsing failed")
    final_index = pd.Index(final.case_id.to_numpy(), name="case_id")
    decision_ns = decisions.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    inputs = part1.resolve_inputs(data_root)
    active = part1.aggregate_date_source(inputs["active_files"], "active", "contractdate_551D", final_index, decision_ns)
    closed = part1.aggregate_date_source(inputs["closed_files"], "closed", "dateofcredstart_739D", final_index, decision_ns)
    history = part1.combine_bureau_history(decision_ns, active, closed)
    history_years = pd.Series(history["years"], index=selected.index, dtype="float64")
    history_labels, history_grouping = part2_assign_history(history_years)
    if history_labels.isna().any():
        raise Task12Blocked("Final bureau-history grouping has unassigned rows")
    groups = final.copy()
    groups["account_group"] = account
    groups["bureau_history_group"] = history_labels
    groups["ad_richness_group"] = richness.astype(str)
    details = {
        "feature_path": str(feature_path),
        "account_excluded": int(groups.account_group.isna().sum()),
        "history_pre_fill_missing": int(history_years.isna().sum()),
        "history_excluded": int(groups.bureau_history_group.isna().sum()),
        "ad_richness_excluded": int(groups.ad_richness_group.isna().sum()),
        "history_post_decision_rows_excluded": {
            "credit_bureau_b_1": int(active.after_rows), "credit_bureau_a_1": int(closed.after_rows),
        },
        "history_grouping_minimum": float(history_grouping.min()),
        "history_grouping_maximum": float(history_grouping.max()),
    }
    return groups, details


def part2_assign_history(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    grouping = pd.to_numeric(values, errors="coerce").astype(float).fillna(0.0)
    labels = pd.Series(pd.NA, index=values.index, dtype="object")
    rules = [
        ("0_to_0_5", grouping.ge(0) & grouping.le(0.5)),
        ("0_5_to_1", grouping.gt(0.5) & grouping.le(1)),
        ("1_to_2", grouping.gt(1) & grouping.le(2)),
        ("2_to_3", grouping.gt(2) & grouping.le(3)),
        ("3_to_4", grouping.gt(3) & grouping.le(4)),
        ("4_to_5", grouping.gt(4) & grouping.le(5)),
        ("gt_5", grouping.gt(5)),
    ]
    for label, mask in rules:
        labels.loc[mask] = label
    return labels, grouping


@contextmanager
def no_fit_guard() -> Iterable[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        raise Task12Blocked("A prohibited fit/retrain/recalibration method was called during Task 12")

    originals = {
        "logistic_fit": LogisticRegression.fit,
        "isotonic_fit": IsotonicRegression.fit,
        "lgb_train": lgb.train,
        "lgb_classifier_fit": lgb.LGBMClassifier.fit,
    }
    LogisticRegression.fit = blocked  # type: ignore[method-assign]
    IsotonicRegression.fit = blocked  # type: ignore[method-assign]
    lgb.train = blocked  # type: ignore[assignment]
    lgb.LGBMClassifier.fit = blocked  # type: ignore[method-assign]
    try:
        yield
    finally:
        LogisticRegression.fit = originals["logistic_fit"]  # type: ignore[method-assign]
        IsotonicRegression.fit = originals["isotonic_fit"]  # type: ignore[method-assign]
        lgb.train = originals["lgb_train"]  # type: ignore[assignment]
        lgb.LGBMClassifier.fit = originals["lgb_classifier_fit"]  # type: ignore[method-assign]


def validate_probability(values: np.ndarray, expected_rows: int) -> dict[str, Any]:
    probability = np.asarray(values, dtype=np.float64)
    result = {
        "rows": len(probability),
        "missing_count": int(np.isnan(probability).sum()),
        "nonfinite_count": int((~np.isfinite(probability)).sum()),
        "below_zero_count": int((probability < 0).sum()),
        "above_one_count": int((probability > 1).sum()),
        "minimum": float(np.min(probability)),
        "maximum": float(np.max(probability)),
        "mean": float(np.mean(probability)),
    }
    result["status"] = "PASS" if result["rows"] == expected_rows and sum(result[key] for key in ("missing_count", "nonfinite_count", "below_zero_count", "above_one_count")) == 0 else "FAIL"
    if result["status"] != "PASS":
        raise Task12Blocked(f"Probability integrity failed: {result}")
    return result


def infer_selected_predictions(
    data_root: Path, final: pd.DataFrame, final_positions: np.ndarray, manifest_keys: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prep = data_root / "interim/task08_followup"
    feature_sets = json.loads((prep / "preprocessing/feature_sets.json").read_text(encoding="utf-8"))
    preprocessor = json.loads((prep / "preprocessing/preprocessor.json").read_text(encoding="utf-8"))
    task09_registry = json.loads((data_root / "audits/task09/first_full/selected_model_registry.json").read_text(encoding="utf-8"))
    task10_registry = json.loads((data_root / "audits/task10/first_full_v3/calibration_registry.json").read_text(encoding="utf-8"))
    task09_by_id = {item["model_id"]: item for item in task09_registry["models"]}
    task10_by_id = {item["model_id"]: item for item in task10_registry["models"]}
    if list(task09_by_id) != MODEL_ORDER or list(task10_by_id) != MODEL_ORDER:
        raise Task12Blocked("Task 9/10 model registry order mismatch")
    authoritative_outputs = {representation: feature_sets[representation] for representation in ("linear_nn", "gbdt")}
    if preprocessor.get("output_features") != authoritative_outputs:
        raise Task12Blocked("Frozen preprocessor output-feature contract differs from feature_sets.json")
    matrix_cache: dict[str, np.ndarray] = {}
    matrix_audit: dict[str, Any] = {}
    for representation, matrix_name, dtype in (
        ("linear_nn", "linear_nn_inputs.parquet", np.float64),
        ("gbdt", "gbdt_inputs.parquet", np.float32),
    ):
        full_features = list(feature_sets[representation]["T_plus_AD"])
        matrix, info = load_matrix_rows(prep / matrix_name, full_features, final_positions, manifest_keys, dtype)
        if info["contains_infinity"] or (representation == "linear_nn" and info["contains_nan"]):
            raise Task12Blocked(f"Frozen prepared matrix numeric contract failed: {representation}")
        matrix_cache[representation] = matrix
        matrix_audit[representation] = info
    output = final.copy()
    audit_rows: list[dict[str, Any]] = []
    with no_fit_guard(), threadpool_limits(limits=4):
        for model_id in MODEL_ORDER:
            started = time.perf_counter()
            item09, item10 = task09_by_id[model_id], task10_by_id[model_id]
            family, representation, expected_features = feature_contract_for(model_id, feature_sets)
            if item09["family"] != family or item09["representation"] != representation or item09["ordered_predictors"] != expected_features:
                raise Task12Blocked(f"Task 9 registered feature contract mismatch: {model_id}")
            if item10["ordered_predictors"] != expected_features or item10["selected_method"] != SELECTED_METHODS[model_id]:
                raise Task12Blocked(f"Task 10 registered feature/calibration contract mismatch: {model_id}")
            full_features = list(feature_sets[representation]["T_plus_AD"])
            indices = [full_features.index(column) for column in expected_features]
            values = matrix_cache[representation][:, indices]
            model_path = Path(item09["model_path"])
            if sha256_file(model_path) != item09["model_sha256"]:
                raise Task12Blocked(f"Task 9 model hash mismatch: {model_id}")
            if family == "logit":
                model = joblib.load(model_path)
                if list(model.classes_) != [0, 1] or int(model.n_features_in_) != len(expected_features):
                    raise Task12Blocked(f"Logit reload contract failed: {model_id}")
                raw = class_one_probability(model, values)
            elif family == "lightgbm":
                model = lgb.Booster(model_file=str(model_path))
                iteration = int(item09["selected_iteration_or_epoch"])
                if model.num_feature() != len(expected_features) or model.current_iteration() != iteration:
                    raise Task12Blocked(f"LightGBM reload contract failed: {model_id}")
                raw = np.asarray(model.predict(values, num_iteration=iteration), dtype=np.float64)
            else:
                architecture_path = Path(item09["architecture_path"])
                architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
                if int(architecture["input_dim"]) != len(expected_features):
                    raise Task12Blocked(f"MLP architecture contract failed: {model_id}")
                model = TabularMLP(len(expected_features))
                model.load_state_dict(torch_load_state(model_path))
                model.eval()
                raw = predict_mlp(model, values.astype(np.float32, copy=False))
            raw_audit = validate_probability(raw, EXPECTED_FINAL)
            calibrator_path = Path(item10["selected_calibrator_path"])
            if sha256_file(calibrator_path) != item10["selected_calibrator_sha256"]:
                raise Task12Blocked(f"Selected calibrator hash mismatch: {model_id}")
            method = SELECTED_METHODS[model_id]
            calibrator = task10.load_calibrator(method, calibrator_path)
            selected = np.asarray(apply_candidate(method, calibrator, raw), dtype=np.float64)
            selected_audit = validate_probability(selected, EXPECTED_FINAL)
            output[f"calibrated_pd__{model_id}"] = selected
            audit_rows.append({
                "model_id": model_id,
                "family": family,
                "representation": representation,
                "feature_count": len(expected_features),
                "selected_calibration_method": method,
                "model_path": str(model_path),
                "model_sha256": item09["model_sha256"],
                "calibrator_path": str(calibrator_path),
                "calibrator_sha256": item10["selected_calibrator_sha256"],
                "raw_probability_minimum": raw_audit["minimum"],
                "raw_probability_maximum": raw_audit["maximum"],
                "selected_probability_minimum": selected_audit["minimum"],
                "selected_probability_maximum": selected_audit["maximum"],
                "selected_probability_mean": selected_audit["mean"],
                "calibrator_application_count": 1,
                "fit_calls": 0,
                "elapsed_seconds": time.perf_counter() - started,
                "status": "PASS",
            })
            del values, model, raw, selected, calibrator
            gc.collect()
    del matrix_cache
    gc.collect()
    expected_columns = ["case_id", "base_order", *[f"calibrated_pd__{model}" for model in MODEL_ORDER]]
    if list(output.columns) != expected_columns:
        raise Task12Blocked("Selected prediction schema/order mismatch before target join")
    return output, pd.DataFrame(audit_rows)


def load_and_join_final_target(data_root: Path, prediction: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = data_root / "interim/task08_followup/application_manifest.parquet"
    target = pd.read_parquet(
        path, columns=["case_id", "base_order", "target", "outer_split"], filters=[("outer_split", "==", "evaluation")],
    ).sort_values("base_order", kind="stable").reset_index(drop=True)
    if len(target) != EXPECTED_FINAL or target.case_id.duplicated().any() or set(target.target.unique()) != {0, 1}:
        raise Task12Blocked("Final target population/binary contract failed")
    if not np.array_equal(target.case_id.to_numpy(), prediction.case_id.to_numpy()) or not np.array_equal(target.base_order.to_numpy(), prediction.base_order.to_numpy()):
        raise Task12Blocked("Final target rows do not align with completed prediction rows")
    result = prediction.copy()
    result.insert(2, "target", target.target.to_numpy(np.int8))
    positives = int(result.target.sum())
    return result, {
        "rows": len(result), "unique_case_ids": int(result.case_id.nunique()), "positives": positives,
        "negatives": len(result) - positives, "observed_default_rate": positives / len(result),
        "target_loaded_after_predictions_completed": True,
    }


def binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    n, positives = len(y), int(y.sum())
    negatives = n - positives
    both = positives > 0 and negatives > 0
    return {
        "sample_count": n,
        "positive_count": positives,
        "negative_count": negatives,
        "observed_default_rate": positives / n if n else np.nan,
        "mean_calibrated_pd": float(np.mean(p)) if n else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if both else np.nan,
        "average_precision": float(average_precision_score(y, p)) if both else np.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if n else np.nan,
        "brier_score": float(brier_score_loss(y, p)) if n else np.nan,
        "metric_status": "PASS" if both else "PARTIAL_SINGLE_CLASS",
        "missing_metric_reason": "" if both else "ROC-AUC and average precision undefined because the group has one target class",
    }


def overall_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    y = predictions.target.to_numpy(np.int8)
    rows = []
    for model in MODEL_ORDER:
        metrics = binary_metrics(y, predictions[f"calibrated_pd__{model}"].to_numpy())
        rows.append({
            "model_id": model, "information_set": information_set(model), "algorithm_family": algorithm_family(model),
            "selected_calibration_method": SELECTED_METHODS[model], "n_cases": metrics.pop("sample_count"),
            "n_defaults": metrics.pop("positive_count"), "n_non_defaults": metrics.pop("negative_count"), **metrics,
        })
    return pd.DataFrame(rows)


def gain_row(reference: pd.Series, candidate: pd.Series, family: str) -> dict[str, Any]:
    return {
        "algorithm_family": family,
        "t_model_id": reference.model_id,
        "t_plus_ad_model_id": candidate.model_id,
        "sample_count": int(reference.get("n_cases", reference.get("sample_count"))),
        "roc_auc_gain": candidate.roc_auc - reference.roc_auc,
        "average_precision_gain": candidate.average_precision - reference.average_precision,
        "log_loss_gain": reference.log_loss - candidate.log_loss,
        "brier_gain": reference.brier_score - candidate.brier_score,
        "mean_calibrated_pd_change": candidate.mean_calibrated_pd - reference.mean_calibrated_pd,
        "gain_status": "PASS" if reference.metric_status == candidate.metric_status == "PASS" else "PARTIAL",
        "missing_metric_reason": "" if reference.metric_status == candidate.metric_status == "PASS" else "At least one paired metric is unavailable",
    }


def overall_gains(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index("model_id", drop=False)
    return pd.DataFrame([gain_row(indexed.loc[pair[0]], indexed.loc[pair[1]], family) for family, pair in ALGORITHM_PAIRS.items()])


def subgroup_definitions(predictions: pd.DataFrame) -> list[tuple[str, str, str, int, np.ndarray]]:
    definitions: list[tuple[str, str, str, int, np.ndarray]] = []
    for dimension, column, groups in (
        ("account_count", "account_group", ACCOUNT_GROUPS),
        ("bureau_history", "bureau_history_group", HISTORY_GROUPS),
        ("ad_richness", "ad_richness_group", AD_GROUPS),
    ):
        values = predictions[column]
        for item in groups:
            definitions.append((dimension, column, item["label"], int(item["order"]), values.eq(item["label"]).to_numpy()))
    return definitions


def subgroup_outputs(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    dimensions = {
        "account_count": ("account_group", ACCOUNT_GROUPS),
        "bureau_history": ("bureau_history_group", HISTORY_GROUPS),
        "ad_richness": ("ad_richness_group", AD_GROUPS),
    }
    for dimension, (column, groups) in dimensions.items():
        eligible = predictions[column].notna()
        eligible_count = int(eligible.sum())
        excluded = len(predictions) - eligible_count
        reconciled = 0
        for item in groups:
            mask = predictions[column].eq(item["label"]).to_numpy()
            n = int(mask.sum())
            reconciled += n
            target = predictions.loc[mask, "target"].to_numpy(np.int8)
            positives = int(target.sum())
            integrity_rows.append({
                "group_dimension": dimension, "group_order": item["order"], "group_label": item["label"],
                "eligible_dimension_count": eligible_count, "group_case_count": n,
                "share_of_eligible_cases": n / eligible_count if eligible_count else np.nan,
                "share_of_full_final_population": n / len(predictions), "target_positive_count": positives,
                "observed_default_rate": positives / n if n else np.nan, "excluded_from_dimension_count": excluded,
                "dimension_reconciled_count": 0, "intended_dimension_denominator": eligible_count,
                "reconciliation_status": "PENDING",
            })
            for model in MODEL_ORDER:
                result = binary_metrics(target, predictions.loc[mask, f"calibrated_pd__{model}"].to_numpy())
                metric_rows.append({
                    "group_dimension": dimension, "group_order": item["order"], "group_label": item["label"],
                    "model_id": model, "selected_calibration_method": SELECTED_METHODS[model], **result,
                })
        if reconciled != eligible_count:
            raise Task12Blocked(f"Subgroup dimension failed reconciliation: {dimension}")
        for row in integrity_rows:
            if row["group_dimension"] == dimension:
                row["dimension_reconciled_count"] = reconciled
                row["reconciliation_status"] = "PASS"
    metrics = pd.DataFrame(metric_rows)
    gain_rows: list[dict[str, Any]] = []
    for (dimension, order, label), group in metrics.groupby(["group_dimension", "group_order", "group_label"], sort=False):
        indexed = group.set_index("model_id", drop=False)
        for family, pair in ALGORITHM_PAIRS.items():
            gain_rows.append({
                "group_dimension": dimension, "group_order": order, "group_label": label,
                **gain_row(indexed.loc[pair[0]], indexed.loc[pair[1]], family),
            })
    return metrics, pd.DataFrame(gain_rows), pd.DataFrame(integrity_rows)


def stable_order(probability: np.ndarray, base_order: np.ndarray) -> np.ndarray:
    return np.lexsort((np.asarray(base_order, dtype=np.int64), np.asarray(probability, dtype=np.float64)))


def risk_percentile(probability: np.ndarray, base_order: np.ndarray) -> np.ndarray:
    order = stable_order(probability, base_order)
    result = np.empty(len(order), dtype=np.float64)
    if len(order) <= 1:
        result[order] = 0.0
    else:
        result[order] = np.arange(len(order), dtype=np.float64) / (len(order) - 1)
    return result


def risk_quintile(probability: np.ndarray, base_order: np.ndarray) -> np.ndarray:
    order = stable_order(probability, base_order)
    result = np.empty(len(order), dtype=np.int8)
    result[order] = np.minimum(5, (np.arange(len(order), dtype=np.int64) * 5 // len(order)) + 1)
    return result


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90], method="linear")
    return {
        "n_cases": len(array), "mean": float(array.mean()), "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()), "p10": float(quantiles[0]), "p25": float(quantiles[1]),
        "median": float(quantiles[2]), "p75": float(quantiles[3]), "p90": float(quantiles[4]),
        "maximum": float(array.max()), "share_below_zero": float(np.mean(array < 0)),
        "share_exactly_zero": float(np.mean(array == 0)), "share_above_zero": float(np.mean(array > 0)),
        "quantile_convention": QUANTILE_CONVENTION,
    }


def strata(predictions: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    result = [("overall", "ALL", np.ones(len(predictions), dtype=bool))]
    for dimension, _column, label, _order, mask in subgroup_definitions(predictions):
        result.append((dimension, label, mask))
    return result


def reassessment_and_migration(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_order = predictions.base_order.to_numpy(np.int64)
    target = predictions.target.to_numpy(np.int8)
    summaries: list[dict[str, Any]] = []
    migrations: list[dict[str, Any]] = []
    all_strata = strata(predictions)
    for family, pair in ALGORITHM_PAIRS.items():
        reference = predictions[f"calibrated_pd__{pair[0]}"].to_numpy(np.float64)
        candidate = predictions[f"calibrated_pd__{pair[1]}"].to_numpy(np.float64)
        delta_pd = candidate - reference
        reference_percentile = risk_percentile(reference, base_order)
        candidate_percentile = risk_percentile(candidate, base_order)
        delta_rank = candidate_percentile - reference_percentile
        reference_quintile = risk_quintile(reference, base_order)
        candidate_quintile = risk_quintile(candidate, base_order)
        for dimension, label, mask in all_strata:
            for measure, values in (("delta_pd", delta_pd), ("delta_risk_percentile", delta_rank)):
                summaries.append({
                    "algorithm_family": family, "t_model_id": pair[0], "t_plus_ad_model_id": pair[1],
                    "stratum_dimension": dimension, "stratum_label": label, "measure": measure,
                    "definition": "T_plus_AD minus T", **distribution_summary(values[mask]),
                })
            stratum_n = int(mask.sum())
            for origin in range(1, 6):
                for destination in range(1, 6):
                    cell = mask & (reference_quintile == origin) & (candidate_quintile == destination)
                    count = int(cell.sum())
                    defaults = int(target[cell].sum())
                    migrations.append({
                        "algorithm_family": family, "t_model_id": pair[0], "t_plus_ad_model_id": pair[1],
                        "stratum_dimension": dimension, "stratum_label": label,
                        "t_risk_quintile": origin, "t_plus_ad_risk_quintile": destination,
                        "case_count": count, "share_of_stratum": count / stratum_n if stratum_n else np.nan,
                        "target_default_count": defaults, "observed_default_rate": defaults / count if count else np.nan,
                        "quintile_assignment": "global sorted position; quintile 1 lowest risk and quintile 5 highest risk",
                    })
    return pd.DataFrame(summaries), pd.DataFrame(migrations)


def approval_curve(predictions: pd.DataFrame, grid: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    n = len(predictions)
    y = predictions.target.to_numpy(np.int8)
    base = predictions.base_order.to_numpy(np.int64)
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    for model in MODEL_ORDER:
        p = predictions[f"calibrated_pd__{model}"].to_numpy(np.float64)
        order = stable_order(p, base)
        sorted_p, sorted_y = p[order], y[order]
        cumulative_p = np.cumsum(sorted_p, dtype=np.float64)
        cumulative_y = np.cumsum(sorted_y, dtype=np.int64)
        cache[model] = {"order": order, "sorted_p": sorted_p, "sorted_y": sorted_y, "cumulative_p": cumulative_p, "cumulative_y": cumulative_y}
        for point in grid.itertuples(index=False):
            rate = Decimal(str(point.approval_rate_percent))
            approved = int((Decimal(n) * rate / Decimal(100)).to_integral_value(rounding=ROUND_FLOOR))
            defaults = int(cumulative_y[approved - 1])
            expected = float(cumulative_p[approved - 1])
            rows.append({
                "model_id": model, "selected_calibration_method": SELECTED_METHODS[model],
                "approval_rate_percent": float(rate), "approval_rate_probability": float(rate / Decimal(100)),
                "is_external_display_point": bool(point.is_external_display_point),
                "approved_count": approved, "rejected_count": n - approved,
                "actual_approval_rate_percent": 100 * approved / n,
                "boundary_calibrated_pd_cutoff": float(sorted_p[approved - 1]),
                "approved_pd_minimum": float(sorted_p[0]), "approved_pd_maximum": float(sorted_p[approved - 1]),
                "approved_portfolio_mean_calibrated_pd": expected / approved,
                "cumulative_expected_defaults": expected,
                "realised_approved_defaults": defaults,
                "realised_approved_non_defaults": approved - defaults,
                "realised_approved_default_rate": defaults / approved,
                "stable_tie_break_rule": STABLE_TIE_BREAK_RULE,
            })
    return pd.DataFrame(rows), cache


def approval_switches(predictions: pd.DataFrame, approval_cache: dict[str, dict[str, Any]], anchors: list[Decimal]) -> pd.DataFrame:
    n = len(predictions)
    y = predictions.target.to_numpy(np.int8)
    all_strata = strata(predictions)
    rows: list[dict[str, Any]] = []
    for family, pair in ALGORITHM_PAIRS.items():
        reference_pd = predictions[f"calibrated_pd__{pair[0]}"].to_numpy(np.float64)
        candidate_pd = predictions[f"calibrated_pd__{pair[1]}"].to_numpy(np.float64)
        delta = candidate_pd - reference_pd
        for anchor in anchors:
            approved_count = int((Decimal(n) * anchor / Decimal(100)).to_integral_value(rounding=ROUND_FLOOR))
            reference_approved = np.zeros(n, dtype=bool)
            candidate_approved = np.zeros(n, dtype=bool)
            reference_approved[approval_cache[pair[0]]["order"][:approved_count]] = True
            candidate_approved[approval_cache[pair[1]]["order"][:approved_count]] = True
            categories = {
                "both_approved": reference_approved & candidate_approved,
                "T_only_approved": reference_approved & ~candidate_approved,
                "T_plus_AD_only_approved": ~reference_approved & candidate_approved,
                "both_rejected": ~reference_approved & ~candidate_approved,
            }
            if int(categories["T_only_approved"].sum()) != int(categories["T_plus_AD_only_approved"].sum()):
                raise Task12Blocked(f"Overall approval swapped-count equality failed: {family}/{anchor}")
            for dimension, label, stratum_mask in all_strata:
                denominator = int(stratum_mask.sum())
                for category, category_mask in categories.items():
                    mask = stratum_mask & category_mask
                    count = int(mask.sum())
                    defaults = int(y[mask].sum())
                    rows.append({
                        "algorithm_family": family, "t_model_id": pair[0], "t_plus_ad_model_id": pair[1],
                        "approval_rate_percent": float(anchor), "approved_count_each_model": approved_count,
                        "stratum_dimension": dimension, "stratum_label": label, "switch_category": category,
                        "case_count": count, "share_of_stratum": count / denominator if denominator else np.nan,
                        "target_default_count": defaults, "observed_default_rate": defaults / count if count else np.nan,
                        "mean_t_calibrated_pd": float(reference_pd[mask].mean()) if count else np.nan,
                        "mean_t_plus_ad_calibrated_pd": float(candidate_pd[mask].mean()) if count else np.nan,
                        "mean_delta_pd": float(delta[mask].mean()) if count else np.nan,
                    })
    return pd.DataFrame(rows)


def risk_budget_curve(predictions: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    n = len(predictions)
    y = predictions.target.to_numpy(np.int8)
    base = predictions.base_order.to_numpy(np.int64)
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        p = predictions[f"calibrated_pd__{model}"].to_numpy(np.float64)
        order = stable_order(p, base)
        sorted_p, sorted_y = p[order], y[order]
        cumulative_p = np.cumsum(sorted_p, dtype=np.float64)
        cumulative_y = np.cumsum(sorted_y, dtype=np.int64)
        cumulative_mean = cumulative_p / np.arange(1, n + 1)
        if np.any(np.diff(cumulative_mean) < -1e-15):
            raise Task12Blocked(f"Cumulative mean PD is not monotonic: {model}")
        previous = -1
        for point in grid.itertuples(index=False):
            budget_percent = Decimal(str(point.risk_budget_percent))
            budget = float(Decimal(str(point.risk_budget_probability)))
            approved = int(np.searchsorted(cumulative_mean, budget, side="right"))
            if approved < previous:
                raise Task12Blocked(f"Risk approval count decreased with budget: {model}")
            previous = approved
            feasible = approved > 0
            mean_pd = float(cumulative_mean[approved - 1]) if feasible else np.nan
            defaults = int(cumulative_y[approved - 1]) if feasible else 0
            rows.append({
                "common_risk_budget_percent": float(budget_percent),
                "common_risk_budget_probability": budget,
                "is_external_display_point": bool(point.is_external_display_point),
                "model_id": model, "selected_calibration_method": SELECTED_METHODS[model],
                "maximum_feasible_approval_rate_percent": 100 * approved / n,
                "approved_count": approved, "rejected_count": n - approved,
                "boundary_calibrated_pd_cutoff": float(sorted_p[approved - 1]) if feasible else np.nan,
                "approved_portfolio_mean_calibrated_pd": mean_pd,
                "cumulative_expected_defaults": float(cumulative_p[approved - 1]) if feasible else 0.0,
                "realised_approved_defaults": defaults,
                "realised_approved_non_defaults": approved - defaults,
                "realised_approved_default_rate": defaults / approved if feasible else np.nan,
                "budget_slack": budget - mean_pd if feasible else np.nan,
                "feasibility_flag": feasible,
                "status": "FEASIBLE" if feasible else "NO_POSITIVE_PREFIX_FEASIBLE",
                "stable_tie_break_rule": STABLE_TIE_BREAK_RULE,
            })
    return pd.DataFrame(rows)


def economic_outputs(
    predictions: pd.DataFrame, scenarios: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(predictions)
    y = predictions.target.to_numpy(np.int8)
    result_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        p = predictions[f"calibrated_pd__{model}"].to_numpy(np.float64)
        for scenario in scenarios.itertuples(index=False):
            ratio = int(scenario.loss_to_gain_ratio)
            threshold = float(scenario.individual_pd_threshold_probability)
            approved_mask = p < threshold
            approved = int(approved_mask.sum())
            defaults = int(y[approved_mask].sum())
            nondefaults = approved - defaults
            predicted_values = (1 - p[approved_mask]) - p[approved_mask] * ratio
            predicted_total = float(predicted_values.sum())
            realised_total = float(nondefaults - defaults * ratio)
            task12_role = scenario.display_role if scenario.display_role != "detailed_only" else "additional_grid"
            result_rows.append({
                "scenario_id": scenario.scenario_id, "normalized_G": 1, "normalized_L": ratio,
                "loss_to_gain_ratio": ratio, "individual_pd_threshold_probability": threshold,
                "individual_pd_threshold_percent": float(scenario.individual_pd_threshold_percent),
                "display_role": scenario.display_role, "scenario_role": task12_role,
                "model_id": model, "information_set": information_set(model), "algorithm_family": algorithm_family(model),
                "selected_calibration_method": SELECTED_METHODS[model],
                "approved_count": approved, "rejected_count": n - approved, "approval_rate_percent": 100 * approved / n,
                "approved_non_defaults": nondefaults, "approved_defaults": defaults,
                "approved_realised_default_rate": defaults / approved if approved else np.nan,
                "approved_predicted_mean_pd": float(p[approved_mask].mean()) if approved else np.nan,
                "expected_scenario_value_total": predicted_total,
                "predicted_expected_value_per_evaluation_case": predicted_total / n,
                "predicted_expected_value_per_approved_case": predicted_total / approved if approved else np.nan,
                "realised_scenario_value_total": realised_total,
                "realised_scenario_value_per_evaluation_case": realised_total / n,
                "realised_scenario_value_per_approved_case": realised_total / approved if approved else np.nan,
                "approval_rule": "selected calibrated PD < economic threshold",
                "accounting_interpretation": "normalized G/L scenario value; not actual accounting profit or currency",
            })
    results = pd.DataFrame(result_rows)
    indexed = results.set_index(["model_id", "scenario_id"], drop=False)
    comparisons = [
        ("within_family_ad", "logit_T", "logit_T_plus_AD", "logit"),
        ("within_family_ad", "lightgbm_T", "lightgbm_T_plus_AD", "lightgbm"),
        ("within_family_ad", "mlp_T", "mlp_T_plus_AD", "mlp"),
        ("t_plus_ad_complexity", "logit_T_plus_AD", "lightgbm_T_plus_AD", "lightgbm_minus_logit"),
        ("t_plus_ad_complexity", "logit_T_plus_AD", "mlp_T_plus_AD", "mlp_minus_logit"),
        ("t_plus_ad_complexity", "lightgbm_T_plus_AD", "mlp_T_plus_AD", "mlp_minus_lightgbm"),
    ]
    incremental_rows: list[dict[str, Any]] = []
    for comparison_type, reference_model, candidate_model, comparison_id in comparisons:
        for scenario in scenarios.itertuples(index=False):
            reference = indexed.loc[(reference_model, scenario.scenario_id)]
            candidate = indexed.loc[(candidate_model, scenario.scenario_id)]
            realised_difference = candidate.realised_scenario_value_total - reference.realised_scenario_value_total
            within = comparison_type == "within_family_ad"
            incremental_rows.append({
                "comparison_type": comparison_type, "comparison_id": comparison_id,
                "scenario_id": scenario.scenario_id, "loss_to_gain_ratio": int(scenario.loss_to_gain_ratio),
                "display_role": scenario.display_role, "scenario_role": candidate.scenario_role,
                "algorithm_family": comparison_id if within else "cross_family_t_plus_ad",
                "reference_model_id": reference_model, "candidate_model_id": candidate_model,
                "t_model_id": reference_model if within else "",
                "t_plus_ad_model_id": candidate_model if within else "",
                "approved_count_reference": int(reference.approved_count), "approved_count_candidate": int(candidate.approved_count),
                "approved_count_change": int(candidate.approved_count - reference.approved_count),
                "approval_rate_percent_change": candidate.approval_rate_percent - reference.approval_rate_percent,
                "realised_default_count_change": int(candidate.approved_defaults - reference.approved_defaults),
                "realised_default_rate_change": candidate.approved_realised_default_rate - reference.approved_realised_default_rate,
                "predicted_expected_total_value_change": candidate.expected_scenario_value_total - reference.expected_scenario_value_total,
                "predicted_expected_value_per_evaluation_case_change": candidate.predicted_expected_value_per_evaluation_case - reference.predicted_expected_value_per_evaluation_case,
                "predicted_expected_value_per_approved_case_change": candidate.predicted_expected_value_per_approved_case - reference.predicted_expected_value_per_approved_case,
                "realised_total_value_change": realised_difference,
                "realised_value_per_evaluation_case_change": candidate.realised_scenario_value_per_evaluation_case - reference.realised_scenario_value_per_evaluation_case,
                "realised_value_per_approved_case_change": candidate.realised_scenario_value_per_approved_case - reference.realised_scenario_value_per_approved_case,
                "normalized_break_even_ad_cost_per_scored_case": realised_difference / n if within else np.nan,
                "expected_scenario_value_T": reference.expected_scenario_value_total if within else np.nan,
                "expected_scenario_value_T_plus_AD": candidate.expected_scenario_value_total if within else np.nan,
                "incremental_expected_value_from_AD": candidate.expected_scenario_value_total - reference.expected_scenario_value_total if within else np.nan,
                "realised_scenario_value_T": reference.realised_scenario_value_total if within else np.nan,
                "realised_scenario_value_T_plus_AD": candidate.realised_scenario_value_total if within else np.nan,
                "incremental_value_from_AD": realised_difference if within else np.nan,
            })
    incremental = pd.DataFrame(incremental_rows)
    within_family = (
        incremental.loc[incremental.comparison_type.eq("within_family_ad")]
        .sort_values(["algorithm_family", "loss_to_gain_ratio"], kind="stable")
        .reset_index(drop=True)
    )
    complexity = (
        incremental.loc[incremental.comparison_type.eq("t_plus_ad_complexity")]
        .sort_values(["comparison_id", "loss_to_gain_ratio"], kind="stable")
        .reset_index(drop=True)
    )
    return results, within_family, complexity


def frame_key_unique(frame: pd.DataFrame, columns: list[str]) -> bool:
    return not frame.duplicated(columns).any()


def validate_outputs(frames: dict[str, pd.DataFrame], data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, component: str, observed: Any, expected: Any, passed: bool, notes: str = "") -> None:
        rows.append({
            "validation_id": check_id, "component": component,
            "observed_result": json.dumps(json_safe(observed), sort_keys=True),
            "expected_result": json.dumps(json_safe(expected), sort_keys=True),
            "status": "PASS" if passed else "FAIL", "notes": notes,
        })

    predictions = frames["final_evaluation_predictions.parquet"]
    add("final_population", "population", len(predictions), EXPECTED_FINAL, len(predictions) == EXPECTED_FINAL)
    add("final_unique_keys", "population", int(predictions.case_id.nunique()), EXPECTED_FINAL, predictions.case_id.is_unique and predictions.case_id.notna().all())
    add("final_base_order", "population", bool(predictions.base_order.is_monotonic_increasing and predictions.base_order.is_unique), True, predictions.base_order.is_monotonic_increasing and predictions.base_order.is_unique)
    add("final_binary_target", "population", sorted(predictions.target.unique().tolist()), [0, 1], set(predictions.target.unique()) == {0, 1})
    probability_columns = [f"calibrated_pd__{model}" for model in MODEL_ORDER]
    probability_ok = bool(np.isfinite(predictions[probability_columns].to_numpy()).all() and predictions[probability_columns].ge(0).all().all() and predictions[probability_columns].le(1).all().all())
    add("prediction_integrity", "predictions", probability_ok, True, probability_ok)
    add("prediction_alignment", "predictions", {column: int(predictions[column].notna().sum()) for column in probability_columns}, EXPECTED_FINAL, all(predictions[column].notna().sum() == EXPECTED_FINAL for column in probability_columns))
    expected_rows = {
        "overall_model_metrics.csv": 6, "overall_incremental_gains.csv": 3,
        "subgroup_model_metrics.csv": 90, "subgroup_incremental_gains.csv": 45, "subgroup_integrity.csv": 15,
        "pd_reassessment_summary.csv": 96, "risk_quintile_migration.csv": 1200,
        "approval_switch_summary.csv": 960, "fixed_approval_rate_curve.csv": 966,
        "fixed_approval_rate_points.csv": 30, "fixed_risk_budget_curve.csv": 756,
        "fixed_risk_budget_points.csv": 36, "economic_scenario_results.csv": 36,
        "economic_incremental_value.csv": 18, "economic_complexity_comparisons.csv": 18,
    }
    for name, expected in expected_rows.items():
        add(f"row_count::{name}", "outputs", len(frames[name]), expected, len(frames[name]) == expected)
    schema = json.loads((data_root / "audits/task11/part3/final_evaluation_output_schema.json").read_text(encoding="utf-8"))
    schema_by_name = {item["file_name"]: item for item in schema["future_output_files"]}
    for name in part3.FUTURE_OUTPUT_NAMES:
        missing = sorted(set(schema_by_name[name]["required_columns"]) - set(frames[name].columns))
        add(f"part3_schema::{name}", "schema", missing, [], not missing)
    keys = {
        "subgroup_model_metrics.csv": ["group_dimension", "group_label", "model_id"],
        "subgroup_incremental_gains.csv": ["group_dimension", "group_label", "algorithm_family"],
        "fixed_approval_rate_curve.csv": ["model_id", "approval_rate_percent"],
        "fixed_approval_rate_points.csv": ["model_id", "approval_rate_percent"],
        "fixed_risk_budget_curve.csv": ["model_id", "common_risk_budget_percent"],
        "fixed_risk_budget_points.csv": ["model_id", "common_risk_budget_percent"],
        "economic_scenario_results.csv": ["model_id", "scenario_id"],
        "economic_incremental_value.csv": ["algorithm_family", "scenario_id"],
        "economic_complexity_comparisons.csv": ["comparison_id", "scenario_id"],
    }
    for name, columns in keys.items():
        add(f"unique_key::{name}", "outputs", frame_key_unique(frames[name], columns), True, frame_key_unique(frames[name], columns))
    integrity = frames["subgroup_integrity.csv"]
    add("subgroup_reconciliation", "subgroups", integrity.reconciliation_status.unique().tolist(), ["PASS"], integrity.reconciliation_status.eq("PASS").all())
    approval = frames["fixed_approval_rate_curve.csv"]
    approval_counts = approval.groupby("model_id").size().to_dict()
    add("approval_grid_shape", "policy", approval_counts, {model: 161 for model in MODEL_ORDER}, approval.groupby("model_id").size().eq(161).all())
    approval_points = frames["fixed_approval_rate_points.csv"]
    approval_reconcile = approval_points.merge(approval, on=["model_id", "approval_rate_percent"], suffixes=("_point", "_curve"))
    add("approval_anchor_reconciliation", "policy", len(approval_reconcile), 30, len(approval_reconcile) == 30 and all(np.allclose(approval_reconcile[f"{column}_point"], approval_reconcile[f"{column}_curve"], equal_nan=True) for column in ("approved_count", "realised_approved_defaults", "approved_portfolio_mean_calibrated_pd")))
    switches = frames["approval_switch_summary.csv"]
    overall_switch = switches[switches.stratum_dimension.eq("overall")]
    pivot = overall_switch.pivot_table(index=["algorithm_family", "approval_rate_percent"], columns="switch_category", values="case_count", aggfunc="sum")
    add("approval_switch_equality", "policy", bool((pivot.T_only_approved == pivot.T_plus_AD_only_approved).all()), True, bool((pivot.T_only_approved == pivot.T_plus_AD_only_approved).all()))
    risk = frames["fixed_risk_budget_curve.csv"]
    monotonic = all(group.sort_values("common_risk_budget_percent").approved_count.is_monotonic_increasing for _, group in risk.groupby("model_id"))
    feasible = risk[risk.feasibility_flag.astype(bool)]
    budget_ok = bool((feasible.approved_portfolio_mean_calibrated_pd <= feasible.common_risk_budget_probability + 1e-12).all())
    add("risk_grid_shape", "policy", risk.groupby("model_id").size().to_dict(), {model: 126 for model in MODEL_ORDER}, risk.groupby("model_id").size().eq(126).all())
    add("risk_monotonic", "policy", monotonic, True, monotonic)
    add("risk_feasible", "policy", budget_ok, True, budget_ok)
    economic = frames["economic_scenario_results.csv"]
    add("economic_shape", "economics", {"rows": len(economic), "models": economic.model_id.nunique(), "scenarios": economic.scenario_id.nunique()}, {"rows": 36, "models": 6, "scenarios": 6}, len(economic) == 36 and economic.model_id.nunique() == 6 and economic.scenario_id.nunique() == 6)
    strict_check = True
    for row in economic.itertuples(index=False):
        p = predictions[f"calibrated_pd__{row.model_id}"]
        strict_check &= int((p < row.individual_pd_threshold_probability).sum()) == int(row.approved_count)
    add("economic_strict_threshold", "economics", strict_check, True, strict_check)
    forbidden_suffixes = {".png", ".pdf", ".html", ".ipynb", ".pptx"}
    add("no_visual_outputs", "boundary", any(Path(name).suffix.lower() in forbidden_suffixes for name in OUTPUT_NAMES), False, not any(Path(name).suffix.lower() in forbidden_suffixes for name in OUTPUT_NAMES))
    add("no_fit_paths", "boundary", int(frames["prediction_integrity_internal"].fit_calls.sum()), 0, int(frames["prediction_integrity_internal"].fit_calls.sum()) == 0)
    validation = pd.DataFrame(rows)
    if validation.status.eq("FAIL").any():
        raise Task12Blocked(f"Output validation failed: {validation.loc[validation.status.eq('FAIL')].to_dict('records')}")
    return validation


def final_readme(readme: str, target_summary: dict[str, Any], overall: pd.DataFrame, output_dir: Path) -> str:
    if readme.count(README_START) != 1 or readme.count(README_END) != 1:
        raise Task12Blocked("Task 12 README markers are missing or duplicated")
    rows = []
    for row in overall.itertuples(index=False):
        rows.append(f"| `{row.model_id}` | {row.roc_auc:.6f} | {row.average_precision:.6f} | {row.log_loss:.6f} | {row.brier_score:.6f} | {row.mean_calibrated_pd:.6f} |")
    relative = os.path.relpath(output_dir, Path.cwd())
    block = f"""{README_START}
## Formal final evaluation

Task 12 formal final evaluation is complete on {target_summary['rows']:,} frozen
applications: {target_summary['positives']:,} defaults and {target_summary['negatives']:,}
non-defaults, for an observed default rate of {target_summary['observed_default_rate']:.6%}.

| Model | ROC-AUC | Average precision | Log loss | Brier score | Mean calibrated PD |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The prespecified primary comparison is `lightgbm_T` versus
`lightgbm_T_plus_AD`; all six frozen model results are retained. Separate
account-count, bureau-history, and AD-richness results are available alongside
the fixed-approval and fixed-portfolio-risk curves.

Economic results use normalized common G/L scenarios and report both
model-predicted expected value and target-based realised scenario value. They
are not actual accounting profit or currency. Within-family increments and
directed T+AD complexity comparisons are stored in separate 18-row files. See
[`overall_model_metrics.csv`]({relative}/overall_model_metrics.csv) and the
[`Task 12 report`]({relative}/task12_final_evaluation_report.md).
{README_END}"""
    start = readme.index(README_START)
    end = readme.index(README_END, start) + len(README_END)
    updated = readme[:start] + block + readme[end:]
    updated = updated.replace(
        "**Status: Final-evaluation ready.** Feature construction, the frozen random\n"
        "split, six-model training, probability calibration, and the Task 11 analysis\n"
        "contract are complete. Formal final evaluation has not yet been executed.",
        "**Status: Task 12 formal final evaluation complete.** Feature construction, the frozen random\n"
        "split, six-model training, calibration, contract freeze, and held-out evaluation are complete.",
    )
    updated = updated.replace("- [ ] Execute the formal 228,999-case final evaluation.", "- [x] Execute the formal 228,999-case final evaluation.")
    if "/Users/" in updated or "Task 12 has not yet been executed" in updated:
        raise Task12Blocked("Final README still contains a private path or stale Task 12 status")
    return updated


def git_state(repo_root: Path) -> tuple[str, list[str]]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, text=True, capture_output=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=repo_root, check=True, text=True, capture_output=True).stdout.splitlines()
    return commit, status


def markdown_table(frame: pd.DataFrame, float_digits: int = 9) -> str:
    headers = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def build_report(
    target_summary: dict[str, Any], subgroup_integrity: pd.DataFrame, prediction_audit: pd.DataFrame,
    overall: pd.DataFrame, gains: pd.DataFrame, frames: dict[str, pd.DataFrame], protected_count: int,
    validations: pd.DataFrame, focused_result: str, full_result: str, command: list[str], output_dir: Path,
) -> str:
    metric_table = markdown_table(overall[["model_id", "selected_calibration_method", "n_cases", "n_defaults", "observed_default_rate", "mean_calibrated_pd", "roc_auc", "average_precision", "log_loss", "brier_score"]], 9)
    gain_table = markdown_table(gains[["algorithm_family", "roc_auc_gain", "average_precision_gain", "log_loss_gain", "brier_gain", "mean_calibrated_pd_change"]], 9)
    subgroup_table = markdown_table(subgroup_integrity[["group_dimension", "group_label", "group_case_count", "share_of_full_final_population", "target_positive_count", "observed_default_rate", "excluded_from_dimension_count"]], 9)
    prediction_table = markdown_table(prediction_audit[["model_id", "selected_calibration_method", "selected_probability_minimum", "selected_probability_maximum", "selected_probability_mean"]], 12)
    row_counts = "\n".join(f"- `{name}`: {len(frame):,} rows" for name, frame in sorted(frames.items()) if name in OUTPUT_NAMES and name not in AUDIT_OUTPUT_NAMES)
    return f"""# Task 12 — formal final evaluation

Status: **COMPLETE**

## Execution boundary and foundation

The comprehensive Phase A README correction was verified before any final-evaluation feature or target rows were loaded. Tasks 8–11 passed their saved read-only verifiers. All {protected_count} protected non-README files retained identical SHA-256 hashes before and after Task 12. No model, preprocessor, feature selector, or calibrator was fitted or modified.

## Final population

The frozen final partition contains {target_summary['rows']:,} unique applications, {target_summary['positives']:,} defaults, and {target_summary['negatives']:,} non-defaults. The observed default rate is {target_summary['observed_default_rate']:.9%}. Membership order, split separation, feature alignment, and target alignment passed.

## Selected prediction integrity

{prediction_table}

Each selected calibrator was applied exactly once. Every probability is finite and within [0, 1].

## Overall metrics

{metric_table}

## Within-family T to T+AD changes

Positive performance gains mean T+AD is better. Mean PD change is reassessment, not a performance gain.

{gain_table}

## Subgroup integrity

{subgroup_table}

Account, history, and AD-richness dimensions were assigned independently from pre-model values. No cross-product, binary thin-file threshold, missing-history group, or composite index was created.

## Saved result dimensions

{row_counts}

The approval curve contains 161 points per model and five anchors per model. The risk-budget curve contains 126 points per model and six anchors per model. Economic results contain six models by six normalized G/L scenarios. Reassessment, rank-percentile changes, quintile migration, and approval switches are saved without winner/loser labels.

## Validation and tests

All {len(validations)} Task 12 validations passed. Focused tests supplied to the run: `{focused_result}`. Full safe suite supplied to the run: `{full_result}`.

Formal command:

```text
{' '.join(command)}
```

Output directory: `{output_dir}`.

## Explicit non-actions

No retraining, refitting, tuning, recalibration, model selection, subgroup selection, threshold optimization, time-split supplement, contract-sensitive loan/accounting analysis, visualization, dashboard, notebook, Task 13 work, commit, push, deployment, or remote operation was performed. Normalized scenario values are not actual Home Credit profit or currency.
"""


def output_path_is_safe(data_root: Path, output_dir: Path) -> bool:
    protected = [data_root / relative for relative in (
        "audits/task08", "audits/task08_followup", "audits/task09", "audits/task10", "audits/task11",
        "interim/task08", "interim/task08_followup", "interim/task09", "interim/task10", "models/task09", "models/task10",
    )]
    resolved = output_dir.resolve()
    return not any(resolved == path.resolve() or path.resolve() in resolved.parents for path in protected)


def run(data_root: Path, output_dir: Path, repo_root: Path, threads: int, focused_result: str, full_result: str) -> dict[str, Any]:
    data_root, output_dir, repo_root = data_root.resolve(), output_dir.resolve(), repo_root.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Task 12 output: {output_dir}")
    if not output_path_is_safe(data_root, output_dir):
        raise Task12Blocked(f"Task 12 output path overlaps a protected foundation: {output_dir}")
    readme_path = repo_root / "README.md"
    phase_a_readme = readme_path.read_text(encoding="utf-8")
    phase_a_rows = verify_phase_a_readme(phase_a_readme)
    phase_a_verified_at = now_utc()
    records = protected_paths(data_root, repo_root)
    before = hash_paths(records)
    baseline_digest = protected_registry_digest(before)
    if baseline_digest != PRE_PHASE_A_PROTECTED_REGISTRY_DIGEST:
        raise Task12Blocked(f"Protected foundation differs from the pre-Phase-A baseline: {baseline_digest}")
    foundation_rows, foundation_context = verify_foundation(data_root, repo_root)
    foundation_frame = pd.DataFrame([*phase_a_rows, *foundation_rows])
    final_data_load_started_at = now_utc()
    final, final_positions, manifest_keys, membership_summary = load_final_membership(data_root)
    groups, subgroup_build = build_subgroups(data_root, final, final_positions)
    prediction_only, prediction_audit = infer_selected_predictions(data_root, final, final_positions, manifest_keys)
    prediction_only = prediction_only.merge(groups, on=["case_id", "base_order"], how="left", validate="one_to_one", sort=False)
    prediction_with_target, target_summary = load_and_join_final_target(data_root, prediction_only[["case_id", "base_order", *[f"calibrated_pd__{model}" for model in MODEL_ORDER]]])
    predictions = prediction_with_target[["case_id", "base_order", "target"]].copy()
    predictions["account_group"] = groups.account_group.to_numpy()
    predictions["bureau_history_group"] = groups.bureau_history_group.to_numpy()
    predictions["ad_richness_group"] = groups.ad_richness_group.to_numpy()
    for model in MODEL_ORDER:
        predictions[f"calibrated_pd__{model}"] = prediction_with_target[f"calibrated_pd__{model}"].to_numpy(np.float64)
    overall = overall_metrics(predictions)
    overall_gain = overall_gains(overall)
    subgroup_metrics, subgroup_gains, subgroup_integrity = subgroup_outputs(predictions)
    reassessment, migration = reassessment_and_migration(predictions)
    approval_grid = pd.read_csv(data_root / "audits/task11/part3/approval_rate_grid.csv", dtype={"approval_rate_percent": str, "approval_rate_probability": str})
    risk_grid = pd.read_csv(data_root / "audits/task11/part3/portfolio_risk_budget_grid.csv", dtype={"risk_budget_percent": str, "risk_budget_probability": str})
    scenarios = pd.read_csv(data_root / "audits/task11/part3/economic_scenario_contract.csv", dtype={"individual_pd_threshold_probability": str, "individual_pd_threshold_percent": str})
    approval, approval_cache = approval_curve(predictions, approval_grid)
    approval_points = approval[approval.is_external_display_point].reset_index(drop=True)
    switches = approval_switches(predictions, approval_cache, part3.APPROVAL_ANCHORS)
    risk = risk_budget_curve(predictions, risk_grid)
    risk_points = risk[risk.is_external_display_point].reset_index(drop=True)
    economic, economic_incremental, economic_complexity = economic_outputs(predictions, scenarios)
    frames: dict[str, pd.DataFrame] = {
        "final_evaluation_predictions.parquet": predictions,
        "subgroup_model_metrics.csv": subgroup_metrics,
        "subgroup_incremental_gains.csv": subgroup_gains,
        "fixed_approval_rate_points.csv": approval_points,
        "fixed_approval_rate_curve.csv": approval,
        "fixed_risk_budget_points.csv": risk_points,
        "fixed_risk_budget_curve.csv": risk,
        "economic_scenario_results.csv": economic,
        "economic_incremental_value.csv": economic_incremental,
        "economic_complexity_comparisons.csv": economic_complexity,
        "overall_model_metrics.csv": overall,
        "overall_incremental_gains.csv": overall_gain,
        "pd_reassessment_summary.csv": reassessment,
        "risk_quintile_migration.csv": migration,
        "approval_switch_summary.csv": switches,
        "subgroup_integrity.csv": subgroup_integrity,
        "prediction_integrity_internal": prediction_audit,
    }
    validation_frame = validate_outputs(frames, data_root)
    readme_final = final_readme(phase_a_readme, target_summary, overall, output_dir)
    readme_changed = False
    temp: Path | None = None
    try:
        atomic_text(readme_path, readme_final)
        readme_changed = True
        after = hash_paths(records)
        hash_frame = pd.DataFrame([
            {
                "protected_path": path, "artifact": item["artifact"], "component_task": item["component_task"],
                "sha256_before": item["sha256"], "sha256_after": after[path]["sha256"],
                "unchanged_flag": item["sha256"] == after[path]["sha256"],
            }
            for path, item in before.items()
        ])
        if not hash_frame.unchanged_flag.all():
            raise Task12Blocked(f"Protected hashes changed: {hash_frame.loc[~hash_frame.unchanged_flag, 'protected_path'].tolist()}")
        validation_frame = pd.concat([validation_frame, pd.DataFrame([
            {
                "validation_id": "protected_hashes", "component": "foundation",
                "observed_result": json.dumps(int(hash_frame.unchanged_flag.sum())),
                "expected_result": json.dumps(len(hash_frame)), "status": "PASS", "notes": "Independent SHA-256 before/after equality.",
            },
            {
                "validation_id": "phase_a_before_final_load", "component": "README",
                "observed_result": json.dumps({"phase_a_verified_at": phase_a_verified_at, "final_data_load_started_at": final_data_load_started_at}),
                "expected_result": json.dumps("Phase A verified before final data load"),
                "status": "PASS" if phase_a_verified_at <= final_data_load_started_at else "FAIL", "notes": "Timestamps are UTC ISO-8601.",
            },
            {
                "validation_id": "final_readme", "component": "README", "observed_result": json.dumps("Task 12 complete"),
                "expected_result": json.dumps("Task 12 complete with repository-relative links"),
                "status": "PASS" if "Task 12 formal final evaluation is complete" in readme_final and "/Users/" not in readme_final else "FAIL", "notes": "Phase K bounded section and main status updated.",
            },
        ])], ignore_index=True)
        if validation_frame.status.eq("FAIL").any():
            raise Task12Blocked("Post-documentation validation failed")
        git_commit, current_status = git_state(repo_root)
        own_new = {"?? scripts/run_task12_final_evaluation.py", "?? tests/test_task12_final_evaluation.py"}
        pre_existing_status = [line for line in current_status if line not in own_new]
        formal_command = [
            sys.executable, "-u", "scripts/run_task12_final_evaluation.py", "--data-root", str(data_root),
            "--output-dir", str(output_dir), "--threads", str(threads),
        ]
        verify_command = [*formal_command, "--verify-completed"]
        config = {
            "task_name": TASK_NAME, "version": VERSION, "timestamp_utc": now_utc(),
            "repository_root": str(repo_root), "data_root": str(data_root), "output_directory": str(output_dir),
            "python_executable": sys.executable,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__, "sklearn": sklearn.__version__, "lightgbm": lgb.__version__, "torch": torch.__version__, "threads": threads},
            "git_commit": git_commit, "pre_existing_user_worktree_status": pre_existing_status,
            "phase_a": {"readme_sha256": hashlib.sha256(phase_a_readme.encode()).hexdigest(), "verified_at_utc": phase_a_verified_at, "final_data_load_started_at_utc": final_data_load_started_at, "completed_before_final_data_load": phase_a_verified_at <= final_data_load_started_at},
            "protected_foundation": {"file_count": len(hash_frame), "pre_phase_a_registry_digest": PRE_PHASE_A_PROTECTED_REGISTRY_DIGEST, "pre_phase_a_console_digest_with_literal_backslash_n_delimiter": PRE_PHASE_A_LEGACY_REGISTRY_DIGEST, "formal_run_before_digest": baseline_digest, "after_digest": protected_registry_digest(after)},
            "membership": membership_summary, "target": target_summary, "subgroup_build": subgroup_build,
            "model_order": MODEL_ORDER, "selected_calibration_methods": SELECTED_METHODS,
            "prediction_integrity": prediction_audit.to_dict("records"),
            "commands": {"formal": formal_command, "independent_read_only_verifier": verify_command},
            "test_results": {"focused_before_formal": focused_result, "full_safe_before_formal": full_result},
            "quantile_convention": QUANTILE_CONVENTION,
            "rank_percentile_formula": "zero_based_stable_rank / (N - 1) for N > 1; 0 for N = 1",
            "risk_quintile_formula": "floor(zero_based_stable_rank * 5 / N) + 1, capped at 5",
            "no_fit_guard": {"active_during_all_model_and_calibrator_inference": True, "fit_calls": 0, "calibrator_application_count_each_model": 1},
            "boundary_flags": {"model_retraining": False, "preprocessor_refitting": False, "calibrator_refitting": False, "final_target_used_for_prediction_or_selection": False, "visualization_created": False, "task13_work": False, "contract_sensitive_economics": False},
            "output_inventory": sorted(OUTPUT_NAMES),
        }
        report = build_report(target_summary, subgroup_integrity, prediction_audit, overall, overall_gain, frames, len(hash_frame), validation_frame, focused_result, full_result, formal_command, output_dir)
        temp = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
        temp.mkdir(parents=True, exist_ok=False)
        for name, frame in frames.items():
            if name == "prediction_integrity_internal":
                continue
            if name.endswith(".parquet"):
                atomic_parquet(temp / name, frame)
            else:
                atomic_csv(temp / name, frame)
        atomic_csv(temp / "foundation_verification.csv", foundation_frame)
        atomic_csv(temp / "protected_hashes_before_after.csv", hash_frame)
        atomic_csv(temp / "validation_results.csv", validation_frame)
        atomic_json(temp / "run_config.json", config)
        atomic_text(temp / "task12_final_evaluation_report.md", report)
        produced = {path.name for path in temp.iterdir() if path.is_file()}
        if produced != OUTPUT_NAMES:
            raise Task12Blocked(f"Task 12 output inventory mismatch: {sorted(produced)}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, output_dir)
        temp = None
        return {
            "status": "COMPLETE", "output_directory": str(output_dir), "protected_files": len(hash_frame),
            "final_rows": target_summary["rows"], "final_defaults": target_summary["positives"],
            "final_default_rate": target_summary["observed_default_rate"], "validations": len(validation_frame),
            "outputs": sorted(produced), "overall_metrics": overall.to_dict("records"),
            "overall_incremental_gains": overall_gain.to_dict("records"),
        }
    except Exception:
        if readme_changed and readme_path.read_text(encoding="utf-8") == readme_final:
            atomic_text(readme_path, phase_a_readme)
        if temp is not None and temp.is_dir():
            shutil.rmtree(temp)
        raise


def verify_completed(data_root: Path, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    data_root, output_dir, repo_root = data_root.resolve(), output_dir.resolve(), repo_root.resolve()
    if not output_dir.is_dir():
        raise Task12Blocked(f"Task 12 output directory missing: {output_dir}")
    inventory = {path.name for path in output_dir.iterdir() if path.is_file()}
    if inventory != OUTPUT_NAMES:
        raise Task12Blocked(f"Task 12 completed inventory mismatch: {sorted(inventory)}")
    hashes = pd.read_csv(output_dir / "protected_hashes_before_after.csv")
    current_records = protected_paths(data_root, repo_root)
    current = hash_paths(current_records)
    if set(hashes.protected_path) != set(current):
        raise Task12Blocked("Protected file inventory changed after Task 12")
    changed = [row.protected_path for row in hashes.itertuples() if current[row.protected_path]["sha256"] != row.sha256_before]
    if changed:
        raise Task12Blocked(f"Protected file hash changed after Task 12: {changed}")
    predictions = pd.read_parquet(output_dir / "final_evaluation_predictions.parquet")
    overall_saved = pd.read_csv(output_dir / "overall_model_metrics.csv")
    overall_recomputed = overall_metrics(predictions)
    metric_columns = ["observed_default_rate", "mean_calibrated_pd", "roc_auc", "average_precision", "log_loss", "brier_score"]
    metric_max_difference = max(float(np.max(np.abs(overall_saved[column] - overall_recomputed[column]))) for column in metric_columns)
    if metric_max_difference > 1e-12:
        raise Task12Blocked(f"Saved overall metrics do not recompute exactly: {metric_max_difference}")
    approval = pd.read_csv(output_dir / "fixed_approval_rate_curve.csv")
    approval_points = pd.read_csv(output_dir / "fixed_approval_rate_points.csv")
    risk = pd.read_csv(output_dir / "fixed_risk_budget_curve.csv")
    economic = pd.read_csv(output_dir / "economic_scenario_results.csv")
    validation = pd.read_csv(output_dir / "validation_results.csv")
    if not validation.status.eq("PASS").all():
        raise Task12Blocked("Saved Task 12 validation table is not all PASS")
    economic_incremental = pd.read_csv(output_dir / "economic_incremental_value.csv")
    economic_complexity = pd.read_csv(output_dir / "economic_complexity_comparisons.csv")
    if (
        len(predictions) != EXPECTED_FINAL
        or len(approval) != 966
        or len(approval_points) != 30
        or len(risk) != 756
        or len(economic) != 36
        or len(economic_incremental) != 18
        or len(economic_complexity) != 18
        or economic_incremental.duplicated(["algorithm_family", "scenario_id"]).any()
        or economic_complexity.duplicated(["comparison_id", "scenario_id"]).any()
    ):
        raise Task12Blocked("Saved Task 12 row-count identity failed")
    economic_value_max_difference = 0.0
    for row in economic.itertuples(index=False):
        p = predictions[f"calibrated_pd__{row.model_id}"].to_numpy(np.float64)
        y = predictions.target.to_numpy(np.int8)
        approved = p < row.individual_pd_threshold_probability
        predicted_total = float(((1 - p[approved]) - p[approved] * row.loss_to_gain_ratio).sum())
        realised_total = float((y[approved] == 0).sum() - (y[approved] == 1).sum() * row.loss_to_gain_ratio)
        economic_value_max_difference = max(economic_value_max_difference, abs(predicted_total - row.expected_scenario_value_total), abs(realised_total - row.realised_scenario_value_total))
    if economic_value_max_difference > 1e-8:
        raise Task12Blocked(f"Saved economic values do not recompute: {economic_value_max_difference}")
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    if "Task 12 formal final evaluation is complete" not in readme or "/Users/" in readme:
        raise Task12Blocked("Final README status/link boundary failed")
    return {
        "status": "PASS", "mode": "INDEPENDENT_READ_ONLY_TASK12_VERIFICATION", "writes_performed": False,
        "protected_files_rehashed": len(current), "protected_hashes_match": True,
        "output_files_verified": len(inventory), "final_rows": len(predictions),
        "final_defaults": int(predictions.target.sum()), "overall_metric_max_abs_difference": metric_max_difference,
        "economic_value_max_abs_difference": economic_value_max_difference,
        "approval_curve_rows": len(approval), "risk_curve_rows": len(risk), "economic_rows": len(economic),
        "economic_incremental_rows": len(economic_incremental),
        "economic_complexity_rows": len(economic_complexity),
        "validation_rows": len(validation), "validation_all_pass": True,
    }


def _validate_economic_split(
    within: pd.DataFrame,
    complexity: pd.DataFrame,
    original: pd.DataFrame | None = None,
) -> None:
    expected_scenarios = {f"normalized_gl_{ratio}" for ratio in (2, 5, 10, 20, 30, 50)}
    expected_families = {"logit", "lightgbm", "mlp"}
    expected_comparisons = {"lightgbm_minus_logit", "mlp_minus_logit", "mlp_minus_lightgbm"}
    checks = [
        len(within) == 18,
        len(complexity) == 18,
        set(within.comparison_type) == {"within_family_ad"},
        set(complexity.comparison_type) == {"t_plus_ad_complexity"},
        set(within.algorithm_family) == expected_families,
        set(within.scenario_id) == expected_scenarios,
        set(complexity.comparison_id) == expected_comparisons,
        set(complexity.scenario_id) == expected_scenarios,
        not within.duplicated(["algorithm_family", "scenario_id"]).any(),
        not complexity.duplicated(["comparison_id", "scenario_id"]).any(),
    ]
    if not all(checks):
        raise Task12Blocked("Economic split failed its 18+18 type, identifier, scenario, or unique-key contract")
    if list(within.columns) != list(complexity.columns):
        raise Task12Blocked("Economic split files do not retain the same source columns")
    if original is not None:
        combined = pd.concat([within, complexity], ignore_index=True)
        sort_key = ["comparison_type", "comparison_id", "scenario_id"]
        try:
            pd.testing.assert_frame_equal(
                original.sort_values(sort_key).reset_index(drop=True),
                combined.sort_values(sort_key).reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=0,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise Task12Blocked(f"The repaired economic rows do not reconcile to the original 36 rows: {exc}") from exc


def repair_economic_output(output_dir: Path) -> dict[str, Any]:
    """Split the confirmed mixed economic file without recalculation or inference."""
    output_dir = output_dir.resolve()
    current_inventory = {path.name for path in output_dir.iterdir() if path.is_file()}
    pre_repair_inventory = OUTPUT_NAMES - {"economic_complexity_comparisons.csv"}
    if current_inventory not in (pre_repair_inventory, OUTPUT_NAMES):
        raise Task12Blocked(f"Unexpected Task 12 inventory before economic repair: {sorted(current_inventory)}")

    mixed_path = output_dir / "economic_incremental_value.csv"
    complexity_path = output_dir / "economic_complexity_comparisons.csv"
    source = pd.read_csv(mixed_path)
    if len(source) == 18 and complexity_path.is_file():
        complexity = pd.read_csv(complexity_path)
        _validate_economic_split(source, complexity)
        return {
            "status": "ALREADY_REPAIRED",
            "within_family_rows": len(source),
            "complexity_rows": len(complexity),
            "writes_performed": False,
        }
    if len(source) != 36 or set(source.comparison_type) != {"within_family_ad", "t_plus_ad_complexity"}:
        raise Task12Blocked("Expected the confirmed 36-row mixed economic file before repair")

    original = source.copy(deep=True)
    within = (
        source.loc[source.comparison_type.eq("within_family_ad")]
        .sort_values(["algorithm_family", "loss_to_gain_ratio"], kind="stable")
        .reset_index(drop=True)
    )
    complexity = (
        source.loc[source.comparison_type.eq("t_plus_ad_complexity")]
        .sort_values(["comparison_id", "loss_to_gain_ratio"], kind="stable")
        .reset_index(drop=True)
    )
    _validate_economic_split(within, complexity, original)

    validation_path = output_dir / "validation_results.csv"
    validation = pd.read_csv(validation_path, keep_default_na=False)
    validation.loc[
        validation.validation_id.eq("row_count::economic_incremental_value.csv"),
        ["observed_result", "expected_result", "status"],
    ] = ["18", "18", "PASS"]
    additions = pd.DataFrame([
        {
            "validation_id": "row_count::economic_complexity_comparisons.csv",
            "component": "outputs", "observed_result": "18", "expected_result": "18",
            "status": "PASS", "notes": "Task 13 structural repair; values copied from the original mixed file.",
        },
        {
            "validation_id": "unique_key::economic_complexity_comparisons.csv",
            "component": "outputs", "observed_result": "true", "expected_result": "true",
            "status": "PASS", "notes": "Key: comparison_id, scenario_id.",
        },
        {
            "validation_id": "economic_split_reconciliation",
            "component": "economics", "observed_result": "36", "expected_result": "36",
            "status": "PASS", "notes": "All original rows occur exactly once across the repaired 18+18 files; rtol=0, atol=1e-12.",
        },
    ])
    validation.loc[
        validation.validation_id.eq("unique_key::economic_incremental_value.csv"), "notes"
    ] = "Key: algorithm_family, scenario_id."
    validation = pd.concat([validation, additions], ignore_index=True)
    if validation.validation_id.duplicated().any() or not validation.status.eq("PASS").all():
        raise Task12Blocked("Repaired validation metadata is not unique and all-PASS")

    config_path = output_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["output_inventory"] = sorted(OUTPUT_NAMES)
    config["task13_economic_schema_repair"] = {
        "status": "COMPLETE",
        "source_rows": 36,
        "within_family_rows": 18,
        "complexity_rows": 18,
        "numerical_reconciliation": "PASS (rtol=0, atol=1e-12)",
        "model_training_or_inference": False,
    }

    report_path = output_dir / "task12_final_evaluation_report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace("- `economic_incremental_value.csv`: 36 rows", "- `economic_incremental_value.csv`: 18 rows\n- `economic_complexity_comparisons.csv`: 18 rows")
    report = report.replace("All 51 Task 12 validations passed.", "All 54 Task 12 validations passed after the Task 13 schema repair.")
    report = report.replace("reloaded all 20 Task 12 outputs", "reloaded all 20 original Task 12 outputs")
    report = report.replace("- `/Users/haoguannan/Projects/home_credit/data/final_evaluation/task12/economic_incremental_value.csv`", "- `/Users/haoguannan/Projects/home_credit/data/final_evaluation/task12/economic_incremental_value.csv`\n- `/Users/haoguannan/Projects/home_credit/data/final_evaluation/task12/economic_complexity_comparisons.csv`")
    repair_note = """

## Task 13 economic-output schema repair

The original 36-row mixed incremental file was mechanically split without recalculation. `economic_incremental_value.csv` now contains the 18 within-family T versus T+AD rows keyed by `algorithm_family, scenario_id`; `economic_complexity_comparisons.csv` contains the 18 directed T+AD cross-algorithm rows keyed by `comparison_id, scenario_id`. All 36 source rows reconcile exactly within `rtol=0, atol=1e-12`. No model training, calibration, or inference occurred.
"""
    report += repair_note

    staged: list[tuple[Path, Path]] = []
    try:
        for path, writer in (
            (mixed_path, lambda p: within.to_csv(p, index=False, lineterminator="\n")),
            (complexity_path, lambda p: complexity.to_csv(p, index=False, lineterminator="\n")),
            (validation_path, lambda p: validation.to_csv(p, index=False, lineterminator="\n")),
            (config_path, lambda p: p.write_text(json.dumps(json_safe(config), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")),
            (report_path, lambda p: p.write_text(report, encoding="utf-8")),
        ):
            temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.repair")
            writer(temp)
            staged.append((temp, path))
        staged_within = pd.read_csv(staged[0][0])
        staged_complexity = pd.read_csv(staged[1][0])
        _validate_economic_split(staged_within, staged_complexity, original)
        for temp, path in staged:
            os.replace(temp, path)
    finally:
        for temp, _path in staged:
            if temp.exists():
                temp.unlink()

    return {
        "status": "REPAIRED",
        "source_rows": len(original),
        "within_family_rows": len(within),
        "complexity_rows": len(complexity),
        "validation_rows": len(validation),
        "reconciliation": "PASS (rtol=0, atol=1e-12)",
        "writes_performed": True,
    }


def record_post_run_evidence(
    data_root: Path, output_dir: Path, repo_root: Path, post_focused_result: str,
    post_full_result: str, verifier_result: str,
) -> dict[str, Any]:
    """Record already-completed test/verifier evidence; this is not the verifier."""
    data_root, output_dir, repo_root = data_root.resolve(), output_dir.resolve(), repo_root.resolve()
    inventory = {path.name for path in output_dir.iterdir() if path.is_file()} if output_dir.is_dir() else set()
    if inventory != OUTPUT_NAMES:
        raise Task12Blocked(f"Cannot record post-run evidence for incomplete inventory: {sorted(inventory)}")
    config_path = output_dir / "run_config.json"
    report_path = output_dir / "task12_final_evaluation_report.md"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    focused_command = [sys.executable, "-m", "pytest", "-q", "tests/test_task12_final_evaluation.py"]
    full_command = [sys.executable, "-m", "pytest", "-q"]
    successful_formal_command = [
        sys.executable, "-u", "scripts/run_task12_final_evaluation.py", "--data-root", str(data_root),
        "--output-dir", str(output_dir), "--threads", "4", "--focused-test-result",
        config["test_results"]["focused_before_formal"], "--full-test-result",
        config["test_results"]["full_safe_before_formal"],
    ]
    verifier_command = [
        sys.executable, "-u", "scripts/run_task12_final_evaluation.py", "--data-root", str(data_root),
        "--output-dir", str(output_dir), "--threads", "4", "--verify-completed",
    ]
    config["commands"].update({
        "focused_task12_before_formal": focused_command,
        "full_safe_suite_before_formal": full_command,
        "successful_formal_invocation_with_recorded_test_arguments": successful_formal_command,
        "focused_task12_after_formal": focused_command,
        "full_safe_suite_after_formal": full_command,
        "independent_read_only_verifier_executed": verifier_command,
    })
    config["post_run_evidence"] = {
        "recorded_at_utc": now_utc(),
        "focused_task12_after_formal": post_focused_result,
        "full_safe_suite_after_formal": post_full_result,
        "independent_read_only_verifier": verifier_result,
        "recorder_is_not_the_independent_verifier": True,
    }
    report = report_path.read_text(encoding="utf-8")
    marker = "## Post-run tests, independent verification, and file inventory"
    if marker in report:
        raise Task12Blocked("Post-run evidence is already recorded")
    created_outputs = "\n".join(f"- `{output_dir / name}`" for name in sorted(OUTPUT_NAMES))
    report += f"""

{marker}

Post-output focused Task 12 tests: `{post_focused_result}`.

Post-output full safe repository suite: `{post_full_result}`.

Independent read-only verifier: `{verifier_result}`. It performed no writes, rehashed all 75 protected files, reloaded all 21 Task 12 outputs, and independently recomputed the overall metrics and normalized economic totals.

Exact executed commands:

```text
{' '.join(focused_command)}
{' '.join(full_command)}
{' '.join(successful_formal_command)}
{' '.join(full_command)}
{' '.join(verifier_command)}
```

Created Task 12 files:

{created_outputs}

Repository files created: `{repo_root / 'scripts/run_task12_final_evaluation.py'}` and `{repo_root / 'tests/test_task12_final_evaluation.py'}`. Repository file modified: `{repo_root / 'README.md'}` (comprehensive Phase A narrative/status correction, then the bounded Task 12 final-status/results section and progress status in Phase K).

Two safely rolled-back attempts preceded the successful formal run: one exposed an aggregate-digest delimiter encoding mismatch while all 75 individual hashes matched, and one exposed the unavailable optional `tabulate` report-rendering dependency after calculations. Neither attempt published a Task 12 directory; the Phase A README was restored each time. No package was installed.
"""
    atomic_json(config_path, config)
    atomic_text(report_path, report)
    return {
        "status": "RECORDED", "output_files": len(inventory),
        "post_focused_test_result": post_focused_result, "post_full_test_result": post_full_result,
        "independent_verifier_result": verifier_result,
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    selected_modes = sum((args.verify_completed, args.record_post_run, args.repair_economic_output))
    if selected_modes > 1:
        raise Task12Blocked("Choose only one of --verify-completed, --record-post-run, or --repair-economic-output")
    if args.repair_economic_output:
        result = repair_economic_output(args.output_dir)
    elif args.verify_completed:
        result = verify_completed(args.data_root, args.output_dir, repo_root)
    elif args.record_post_run:
        result = record_post_run_evidence(
            args.data_root, args.output_dir, repo_root, args.post_focused_test_result,
            args.post_full_test_result, args.independent_verifier_result,
        )
    else:
        result = run(args.data_root, args.output_dir, repo_root, args.threads, args.focused_test_result, args.full_test_result)
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
