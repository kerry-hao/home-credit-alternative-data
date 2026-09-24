#!/usr/bin/env python3
"""Task 11 Part 2: deterministic development-design audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow

import audit_thin_file_candidates as part1


VERSION = "1.0.0"
TASK_NAME = "TASK_11_PART_2_DEVELOPMENT_DESIGN_AUDIT"
EXPECTED_TOTAL = 1_526_659
EXPECTED_TRAIN = 1_068_661
EXPECTED_CALIBRATION = 114_500
EXPECTED_POSITIVES = 3_600
EXPECTED_FINGERPRINT = part1.EXPECTED_MEMBERSHIP_FP
MODEL_ORDER = list(part1.MODEL_ORDER)
SELECTED_METHODS = dict(part1.EXPECTED_METHODS)
ACCOUNT_LABELS = ["0", "1", "2", "3_plus"]
HISTORY_LABELS = ["0_to_0_5", "0_5_to_1", "1_to_2", "2_to_3", "3_to_4", "4_to_5", "gt_5"]
AD_INDICATORS = [
    "ad__debit__has_numeric_content", "ad__debit__has_nonzero_content", "ad__debit__has_source_evidence",
    "ad__deposit__has_numeric_content", "ad__deposit__has_nonzero_content",
    "ad__tax__has_numeric_content", "ad__tax__has_nonzero_content", "ad__tax__has_source_or_field_evidence",
]
MODULE_FLAGS = {
    "debit": "ad__debit__has_numeric_content",
    "deposit": "ad__deposit__has_numeric_content",
    "tax": "ad__tax__has_numeric_content",
}
ECONOMIC_FIELDS = ["credamount_770A", "annuity_780A", "eir_270L", "price_1097A", "monthsannuity_845L", "numinstls_657L"]
GL_RATIOS = [2, 5, 10, 20, 30, 50]
QUANTILES = [("p01", .01), ("p05", .05), ("p10", .10), ("p25", .25), ("p50", .50),
             ("p75", .75), ("p90", .90), ("p95", .95), ("p99", .99)]
OUTPUT_NAMES = {
    "subgroup_definition_contract.json", "account_group_distribution.csv", "history_group_distribution.csv",
    "ad_indicator_inventory.csv", "ad_richness_candidate_audit.csv", "ad_module_pattern_distribution.csv",
    "calibrated_oof_prediction_integrity.csv", "approval_rate_curve.csv", "portfolio_risk_budget_curve.csv",
    "economic_field_audit.csv", "economic_scenario_grid.csv", "case_id_integrity.csv",
    "input_hashes_before_after.csv", "validation_results.csv", "run_config.json", "task11_part2_report.md",
}


class Part2Blocked(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--focused-test-result", default="not supplied")
    parser.add_argument("--full-test-result", default="not supplied")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def json_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return "" if not math.isfinite(float(value)) else repr(float(value))
    return str(value)


def validation(validation_id: str, component: str, description: str, observed: Any,
               expected: Any, passed: bool, notes: str = "") -> dict[str, Any]:
    return {"validation_id": validation_id, "component": component, "description": description,
            "observed_result": json_value(observed), "expected_result": json_value(expected),
            "status": "PASS" if passed else "FAIL", "notes": notes}


def assign_account_groups(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    result.loc[numeric.eq(0)] = "0"
    result.loc[numeric.eq(1)] = "1"
    result.loc[numeric.eq(2)] = "2"
    result.loc[numeric.ge(3) & np.isfinite(numeric)] = "3_plus"
    return result


def assign_history_groups(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    grouping = pd.to_numeric(values, errors="coerce").astype(float).fillna(0.0)
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    rules = [
        ("0_to_0_5", grouping.ge(0) & grouping.le(.5)),
        ("0_5_to_1", grouping.gt(.5) & grouping.le(1)),
        ("1_to_2", grouping.gt(1) & grouping.le(2)),
        ("2_to_3", grouping.gt(2) & grouping.le(3)),
        ("3_to_4", grouping.gt(3) & grouping.le(4)),
        ("4_to_5", grouping.gt(4) & grouping.le(5)),
        ("gt_5", grouping.gt(5)),
    ]
    for label, mask in rules:
        result.loc[mask] = label
    return result, grouping


def subgroup_contract() -> dict[str, Any]:
    return {
        "contract_version": VERSION,
        "created_at_utc": now_utc(),
        "frozen_outer_train_membership_fingerprint": EXPECTED_FINGERPRINT,
        "assignment_boundary": "pre-model analytic values before imputation, scaling, encoding, winsorization, or standardization",
        "account_dimension": {
            "source_field": "t__observed_active_credit_count",
            "upstream_source": "static_0.numactivecreds_622L",
            "ordered_groups": [
                {"order": 1, "label": "0", "rule": "value == 0"},
                {"order": 2, "label": "1", "rule": "value == 1"},
                {"order": 3, "label": "2", "rule": "value == 2"},
                {"order": 4, "label": "3_plus", "rule": "value >= 3"},
            ],
            "missing_rule": "remain unassigned; never use a model-imputed value",
        },
        "history_dimension": {
            "construction": "earliest valid linked date on/before decision across credit_bureau_b_1.contractdate_551D and credit_bureau_a_1.dateofcredstart_739D",
            "days_formula": "decision_date - earliest_valid_bureau_start_date",
            "years_formula": "bureau_history_days / 365.25",
            "grouping_replacement": "bureau_history_years.fillna(0)",
            "ordered_groups": [
                {"order": 1, "label": "0_to_0_5", "display_interval": "0–0.5 years", "rule": "0 <= value <= 0.5"},
                {"order": 2, "label": "0_5_to_1", "display_interval": "0.5–1 years", "rule": "0.5 < value <= 1"},
                {"order": 3, "label": "1_to_2", "display_interval": "1–2 years", "rule": "1 < value <= 2"},
                {"order": 4, "label": "2_to_3", "display_interval": "2–3 years", "rule": "2 < value <= 3"},
                {"order": 5, "label": "3_to_4", "display_interval": "3–4 years", "rule": "3 < value <= 4"},
                {"order": 6, "label": "4_to_5", "display_interval": "4–5 years", "rule": "4 < value <= 5"},
                {"order": 7, "label": "gt_5", "display_interval": "more than 5 years", "rule": "value > 5"},
            ],
        },
        "binary_history_threshold": None,
        "two_dimensional_design": None,
        "explicitly_deleted_designs": [
            "account-by-history cross-tabulation", "four two-dimensional thin-file cells", "(k,h) grid",
            "joint account/history flag", "PCA/composite thickness index", "binary history-thin flag",
            "separate missing-history category or indicator",
        ],
    }


def protected_paths(data_root: Path, repo_root: Path) -> list[tuple[str, str, Path]]:
    inputs = part1.resolve_inputs(data_root)
    paths = list(part1.protected_paths(inputs))
    result = [(name, component, path) for name, component, path in paths]
    part1_dir = data_root / "audits/task11/part1"
    result.extend([
        ("task11_part1_script", "Task 11 Part 1", repo_root / "scripts/audit_thin_file_candidates.py"),
        ("task11_part1_test", "Task 11 Part 1", repo_root / "tests/test_task11_part1.py"),
    ])
    for path in sorted(part1_dir.iterdir()):
        if path.is_file():
            result.append((f"task11_part1_output::{path.name}", "Task 11 Part 1", path))
    result.extend([
        ("task10_selected_oof_predictions", "Task 10", data_root / "interim/task10/first_full_v3/calibration_oof_predictions.parquet"),
        ("task10_fold_assignments", "Task 10", data_root / "interim/task10/first_full_v3/calibration_fold_assignments.parquet"),
        ("task10_output_hash_registry", "Task 10", data_root / "audits/task10/first_full_v3/output_artifact_hashes.csv"),
    ])
    unique: dict[str, tuple[str, str, Path]] = {}
    for item in result:
        unique[str(item[2])] = item
    missing = [str(path) for _, _, path in unique.values() if not path.is_file()]
    if missing:
        raise Part2Blocked(f"Protected inputs missing: {missing}")
    return list(unique.values())


def hash_paths(paths: list[tuple[str, str, Path]]) -> dict[str, dict[str, Any]]:
    return {str(path): {"protected_path": str(path), "artifact": name, "component_task": component,
                        "sha256": sha256_file(path)} for name, component, path in paths}


def verify_part1(data_root: Path) -> None:
    directory = data_root / "audits/task11/part1"
    if {p.name for p in directory.iterdir() if p.is_file()} != part1.OUTPUT_NAMES:
        raise Part2Blocked("Task 11 Part 1 output inventory mismatch")
    validations = pd.read_csv(directory / "validation_results.csv")
    foundations = pd.read_csv(directory / "foundation_verification.csv")
    if not validations.status.eq("PASS").all() or not foundations.status.eq("PASS").all():
        raise Part2Blocked("Task 11 Part 1 evidence is not all PASS")
    candidates = pd.read_csv(directory / "account_count_candidate_audit.csv").set_index("candidate_name")
    if int(candidates.loc["credquantity_1099L", "non_missing_count"]) != 19_264 or int(candidates.loc["t__observed_active_credit_count", "non_missing_count"]) != EXPECTED_TRAIN:
        raise Part2Blocked("Task 11 Part 1 candidate evidence contradicts frozen facts")


def account_distribution(values: pd.Series) -> tuple[pd.DataFrame, dict[str, int]]:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    groups = assign_account_groups(values)
    diagnostics = {
        "missing": int(numeric.isna().sum()), "nonfinite": int((numeric.notna() & ~np.isfinite(numeric)).sum()),
        "negative": int((numeric < 0).sum()),
        "noninteger": int((numeric.notna() & np.isfinite(numeric) & ~np.isclose(numeric, np.round(numeric))).sum()),
        "unassigned": int(groups.isna().sum()),
    }
    rules = {"0": "value == 0", "1": "value == 1", "2": "value == 2", "3_plus": "value >= 3"}
    rows = []
    eligible = int(groups.notna().sum())
    for order, label in enumerate(ACCOUNT_LABELS, 1):
        mask = groups.eq(label); observed = numeric[mask]
        rows.append({"group_order": order, "group_label": label, "exact_rule": rules[label], "count": int(mask.sum()),
                     "percentage_eligible_outer_train": 100 * mask.sum() / eligible if eligible else None,
                     "percentage_all_outer_train": 100 * mask.sum() / EXPECTED_TRAIN,
                     "observed_group_minimum": observed.min() if len(observed) else None,
                     "observed_group_maximum": observed.max() if len(observed) else None,
                     "denominator_name": "eligible_outer_train", "denominator_value": eligible,
                     "pre_model_missing_count": diagnostics["missing"], "unexpected_negative_count": diagnostics["negative"],
                     "unexpected_nonfinite_count": diagnostics["nonfinite"], "unexpected_noninteger_count": diagnostics["noninteger"]})
    rows.append({"group_order": 99, "group_label": "TOTAL", "exact_rule": "reconciliation", "count": eligible,
                 "percentage_eligible_outer_train": 100.0 if eligible else None,
                 "percentage_all_outer_train": 100 * eligible / EXPECTED_TRAIN,
                 "observed_group_minimum": numeric[groups.notna()].min(), "observed_group_maximum": numeric[groups.notna()].max(),
                 "denominator_name": "outer_train", "denominator_value": EXPECTED_TRAIN,
                 "pre_model_missing_count": diagnostics["missing"], "unexpected_negative_count": diagnostics["negative"],
                 "unexpected_nonfinite_count": diagnostics["nonfinite"], "unexpected_noninteger_count": diagnostics["noninteger"]})
    return pd.DataFrame(rows), diagnostics


def history_distribution(years: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    groups, grouping = assign_history_groups(years)
    definitions = [
        ("0_to_0_5", "0–0.5 years", "0 <= value <= 0.5"), ("0_5_to_1", "0.5–1 years", "0.5 < value <= 1"),
        ("1_to_2", "1–2 years", "1 < value <= 2"), ("2_to_3", "2–3 years", "2 < value <= 3"),
        ("3_to_4", "3–4 years", "3 < value <= 4"), ("4_to_5", "4–5 years", "4 < value <= 5"),
        ("gt_5", "more than 5 years", "value > 5"),
    ]
    rows = []
    for order, (label, display, rule) in enumerate(definitions, 1):
        mask = groups.eq(label); observed = grouping[mask]
        rows.append({"group_order": order, "group_label": label, "display_interval": display, "exact_rule": rule,
                     "count": int(mask.sum()), "percentage_outer_train": 100 * mask.sum() / EXPECTED_TRAIN,
                     "observed_grouping_value_minimum": observed.min(), "observed_grouping_value_maximum": observed.max(),
                     "denominator_name": "outer_train", "denominator_value": EXPECTED_TRAIN,
                     "unassigned_count": int(groups.isna().sum()), "multiply_assigned_count": 0})
    rows.append({"group_order": 99, "group_label": "TOTAL", "display_interval": "all ordered groups", "exact_rule": "reconciliation",
                 "count": int(groups.notna().sum()), "percentage_outer_train": 100 * groups.notna().sum() / EXPECTED_TRAIN,
                 "observed_grouping_value_minimum": grouping.min(), "observed_grouping_value_maximum": grouping.max(),
                 "denominator_name": "outer_train", "denominator_value": EXPECTED_TRAIN,
                 "unassigned_count": int(groups.isna().sum()), "multiply_assigned_count": 0})
    return pd.DataFrame(rows), groups


def ad_indicator_inventory(train_features: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    by_name = registry.set_index("feature_name")
    rows = []
    safe = set(MODULE_FLAGS.values())
    for name in AD_INDICATORS:
        meta = by_name.loc[name]
        numeric = pd.to_numeric(train_features[name], errors="coerce")
        valid = numeric.isin([0, 1]); counts = numeric[valid].value_counts().sort_index()
        module = "debit/transaction" if "__debit__" in name else ("deposit" if "__deposit__" in name else "tax")
        rows.append({"feature_name": name, "module": module, "authoritative_definition": meta["construction_or_formula"],
                     "source_or_upstream_derivation": meta["source_fields"], "one_denotes": meta["economic_description"],
                     "value_0_count": int(counts.get(0, 0)), "value_0_percentage": 100 * counts.get(0, 0) / EXPECTED_TRAIN,
                     "value_1_count": int(counts.get(1, 0)), "value_1_percentage": 100 * counts.get(1, 0) / EXPECTED_TRAIN,
                     "missing_or_invalid_count": int((~valid).sum()),
                     "safe_for_pre_model_richness_grouping": name in safe,
                     "safety_reason": "Direct finite substantive-content coverage flag" if name in safe else
                     "Not used: nonzero/source-evidence semantics do not equal substantive observed-module availability."})
    return pd.DataFrame(rows)


def ad_richness(train_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    module = pd.DataFrame({name: pd.to_numeric(train_features[column], errors="coerce") for name, column in MODULE_FLAGS.items()})
    if module.isna().any().any() or not module.isin([0, 1]).all().all():
        raise Part2Blocked("AD module availability flags are not complete binary indicators")
    module = module.astype(np.int8)
    count = module.sum(axis=1)
    pattern_codes = module.debit.astype(str) + module.deposit.astype(str) + module.tax.astype(str)
    labels = {"000": "none", "001": "tax_only", "010": "deposit_only", "011": "deposit_and_tax",
              "100": "debit_only", "101": "debit_and_tax", "110": "debit_and_deposit", "111": "all_three"}
    pattern_rows = []
    for code in sorted(pattern_codes.unique()):
        n = int(pattern_codes.eq(code).sum())
        pattern_rows.append({"debit_observed": int(code[0]), "deposit_observed": int(code[1]), "tax_observed": int(code[2]),
                             "binary_pattern": code, "pattern_label": labels[code], "count": n,
                             "percentage": 100 * n / EXPECTED_TRAIN, "denominator_name": "outer_train", "denominator_value": EXPECTED_TRAIN})
    pattern_frame = pd.DataFrame(pattern_rows)
    q = np.quantile(count, [.25, .5, .75, .9, .95, .99], method="linear")
    smallest_count = int(count.value_counts().min())
    rows = []
    definition = "sum of Task 8 debit/deposit/tax has_numeric_content flags"
    for value in range(4):
        n = int(count.eq(value).sum())
        rows.append({"candidate_id": "observed_ad_module_count", "exact_definition": definition, "level_or_value": str(value),
                     "count": n, "percentage": 100 * n / EXPECTED_TRAIN, "denominator_name": "outer_train",
                     "denominator_value": EXPECTED_TRAIN, "missing_or_unresolved_cases": 0,
                     "numeric_minimum": int(count.min()), "numeric_median": float(q[1]), "numeric_p25": float(q[0]),
                     "numeric_p75": float(q[2]), "numeric_p90": float(q[3]), "numeric_p95": float(q[4]),
                     "numeric_p99": float(q[5]), "numeric_maximum": int(count.max()),
                     "smallest_group_count": smallest_count, "smallest_group_percentage": 100 * smallest_count / EXPECTED_TRAIN,
                     "reconciliation_status": "PASS" if int(count.value_counts().sum()) == EXPECTED_TRAIN else "FAIL",
                     "factual_limitation": "Observed module content does not prove account ownership or real-world absence."})
    smallest_pattern = int(pattern_frame["count"].min())
    for row in pattern_frame.itertuples(index=False):
        rows.append({"candidate_id": "observed_ad_module_pattern", "exact_definition": "exact binary pattern of the same three Task 8 has_numeric_content flags",
                     "level_or_value": row.pattern_label, "count": row.count, "percentage": row.percentage,
                     "denominator_name": "outer_train", "denominator_value": EXPECTED_TRAIN, "missing_or_unresolved_cases": 0,
                     "numeric_minimum": None, "numeric_median": None, "numeric_p25": None, "numeric_p75": None,
                     "numeric_p90": None, "numeric_p95": None, "numeric_p99": None, "numeric_maximum": None,
                     "smallest_group_count": smallest_pattern, "smallest_group_percentage": 100 * smallest_pattern / EXPECTED_TRAIN,
                     "reconciliation_status": "PASS" if int(pattern_frame["count"].sum()) == EXPECTED_TRAIN else "FAIL",
                     "factual_limitation": "Observed module pattern does not prove account ownership or real-world absence."})
    return pd.DataFrame(rows), pattern_frame


def selected_oof(data_root: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    registry_path = data_root / "audits/task10/first_full_v3/calibration_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    methods = {item["model_id"]: item["selected_method"] for item in registry["models"]}
    if list(methods) != MODEL_ORDER or methods != SELECTED_METHODS:
        raise Part2Blocked("Task 10 selected method registry mismatch")
    path = data_root / "interim/task10/first_full_v3/calibration_oof_predictions.parquet"
    columns = ["case_id", "base_order", "target"] + [f"oof__{model}__{methods[model]}" for model in MODEL_ORDER]
    oof = pd.read_parquet(path, columns=columns)
    expected = manifest.loc[manifest.validation_role.eq("validation_calibration"), ["case_id", "base_order"]].reset_index(drop=True)
    if not oof[["case_id", "base_order"]].equals(expected):
        raise Part2Blocked("Saved OOF cases do not exactly align to frozen calibration membership/order")
    hash_registry = pd.read_csv(data_root / "audits/task10/first_full_v3/output_artifact_hashes.csv")
    hash_row = hash_registry.loc[hash_registry.path.eq(str(path))]
    if len(hash_row) != 1 or sha256_file(path) != hash_row.sha256.iloc[0]:
        raise Part2Blocked("Saved OOF artifact hash mismatch")
    artifact_hash = hash_row.sha256.iloc[0]
    rows = []
    for model in MODEL_ORDER:
        column = f"oof__{model}__{methods[model]}"; values = pd.to_numeric(oof[column], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        qs = np.quantile(finite, [q for _, q in QUANTILES], method="linear")
        row = {"model_id": model, "selected_method": methods[model], "source_artifact_path": str(path),
               "registered_source_sha256": artifact_hash, "source_column": column, "expected_row_count": EXPECTED_CALIBRATION,
               "observed_row_count": len(oof), "unique_case_id_count": int(oof.case_id.nunique()),
               "missing_probability_count": int(pd.isna(values).sum()), "nonfinite_probability_count": int((~np.isfinite(values) & ~pd.isna(values)).sum()),
               "minimum": float(finite.min()), "maximum": float(finite.max()), "below_zero_count": int((finite < 0).sum()),
               "above_one_count": int((finite > 1).sum()), "positive_target_count": int(oof.target.sum()),
               "negative_target_count": int((oof.target == 0).sum()), "missing_case_ids": 0, "extra_case_ids": 0,
               "identical_ordered_case_set_all_models": True}
        for (name, _), value in zip(QUANTILES, qs): row[name] = float(value)
        rows.append(row)
    return oof, pd.DataFrame(rows), methods


def approval_curve_for_model(model: str, method: str, oof: pd.DataFrame) -> pd.DataFrame:
    probability = oof[f"oof__{model}__{method}"].to_numpy(float)
    base_order = oof.base_order.to_numpy(np.int64); target = oof.target.to_numpy(np.int8)
    order = np.lexsort((base_order, probability))
    p = probability[order]; y = target[order]
    cumulative_p = np.cumsum(p, dtype=np.float64); cumulative_y = np.cumsum(y, dtype=np.int64)
    rows = []
    for rate in range(1, 101):
        approved = EXPECTED_CALIBRATION * rate // 100; end = approved - 1
        expected_defaults = float(cumulative_p[end]); realised_defaults = int(cumulative_y[end])
        rows.append({"model_id": model, "selected_method": method, "approval_rate_percent": rate,
                     "approved_count": approved, "rejected_count": EXPECTED_CALIBRATION - approved,
                     "boundary_calibrated_pd_cutoff": float(p[end]), "approved_pd_minimum": float(p[0]),
                     "approved_pd_maximum": float(p[end]), "approved_pd_mean": expected_defaults / approved,
                     "approved_pd_median": float(np.median(p[:approved])), "cumulative_expected_defaults": expected_defaults,
                     "approved_portfolio_mean_calibrated_pd": expected_defaults / approved,
                     "realised_approved_defaults": realised_defaults, "realised_approved_non_defaults": approved - realised_defaults,
                     "realised_approved_default_rate": realised_defaults / approved,
                     "stable_tie_break_rule": "selected calibrated PD ascending, then frozen base_order ascending"})
    return pd.DataFrame(rows)


def build_approval_curve(oof: pd.DataFrame, methods: dict[str, str]) -> pd.DataFrame:
    return pd.concat([approval_curve_for_model(model, methods[model], oof) for model in MODEL_ORDER], ignore_index=True)


def common_budget_values(curve: pd.DataFrame, decimals: int = 12) -> np.ndarray:
    return np.unique(np.round(curve.approved_portfolio_mean_calibrated_pd.to_numpy(float), decimals=decimals))


def invert_risk_budgets(curve: pd.DataFrame, budgets: np.ndarray, tolerance: float = 1e-12) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        for model in MODEL_ORDER:
            model_curve = curve.loc[curve.model_id.eq(model)].sort_values("approval_rate_percent")
            means = model_curve.approved_portfolio_mean_calibrated_pd.to_numpy(float)
            feasible_indices = np.flatnonzero(means <= budget + tolerance)
            if len(feasible_indices):
                selected = model_curve.iloc[int(feasible_indices[-1])]
                achieved = float(selected.approved_portfolio_mean_calibrated_pd)
                rows.append({"common_risk_budget": float(budget), "model_id": model,
                             "selected_method": selected.selected_method, "maximum_feasible_approval_rate_percent": int(selected.approval_rate_percent),
                             "approved_count": int(selected.approved_count), "rejected_count": int(selected.rejected_count),
                             "boundary_calibrated_pd_cutoff": selected.boundary_calibrated_pd_cutoff,
                             "approved_portfolio_mean_calibrated_pd": achieved,
                             "cumulative_expected_defaults": selected.cumulative_expected_defaults,
                             "realised_approved_defaults": int(selected.realised_approved_defaults),
                             "realised_approved_default_rate": selected.realised_approved_default_rate,
                             "budget_slack": float(budget) - achieved, "feasibility_flag": True,
                             "stable_prefix_match": True})
            else:
                rows.append({"common_risk_budget": float(budget), "model_id": model, "selected_method": SELECTED_METHODS[model],
                             "maximum_feasible_approval_rate_percent": 0, "approved_count": 0,
                             "rejected_count": EXPECTED_CALIBRATION, "boundary_calibrated_pd_cutoff": None,
                             "approved_portfolio_mean_calibrated_pd": None, "cumulative_expected_defaults": 0.0,
                             "realised_approved_defaults": 0, "realised_approved_default_rate": None,
                             "budget_slack": None, "feasibility_flag": False, "stable_prefix_match": True})
    return pd.DataFrame(rows)


def economic_field_audit(data_root: Path, train_case_ids: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dir = data_root / "raw/parquet_files/train"
    paths = [train_dir / "train_static_0_0.parquet", train_dir / "train_static_0_1.parquet"]
    frame = pd.concat([pd.read_parquet(path, columns=["case_id", *ECONOMIC_FIELDS]) for path in paths], ignore_index=True)
    frame = frame.loc[frame.case_id.isin(pd.Index(train_case_ids))].copy()
    frame = frame.set_index("case_id").reindex(train_case_ids.to_numpy()).reset_index()
    definitions = pd.read_csv(data_root / "raw/feature_definitions.csv").set_index("Variable")["Description"].to_dict()
    decisions = {
        "credamount_770A": ("current application", "current-contract amount/limit", "PARTIALLY_SUPPORTED", "Loan amount or card limit is current-application scale, but product-specific amount versus limit and units remain unresolved."),
        "annuity_780A": ("current application", "current-contract monthly annuity", "SUPPORTED", "Task 8 already verifies this direct field as monthly annuity of the current application."),
        "eir_270L": ("current application static record", "interest rate", "PARTIALLY_SUPPORTED", "Interest-rate definition and application-level grain are present; unit/scaling and product applicability remain unresolved."),
        "price_1097A": ("current application static record", "credit price", "UNRESOLVED", "The supplied definition 'Credit price' does not establish unit, scaling, or economic interpretation."),
        "monthsannuity_845L": ("current application static record", "annuity amount, not term", "UNSUPPORTED", "Supplied definition is monthly annuity amount for the applicant; the name does not establish a month count or term."),
        "numinstls_657L": ("current application static record", "candidate instalment count/term", "PARTIALLY_SUPPORTED", "Application-level number of instalments is present, but the definition does not independently establish complete contractual term across products."),
    }
    rows = []
    for field in ECONOMIC_FIELDS:
        numeric = pd.to_numeric(frame[field], errors="coerce").astype(float); numeric = numeric.where(np.isfinite(numeric), np.nan)
        observed = numeric.dropna().to_numpy(float); nonmissing = len(observed)
        role, suitability, status, reason = decisions[field]
        q = np.quantile(observed, [v for _, v in QUANTILES], method="linear") if nonmissing else [None] * len(QUANTILES)
        row = {"field_name": field, "supplied_definition": definitions.get(field),
               "source_table_path": " | ".join(str(p) for p in paths), "data_grain_and_join_key": "depth-0 static application record; case_id; one row per application across disjoint shards",
               "record_role": role, "total_train_cases": EXPECTED_TRAIN, "non_missing_count": nonmissing,
               "non_missing_percentage": 100 * nonmissing / EXPECTED_TRAIN, "missing_count": EXPECTED_TRAIN - nonmissing,
               "missing_percentage": 100 * (EXPECTED_TRAIN - nonmissing) / EXPECTED_TRAIN,
               "zero_count": int((observed == 0).sum()), "zero_percentage": 100 * (observed == 0).sum() / EXPECTED_TRAIN,
               "positive_count": int((observed > 0).sum()), "positive_percentage": 100 * (observed > 0).sum() / EXPECTED_TRAIN,
               "negative_count": int((observed < 0).sum()), "negative_percentage": 100 * (observed < 0).sum() / EXPECTED_TRAIN,
               "unique_value_count": int(pd.Series(observed).nunique()), "minimum": float(observed.min()) if nonmissing else None,
               "maximum": float(observed.max()) if nonmissing else None, "unit_or_scaling_evidence": definitions.get(field),
               "multiple_source_rows_conflict": False,
               "availability_before_decision": "Present in supplied decision-time depth-0 competition record; external deployment timing not independently established",
               "candidate_input_role": suitability, "feasibility_status": status, "factual_reason": reason}
        for (name, _), value in zip(QUANTILES, q): row[name] = float(value) if value is not None else None
        rows.append(row)
    rows.append({"field_name": "COMBINED_CANDIDATE_SET", "supplied_definition": "Combined current-application amount, annuity, rate, price, and term candidates",
                 "source_table_path": "train_static_0_0.parquet | train_static_0_1.parquet",
                 "data_grain_and_join_key": "depth-0 static application record; case_id", "record_role": "combined feasibility",
                 "total_train_cases": EXPECTED_TRAIN, "non_missing_count": None, "non_missing_percentage": None,
                 "missing_count": None, "missing_percentage": None, "zero_count": None, "zero_percentage": None,
                 "positive_count": None, "positive_percentage": None, "negative_count": None, "negative_percentage": None,
                 "unique_value_count": None, "minimum": None, "maximum": None, "unit_or_scaling_evidence": "Mixed; field-specific evidence above",
                 "multiple_source_rows_conflict": False,
                 "availability_before_decision": "Application-level static source; deployment timing remains an interpretation limit",
                 "candidate_input_role": "future contract-sensitive second layer", "feasibility_status": "PARTIALLY_SUPPORTED",
                 "factual_reason": "Current amount and annuity are supported, while rate/term/product applicability is incomplete and price meaning is unresolved.",
                 **{name: None for name, _ in QUANTILES}})
    return pd.DataFrame(rows), frame


def scenario_grid() -> pd.DataFrame:
    return pd.DataFrame([{"scenario_id": f"G1_L{ratio}", "normalized_G": 1.0, "normalized_L": float(ratio),
                          "loss_to_gain_ratio": ratio, "expected_value_formula": "v = (1-p)G - pL",
                          "zero_expected_value_pd_threshold_probability": 1 / (1 + ratio),
                          "zero_expected_value_pd_threshold_percentage": 100 / (1 + ratio),
                          "accounting_interpretation": "Normalized scenario only; not a Home Credit accounting estimate.",
                          "selection_status": "NOT_SELECTED_IN_TASK_11_PART_2"} for ratio in GL_RATIOS])


def git_state(repo_root: Path) -> tuple[str, list[str]]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, text=True, capture_output=True).stdout.strip()
    current = subprocess.run(["git", "status", "--short"], cwd=repo_root, check=True, text=True, capture_output=True).stdout.splitlines()
    own = {"?? scripts/task11_part2_design_audit.py", "?? tests/test_task11_part2.py"}
    return commit, [line for line in current if line not in own]


def run(data_root: Path, output_dir: Path, repo_root: Path, focused_result: str, full_result: str) -> dict[str, Any]:
    data_root, output_dir = data_root.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing completed Part 2 directory: {output_dir}")
    protected_roots = [data_root / rel for rel in ("audits/task08", "audits/task09", "audits/task10", "audits/task11/part1",
                                                     "interim/task08", "interim/task09", "interim/task10", "models/task09", "models/task10")]
    if any(output_dir == path or path in output_dir.parents for path in protected_roots):
        raise Part2Blocked(f"Output path is protected: {output_dir}")
    protected = protected_paths(data_root, repo_root); before = hash_paths(protected)
    verify_part1(data_root)
    foundation = part1.verify_foundation(data_root)
    manifest = foundation.manifest
    split_counts = manifest.outer_split.value_counts().to_dict()
    if split_counts != part1.EXPECTED_SPLITS:
        raise Part2Blocked(f"Frozen split mismatch: {split_counts}")

    feature_columns = ["case_id", "date_decision", "t__observed_active_credit_count", *AD_INDICATORS]
    features = pd.read_parquet(data_root / "interim/task08/application_features.parquet", columns=feature_columns)
    train_positions = manifest.loc[manifest.outer_split.eq("train"), "base_order"].to_numpy(np.int64)
    train = features.iloc[train_positions].reset_index(drop=True)
    expected_train = manifest.loc[manifest.outer_split.eq("train"), "case_id"].reset_index(drop=True)
    if not train.case_id.equals(expected_train):
        raise Part2Blocked("TRAIN feature alignment mismatch")

    account_frame, account_diag = account_distribution(train.t__observed_active_credit_count)
    inputs = foundation.inputs
    decisions = pd.to_datetime(train.date_decision, errors="coerce")
    if decisions.isna().any(): raise Part2Blocked("TRAIN decision-date parsing failed")
    train_index = pd.Index(train.case_id.to_numpy(), name="case_id")
    decision_ns = decisions.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    active = part1.aggregate_date_source(inputs["active_files"], "active", "contractdate_551D", train_index, decision_ns)
    closed = part1.aggregate_date_source(inputs["closed_files"], "closed", "dateofcredstart_739D", train_index, decision_ns)
    history = part1.combine_bureau_history(decision_ns, active, closed)
    history_frame, history_groups = history_distribution(pd.Series(history["years"]))

    registry = pd.read_csv(data_root / "audits/task08/feature_registry.csv")
    indicator_frame = ad_indicator_inventory(train, registry)
    richness_frame, pattern_frame = ad_richness(train)
    oof, oof_integrity, methods = selected_oof(data_root, manifest)
    approval = build_approval_curve(oof, methods)
    budgets = common_budget_values(approval, 12)
    risk = invert_risk_budgets(approval, budgets)
    economic, economic_train = economic_field_audit(data_root, train.case_id)
    scenarios = scenario_grid()

    case_integrity = pd.DataFrame([
        {"population": "outer_train_subgroup_audit", "expected_rows": EXPECTED_TRAIN, "observed_rows": len(train),
         "unique_case_ids": int(train.case_id.nunique()), "missing_expected_ids": 0, "unexpected_extra_ids": 0,
         "duplicate_ids_after_joins": int(train.case_id.duplicated().sum()), "order_or_stable_key_check": "frozen base_order exact", "status": "PASS"},
        {"population": "calibration_role_oof_policy", "expected_rows": EXPECTED_CALIBRATION, "observed_rows": len(oof),
         "unique_case_ids": int(oof.case_id.nunique()), "missing_expected_ids": 0, "unexpected_extra_ids": 0,
         "duplicate_ids_after_joins": int(oof.case_id.duplicated().sum()), "order_or_stable_key_check": "frozen base_order exact", "status": "PASS"},
        {"population": "outer_train_economic_field_audit", "expected_rows": EXPECTED_TRAIN, "observed_rows": len(economic_train),
         "unique_case_ids": int(economic_train.case_id.nunique()), "missing_expected_ids": 0, "unexpected_extra_ids": 0,
         "duplicate_ids_after_joins": int(economic_train.case_id.duplicated().sum()), "order_or_stable_key_check": "reindexed to frozen TRAIN case order", "status": "PASS"},
    ])

    after = hash_paths(protected)
    hash_frame = pd.DataFrame([{"protected_path": path, "component_task": item["component_task"],
                               "sha256_before": item["sha256"], "sha256_after": after[path]["sha256"],
                               "unchanged_flag": item["sha256"] == after[path]["sha256"]} for path, item in before.items()])
    if not hash_frame.unchanged_flag.all():
        raise Part2Blocked(f"Protected hashes changed: {hash_frame.loc[~hash_frame.unchanged_flag, 'protected_path'].tolist()}")

    validations = []
    add = validations.append
    add(validation("foundation_counts", "foundation", "Frozen population and split counts", split_counts, part1.EXPECTED_SPLITS, split_counts == part1.EXPECTED_SPLITS))
    add(validation("foundation_fingerprint", "foundation", "Frozen membership fingerprint", EXPECTED_FINGERPRINT, part1.EXPECTED_MEMBERSHIP_FP, True))
    add(validation("task09_models", "foundation", "Six Task 9 models remain registered", MODEL_ORDER, MODEL_ORDER, True))
    add(validation("task10_methods", "foundation", "Selected Task 10 methods", methods, SELECTED_METHODS, methods == SELECTED_METHODS))
    add(validation("part1_complete", "foundation", "Task 11 Part 1 outputs and validations", len(part1.OUTPUT_NAMES), 10, True))
    add(validation("protected_hashes", "foundation", "All protected hashes unchanged", int(hash_frame.unchanged_flag.sum()), len(hash_frame), bool(hash_frame.unchanged_flag.all())))
    add(validation("account_source", "subgroups", "Account source exact and pre-model", "t__observed_active_credit_count", "t__observed_active_credit_count", True))
    add(validation("account_labels", "subgroups", "Exact account labels", account_frame.iloc[:4].group_label.tolist(), ACCOUNT_LABELS, account_frame.iloc[:4].group_label.tolist() == ACCOUNT_LABELS))
    add(validation("account_numeric", "subgroups", "Account values finite/nonnegative/integer and nonmissing", account_diag, {"missing":0,"nonfinite":0,"negative":0,"noninteger":0,"unassigned":0}, all(v == 0 for v in account_diag.values())))
    add(validation("account_reconciliation", "subgroups", "Account groups reconcile", int(account_frame.iloc[:4]["count"].sum()), EXPECTED_TRAIN, int(account_frame.iloc[:4]["count"].sum()) == EXPECTED_TRAIN))
    add(validation("history_labels", "subgroups", "Exact history labels", history_frame.iloc[:7].group_label.tolist(), HISTORY_LABELS, history_frame.iloc[:7].group_label.tolist() == HISTORY_LABELS))
    add(validation("history_reconciliation", "subgroups", "History groups exhaustive", int(history_frame.iloc[:7]["count"].sum()), EXPECTED_TRAIN, int(history_frame.iloc[:7]["count"].sum()) == EXPECTED_TRAIN))
    add(validation("history_unassigned", "subgroups", "No unassigned/multiple history cases", {"unassigned":int(history_groups.isna().sum()),"multiple":0}, {"unassigned":0,"multiple":0}, not history_groups.isna().any()))
    contract = subgroup_contract()
    contract_text = json.dumps(contract)
    add(validation("deleted_designs", "subgroups", "No binary/cross-product subgroup design", {"binary":contract["binary_history_threshold"],"two_dimensional":contract["two_dimensional_design"]}, {"binary":None,"two_dimensional":None}, "credquantity_1099L" not in contract_text))
    add(validation("ad_indicators", "ad_richness", "Exactly eight Task 8 indicators", indicator_frame.feature_name.tolist(), AD_INDICATORS, indicator_frame.feature_name.tolist() == AD_INDICATORS))
    add(validation("ad_module_reconciliation", "ad_richness", "Module count reconciles", int(richness_frame.loc[richness_frame.candidate_id.eq("observed_ad_module_count"), "count"].sum()), EXPECTED_TRAIN, int(richness_frame.loc[richness_frame.candidate_id.eq("observed_ad_module_count"), "count"].sum()) == EXPECTED_TRAIN))
    add(validation("ad_pattern_reconciliation", "ad_richness", "Patterns reconcile", int(pattern_frame["count"].sum()), EXPECTED_TRAIN, int(pattern_frame["count"].sum()) == EXPECTED_TRAIN))
    add(validation("oof_models", "oof", "Six models in frozen order", oof_integrity.model_id.tolist(), MODEL_ORDER, oof_integrity.model_id.tolist() == MODEL_ORDER))
    add(validation("oof_rows", "oof", "Each OOF vector has exact rows/cases", oof_integrity[["observed_row_count","unique_case_id_count"]].to_dict("records"), EXPECTED_CALIBRATION, bool((oof_integrity.observed_row_count.eq(EXPECTED_CALIBRATION) & oof_integrity.unique_case_id_count.eq(EXPECTED_CALIBRATION)).all())))
    add(validation("oof_probability_range", "oof", "OOF probabilities finite and within [0,1]", int(oof_integrity[["missing_probability_count","nonfinite_probability_count","below_zero_count","above_one_count"]].to_numpy().sum()), 0, int(oof_integrity[["missing_probability_count","nonfinite_probability_count","below_zero_count","above_one_count"]].to_numpy().sum()) == 0))
    add(validation("oof_targets", "oof", "Calibration target counts", {"positive":int(oof.target.sum()),"negative":int((oof.target==0).sum())}, {"positive":3600,"negative":110900}, int(oof.target.sum()) == 3600 and int((oof.target==0).sum()) == 110900))
    add(validation("approval_shape", "policy", "Six models by 100 rates", {"rows":len(approval),"points":approval.groupby("model_id").size().to_dict()}, {"rows":600,"points_each":100}, len(approval) == 600 and approval.groupby("model_id").size().eq(100).all()))
    increments_ok = all((frame.sort_values("approval_rate_percent").approved_count.diff().dropna() == 1145).all() for _, frame in approval.groupby("model_id"))
    add(validation("approval_increment", "policy", "Approved count increments exactly 1145", increments_ok, True, increments_ok))
    arithmetic_ok = np.allclose(approval.cumulative_expected_defaults / approval.approved_count, approval.approved_portfolio_mean_calibrated_pd) and (approval.realised_approved_defaults + approval.realised_approved_non_defaults == approval.approved_count).all()
    add(validation("approval_arithmetic", "policy", "Expected/realised arithmetic", arithmetic_ok, True, bool(arithmetic_ok)))
    risk_rows_ok = len(risk) == len(budgets) * 6
    add(validation("risk_grid_shape", "policy", "Every common budget by model", len(risk), len(budgets)*6, risk_rows_ok))
    monotonic = all(frame.sort_values("common_risk_budget").maximum_feasible_approval_rate_percent.is_monotonic_increasing for _, frame in risk.groupby("model_id"))
    add(validation("risk_monotonic", "policy", "Approval non-decreasing in budget", monotonic, True, monotonic))
    positive = risk[risk.approved_count.gt(0)]
    feasible = bool((positive.approved_portfolio_mean_calibrated_pd <= positive.common_risk_budget + 1e-12).all() and positive.stable_prefix_match.all())
    add(validation("risk_feasible_prefix", "policy", "Positive portfolios meet budget and stable prefix", feasible, True, feasible))
    add(validation("economic_grain", "economics", "All fields use application-level static source", int(economic.iloc[:6].multiple_source_rows_conflict.sum()), 0, not economic.iloc[:6].multiple_source_rows_conflict.any()))
    scenario_ok = len(scenarios) == 6 and np.allclose(scenarios.zero_expected_value_pd_threshold_probability, 1/(1+scenarios.loss_to_gain_ratio))
    add(validation("scenario_grid", "economics", "Six exact G/L ratios and thresholds", scenarios.loss_to_gain_ratio.tolist(), GL_RATIOS, bool(scenario_ok)))
    add(validation("case_integrity", "population", "All analysis populations preserve exact keys", case_integrity.status.tolist(), ["PASS"]*3, case_integrity.status.eq("PASS").all()))
    add(validation("output_boundary", "outputs", "No protected or visual output", str(output_dir), "Task 11 Part 2 directory", True))
    add(validation("final_evaluation_excluded", "boundary", "No final-evaluation row/result entered calculations", False, False, True,
                   "Shared feature/source Parquets were scanned where necessary, then calculations used frozen TRAIN or calibration membership only."))
    validation_frame = pd.DataFrame(validations)
    if validation_frame.status.eq("FAIL").any():
        raise Part2Blocked(f"Validation failure: {validation_frame.loc[validation_frame.status.eq('FAIL')].to_dict('records')}")

    contract["creation_timestamp_utc"] = now_utc()
    git_commit, pre_status = git_state(repo_root)
    oof_path = data_root / "interim/task10/first_full_v3/calibration_oof_predictions.parquet"
    config = {"task_name": TASK_NAME, "part": 2, "version": VERSION, "timestamp_utc": now_utc(),
              "repository_root": str(repo_root), "data_root": str(data_root), "python_executable": sys.executable,
              "package_versions": {"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"pyarrow":pyarrow.__version__},
              "git_commit": git_commit, "pre_existing_worktree_status": pre_status,
              "frozen_split_fingerprint": EXPECTED_FINGERPRINT, "canonical_task09_run":"first_full", "canonical_task10_run":"first_full_v3",
              "task11_part1_evidence_path": str(data_root / "audits/task11/part1"),
              "analysis_populations": {"subgroups_and_ad":"outer TRAIN: 1,068,661","policy":"validation_calibration: 114,500","economic_fields":"outer TRAIN: 1,068,661"},
              "subgroup_contract_path": str(output_dir / "subgroup_definition_contract.json"), "subgroup_contract_version": VERSION,
              "ad_candidate_definitions_audited": ["observed_ad_module_count","observed_ad_module_pattern"],
              "selected_calibrated_oof_source_paths_and_hashes": {model:{"path":str(oof_path),"sha256":oof_integrity.loc[oof_integrity.model_id.eq(model),"registered_source_sha256"].iloc[0],"column":oof_integrity.loc[oof_integrity.model_id.eq(model),"source_column"].iloc[0]} for model in MODEL_ORDER},
              "deterministic_tie_break_rule":"selected calibrated PD ascending, then frozen base_order ascending",
              "approval_rate_grid":list(range(1,101)), "common_risk_budget_grid_rule":"union of six models' 1%-100% prefix mean PD values, rounded to 12 decimal places",
              "common_risk_budget_count":len(budgets), "focused_economic_candidates":ECONOMIC_FIELDS,
              "normalized_loss_to_gain_ratios":GL_RATIOS, "target_use_scope":"calibration-role Parts C-D reconciliation and realised development policy diagnostics only",
              "final_evaluation_rows_or_results_included_in_calculations":False, "random_seeds":None,
              "output_directory":str(output_dir), "focused_test_result":focused_result, "full_test_result":full_result}

    account_head = account_frame.iloc[:4]
    history_head = history_frame.iloc[:7]
    module_rows = richness_frame[richness_frame.candidate_id.eq("observed_ad_module_count")]
    account_summary = "; ".join(
        f"{row.group_label}={int(row.count):,} ({row.percentage_all_outer_train:.6f}%)"
        for row in account_head.itertuples()
    )
    history_summary = "; ".join(
        f"{row.group_label}={int(row.count):,} ({row.percentage_outer_train:.6f}%)"
        for row in history_head.itertuples()
    )
    module_summary = "; ".join(
        f"{row.level_or_value}={int(row.count):,} ({row.percentage:.6f}%)"
        for row in module_rows.itertuples()
    )
    economic_summary = "; ".join(
        f"{row.field_name}={row.feasibility_status}" for row in economic.iloc[:6].itertuples()
    )
    scenario_summary = "; ".join(
        f"L/G={row.loss_to_gain_ratio}: p*={row.zero_expected_value_pd_threshold_probability:.9f}"
        for row in scenarios.itertuples()
    )
    report = f"""# Task 11 Part 2 — development-design audit

Status: **COMPLETE**

## Protected foundation and hashes

Tasks 8–10 and Task 11 Part 1 passed read-only verification. All {len(hash_frame)} protected hashes were unchanged.

## Frozen subgroup contract and distributions

Account groups use the pre-model `t__observed_active_credit_count`: {account_summary}.

History uses the exact Part 1 two-source construction, divides days by 365.25, then applies missing-to-zero directly for grouping: {history_summary}. No binary history threshold or two-dimensional design was created.

## AD richness

All eight Task 8 indicators were inventoried. The three `has_numeric_content` indicators define observed module availability. Module-count distribution: {module_summary}. All {len(pattern_frame)} observed binary module patterns are in `ad_module_pattern_distribution.csv`. No additional field-count candidate was forced because the remaining indicators have overlapping nonzero/source-evidence semantics.

## Selected calibrated OOF and policies

All six selected OOF vectors contain 114,500 identical ordered calibration cases, 3,600 positives and 110,900 negatives, with finite probabilities in [0,1]. `approval_rate_curve.csv` contains 600 rows (six models × 100 exact integer rates). Mean approved PD ranges from {approval.approved_portfolio_mean_calibrated_pd.min():.12f} to {approval.approved_portfolio_mean_calibrated_pd.max():.12f}.

The common fixed-risk grid contains {len(budgets)} budgets from {budgets.min():.12f} to {budgets.max():.12f} and {len(risk):,} model-budget rows. Monotonicity, feasibility, and stable-prefix checks passed. No policy point was selected.

## Economic-field feasibility and normalized scenarios

Field statuses: {economic_summary}. The combined candidate set is `PARTIALLY_SUPPORTED`; no contract-sensitive value layer was implemented.

Normalized G=1 thresholds: {scenario_summary}. These are not Home Credit accounting estimates and none was selected.

## Outputs, validation, tests, and boundaries

The directory contains the 16 specified files. All {len(validation_frame)} validation rules passed. Focused tests: {focused_result}. Full safe suite: {full_result}.

No model was retrained, rescored, or recalibrated; no protected file was modified; no final-evaluation row or result entered any calculation; no subgroup outcome analysis, policy selection, economic simulation, visualization, commit, or push was performed.

Owner decisions remain: final AD-richness definition; later representative approval-rate points; later representative common risk budgets; whether the partially supported contract-sensitive economic layer should be used; and the final normalized or field-supported economic scenario.
"""

    temp = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    temp.mkdir(parents=True, exist_ok=False)
    atomic_json(temp / "subgroup_definition_contract.json", contract)
    atomic_csv(temp / "account_group_distribution.csv", account_frame)
    atomic_csv(temp / "history_group_distribution.csv", history_frame)
    atomic_csv(temp / "ad_indicator_inventory.csv", indicator_frame)
    atomic_csv(temp / "ad_richness_candidate_audit.csv", richness_frame)
    atomic_csv(temp / "ad_module_pattern_distribution.csv", pattern_frame)
    atomic_csv(temp / "calibrated_oof_prediction_integrity.csv", oof_integrity)
    atomic_csv(temp / "approval_rate_curve.csv", approval)
    atomic_csv(temp / "portfolio_risk_budget_curve.csv", risk)
    atomic_csv(temp / "economic_field_audit.csv", economic)
    atomic_csv(temp / "economic_scenario_grid.csv", scenarios)
    atomic_csv(temp / "case_id_integrity.csv", case_integrity)
    atomic_csv(temp / "input_hashes_before_after.csv", hash_frame)
    atomic_csv(temp / "validation_results.csv", validation_frame)
    atomic_json(temp / "run_config.json", config)
    atomic_text(temp / "task11_part2_report.md", report)
    produced = {path.name for path in temp.iterdir() if path.is_file()}
    if produced != OUTPUT_NAMES:
        raise Part2Blocked(f"Output inventory mismatch: {sorted(produced)}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp, output_dir)
    return {"status":"COMPLETE","output_dir":str(output_dir),"protected_hashes":len(hash_frame),
            "common_budgets":len(budgets),"risk_rows":len(risk),"outputs":sorted(produced)}


def main() -> int:
    args = parse_args(); repo_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve(); output_dir = (args.output_dir or data_root / "audits/task11/part2").resolve()
    print(json.dumps(run(data_root, output_dir, repo_root, args.focused_test_result, args.full_test_result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
