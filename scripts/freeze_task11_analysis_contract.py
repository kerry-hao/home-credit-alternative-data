#!/usr/bin/env python3
"""Task 11 Part 3: freeze the future final-evaluation analysis contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import audit_thin_file_candidates as part1
import calibrate_baseline_models as task10
import task11_part2_design_audit as part2


VERSION = "1.0.0"
TASK_NAME = "TASK_11_PART_3_FINAL_EVALUATION_CONTRACT_FREEZE"
EXPECTED_FINGERPRINT = "8bfb238774655f46a4f21c2668b2815f321c9298cb6d38a8da33f35a1ab91714"
POPULATION_COUNTS = {
    "total_applications": 1_526_659,
    "outer_train": 1_068_661,
    "outer_validation": 228_999,
    "outer_final_evaluation": 228_999,
    "validation_tuning": 114_499,
    "validation_calibration": 114_500,
    "calibration_positives": 3_600,
    "calibration_negatives": 110_900,
}
MODEL_ORDER = [
    "logit_T", "logit_T_plus_AD", "lightgbm_T", "lightgbm_T_plus_AD", "mlp_T", "mlp_T_plus_AD",
]
SELECTED_METHODS = {
    "logit_T": "identity",
    "logit_T_plus_AD": "identity",
    "lightgbm_T": "identity",
    "lightgbm_T_plus_AD": "identity",
    "mlp_T": "logistic",
    "mlp_T_plus_AD": "identity",
}
PRIMARY_COMPARISON = ["lightgbm_T", "lightgbm_T_plus_AD"]
SECONDARY_COMPARISON = ["logit_T_plus_AD", "lightgbm_T_plus_AD", "mlp_T_plus_AD"]
ALGORITHM_PAIRS = {
    "logit": ["logit_T", "logit_T_plus_AD"],
    "lightgbm": ["lightgbm_T", "lightgbm_T_plus_AD"],
    "mlp": ["mlp_T", "mlp_T_plus_AD"],
}
PRE_MODEL_BOUNDARY = (
    "assign from pre-model analytic values before model imputation, winsorization, scaling, encoding, "
    "standardization, or any other fitted model transformation"
)
ACCOUNT_GROUPS = [
    {"order": 1, "label": "0", "rule": "value == 0"},
    {"order": 2, "label": "1", "rule": "value == 1"},
    {"order": 3, "label": "2", "rule": "value == 2"},
    {"order": 4, "label": "3_plus", "rule": "value >= 3"},
]
HISTORY_GROUPS = [
    {"order": 1, "label": "0_to_0_5", "rule": "0 <= value <= 0.5"},
    {"order": 2, "label": "0_5_to_1", "rule": "0.5 < value <= 1"},
    {"order": 3, "label": "1_to_2", "rule": "1 < value <= 2"},
    {"order": 4, "label": "2_to_3", "rule": "2 < value <= 3"},
    {"order": 5, "label": "3_to_4", "rule": "3 < value <= 4"},
    {"order": 6, "label": "4_to_5", "rule": "4 < value <= 5"},
    {"order": 7, "label": "gt_5", "rule": "value > 5"},
]
AD_INDICATORS = [
    "ad__debit__has_numeric_content",
    "ad__deposit__has_numeric_content",
    "ad__tax__has_numeric_content",
]
AD_GROUPS = [
    {"order": index + 1, "label": str(index), "rule": f"observed module count equals {index}"}
    for index in range(4)
]
PROHIBITED_SUBGROUP_DESIGNS = [
    "separate history-missing category",
    "history-missing indicator",
    "binary history-thin threshold",
    "account-by-history cross-product",
    "cross-product among account, history, and AD-richness dimensions",
    "PCA or composite thickness index",
    "eight exact binary AD-module patterns as the primary grouping",
]
APPROVAL_ANCHORS = [Decimal("40.00"), Decimal("50.00"), Decimal("60.00"), Decimal("70.00"), Decimal("80.00")]
RISK_ANCHORS = [Decimal("0.75"), Decimal("1.00"), Decimal("1.25"), Decimal("1.50"), Decimal("1.75"), Decimal("2.00")]
GL_RATIOS = [2, 5, 10, 20, 30, 50]
STABLE_TIE_BREAK_RULE = "selected calibrated PD ascending, then frozen base_order ascending"
APPROVED_COUNT_RULE = "floor(final_evaluation_row_count * approval_rate_percent / 100)"
RISK_SELECTION_RULE = (
    "calculate cumulative mean selected calibrated PD at every individual ranked prefix and select the "
    "largest prefix not exceeding the common budget; retain zero approval with feasibility_flag=false if none"
)
README_START = "<!-- TASK11_DESIGN_FREEZE_START -->"
README_END = "<!-- TASK11_DESIGN_FREEZE_END -->"
OUTPUT_NAMES = {
    "analysis_contract.json",
    "approval_rate_grid.csv",
    "portfolio_risk_budget_grid.csv",
    "economic_scenario_contract.csv",
    "final_evaluation_output_schema.json",
    "input_hashes_before_after.csv",
    "validation_results.csv",
    "run_config.json",
    "task11_part3_report.md",
}
FUTURE_OUTPUT_NAMES = [
    "final_evaluation_predictions.parquet",
    "subgroup_model_metrics.csv",
    "subgroup_incremental_gains.csv",
    "fixed_approval_rate_points.csv",
    "fixed_approval_rate_curve.csv",
    "fixed_risk_budget_points.csv",
    "fixed_risk_budget_curve.csv",
    "economic_scenario_results.csv",
    "economic_incremental_value.csv",
]
PART2_OUTPUT_NAMES = set(part2.OUTPUT_NAMES)


class Part3Blocked(RuntimeError):
    """The protected foundation or frozen contract did not pass."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--verify-completed", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(value, encoding="utf-8")
    os.replace(temp, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temp, index=False, lineterminator="\n")
    os.replace(temp, path)


def assign_account_group(value: float) -> str | None:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value >= 3:
        return "3_plus"
    return None


def assign_history_group(value: float | None) -> str | None:
    grouped = 0.0 if value is None else float(value)
    if 0 <= grouped <= 0.5:
        return "0_to_0_5"
    if 0.5 < grouped <= 1:
        return "0_5_to_1"
    if 1 < grouped <= 2:
        return "1_to_2"
    if 2 < grouped <= 3:
        return "2_to_3"
    if 3 < grouped <= 4:
        return "3_to_4"
    if 4 < grouped <= 5:
        return "4_to_5"
    if grouped > 5:
        return "gt_5"
    return None


def assign_ad_richness_group(debit: int, deposit: int, tax: int) -> str:
    values = (debit, deposit, tax)
    if any(value not in (0, 1) for value in values):
        raise ValueError("AD module indicators must be binary pre-model values")
    return str(sum(values))


def decimal_grid(start: str, end: str, step: str) -> list[Decimal]:
    first, last, increment = Decimal(start), Decimal(end), Decimal(step)
    count = int((last - first) / increment) + 1
    return [first + increment * index for index in range(count)]


def approval_rate_grid() -> pd.DataFrame:
    values = decimal_grid("40.00", "80.00", "0.25")
    anchors = set(APPROVAL_ANCHORS)
    return pd.DataFrame([
        {
            "grid_order": order,
            "approval_rate_percent": format(value, ".2f"),
            "approval_rate_probability": format(value / Decimal(100), ".4f"),
            "is_external_display_point": value in anchors,
            "approved_count_rule": APPROVED_COUNT_RULE,
        }
        for order, value in enumerate(values, 1)
    ])


def portfolio_risk_budget_grid() -> pd.DataFrame:
    values = decimal_grid("0.75", "2.00", "0.01")
    anchors = set(RISK_ANCHORS)
    return pd.DataFrame([
        {
            "grid_order": order,
            "risk_budget_percent": format(value, ".2f"),
            "risk_budget_probability": format(value / Decimal(100), ".4f"),
            "is_external_display_point": value in anchors,
            "selection_rule": RISK_SELECTION_RULE,
        }
        for order, value in enumerate(values, 1)
    ])


def economic_scenario_contract() -> pd.DataFrame:
    roles = {
        2: "detailed_only",
        5: "detailed_only",
        10: "detailed_only",
        20: "lower_loss_sensitivity",
        30: "primary",
        50: "higher_loss_sensitivity",
    }
    rows = []
    for ratio in GL_RATIOS:
        threshold = Decimal(1) / (Decimal(1) + Decimal(ratio))
        rows.append({
            "scenario_id": f"normalized_gl_{ratio}",
            "normalized_G": "1",
            "normalized_L": str(ratio),
            "loss_to_gain_ratio": ratio,
            "expected_value_formula": "v = (1 - p) * G - p * L",
            "individual_pd_threshold_probability": format(threshold, ".18f"),
            "individual_pd_threshold_percent": format(threshold * Decimal(100), ".16f"),
            "approval_rule": "approve if selected calibrated p < p_star (strict inequality)",
            "realised_value_formula": "V_realised = approved_non_defaults * G - approved_defaults * L",
            "display_role": roles[ratio],
            "accounting_interpretation": "normalized scenario value; not observed or reconstructed Home Credit accounting profit",
        })
    return pd.DataFrame(rows)


def future_output_schema() -> dict[str, Any]:
    probability_columns = [f"calibrated_pd__{model}" for model in MODEL_ORDER]
    subgroup_metrics = [
        "group_dimension", "group_order", "group_label", "model_id", "selected_calibration_method",
        "sample_count", "positive_count", "negative_count", "observed_default_rate", "mean_calibrated_pd",
        "roc_auc", "average_precision", "log_loss", "brier_score", "metric_status", "missing_metric_reason",
    ]
    approval_fields = [
        "model_id", "selected_calibration_method", "approval_rate_percent", "approved_count", "rejected_count",
        "boundary_calibrated_pd_cutoff", "approved_pd_minimum", "approved_pd_maximum",
        "approved_portfolio_mean_calibrated_pd", "cumulative_expected_defaults", "realised_approved_defaults",
        "realised_approved_non_defaults", "realised_approved_default_rate", "stable_tie_break_rule",
    ]
    risk_fields = [
        "common_risk_budget_percent", "common_risk_budget_probability", "model_id", "selected_calibration_method",
        "maximum_feasible_approval_rate_percent", "approved_count", "rejected_count",
        "boundary_calibrated_pd_cutoff", "approved_portfolio_mean_calibrated_pd", "cumulative_expected_defaults",
        "realised_approved_defaults", "realised_approved_non_defaults", "realised_approved_default_rate",
        "budget_slack", "feasibility_flag", "stable_tie_break_rule",
    ]
    economic_fields = [
        "scenario_id", "normalized_G", "normalized_L", "loss_to_gain_ratio",
        "individual_pd_threshold_probability", "individual_pd_threshold_percent", "display_role", "model_id",
        "selected_calibration_method", "approved_count", "rejected_count", "approval_rate_percent",
        "approved_non_defaults", "approved_defaults", "approved_realised_default_rate",
        "expected_scenario_value_total", "realised_scenario_value_total",
        "realised_scenario_value_per_evaluation_case", "realised_scenario_value_per_approved_case",
    ]
    files = [
        {
            "file_name": "final_evaluation_predictions.parquet",
            "grain": "one row per frozen final-evaluation case_id",
            "required_columns": ["case_id", "base_order", "target", "account_group", "bureau_history_group", "ad_richness_group", *probability_columns],
            "key_columns": ["case_id"],
            "sort_order": ["base_order"],
            "purpose": "authoritative selected calibrated probabilities and pre-model subgroup labels for the later final evaluation",
        },
        {
            "file_name": "subgroup_model_metrics.csv",
            "grain": "one row per group_dimension x group_label x model_id",
            "required_columns": subgroup_metrics,
            "key_columns": ["group_dimension", "group_label", "model_id"],
            "sort_order": ["group_dimension", "group_order", "model_id"],
            "purpose": "future subgroup counts, outcomes, and probability metrics for all six models",
        },
        {
            "file_name": "subgroup_incremental_gains.csv",
            "grain": "one row per group_dimension x group_label x algorithm_family",
            "required_columns": [
                "group_dimension", "group_order", "group_label", "algorithm_family", "t_model_id", "t_plus_ad_model_id",
                "sample_count", "roc_auc_gain", "average_precision_gain", "log_loss_gain", "brier_gain",
                "mean_calibrated_pd_change", "gain_status", "missing_metric_reason",
            ],
            "key_columns": ["group_dimension", "group_label", "algorithm_family"],
            "sort_order": ["group_dimension", "group_order", "algorithm_family"],
            "purpose": "future within-family T versus T+AD gains with positive performance gain meaning improvement",
        },
        {
            "file_name": "fixed_approval_rate_points.csv",
            "grain": "one row per model_id x external approval-rate display point",
            "required_columns": approval_fields,
            "key_columns": ["model_id", "approval_rate_percent"],
            "sort_order": ["model_id", "approval_rate_percent"],
            "purpose": "five-point external table retaining predicted and realised approved-portfolio risk",
        },
        {
            "file_name": "fixed_approval_rate_curve.csv",
            "grain": "one row per model_id x detailed approval-rate grid point",
            "required_columns": approval_fields,
            "key_columns": ["model_id", "approval_rate_percent"],
            "sort_order": ["model_id", "approval_rate_percent"],
            "purpose": "authoritative exact XY and hover-data source for later fixed-approval static and interactive views",
        },
        {
            "file_name": "fixed_risk_budget_points.csv",
            "grain": "one row per model_id x external common-risk-budget display point",
            "required_columns": risk_fields,
            "key_columns": ["model_id", "common_risk_budget_percent"],
            "sort_order": ["model_id", "common_risk_budget_percent"],
            "purpose": "six-point external fixed-risk table based on individual-prefix inversion",
        },
        {
            "file_name": "fixed_risk_budget_curve.csv",
            "grain": "one row per model_id x detailed common-risk-budget grid point",
            "required_columns": risk_fields,
            "key_columns": ["model_id", "common_risk_budget_percent"],
            "sort_order": ["model_id", "common_risk_budget_percent"],
            "purpose": "authoritative exact XY and hover-data source for later fixed-risk views without smoothing",
        },
        {
            "file_name": "economic_scenario_results.csv",
            "grain": "one row per model_id x normalized G/L scenario",
            "required_columns": economic_fields,
            "key_columns": ["model_id", "scenario_id"],
            "sort_order": ["model_id", "loss_to_gain_ratio"],
            "purpose": "future predicted expected and target-based realised normalized scenario values for all models and scenarios",
        },
        {
            "file_name": "economic_incremental_value.csv",
            "grain": "one row per algorithm_family x normalized G/L scenario",
            "required_columns": [
                "scenario_id", "loss_to_gain_ratio", "display_role", "algorithm_family", "t_model_id", "t_plus_ad_model_id",
                "expected_scenario_value_T", "expected_scenario_value_T_plus_AD", "incremental_expected_value_from_AD",
                "realised_scenario_value_T", "realised_scenario_value_T_plus_AD", "incremental_value_from_AD",
            ],
            "key_columns": ["algorithm_family", "scenario_id"],
            "sort_order": ["algorithm_family", "loss_to_gain_ratio"],
            "purpose": "future within-family T versus T+AD normalized value differences using common scenarios and target outcomes",
        },
    ]
    return {
        "schema_version": VERSION,
        "status": "CONTRACT_ONLY_NO_FINAL_EVALUATION_VALUES",
        "future_output_files": files,
        "visualization_plan": {
            "static_readme_views": [PRIMARY_COMPARISON, SECONDARY_COMPARISON],
            "interactive_models": MODEL_ORDER,
            "approval_x_axis": "approval_rate_percent",
            "approval_selectable_y_metrics": ["realised_approved_default_rate", "approved_portfolio_mean_calibrated_pd"],
            "approval_default_y_metric": "realised_approved_default_rate",
            "approval_hover_fields": [
                "model_id", "approval_rate_percent", "approved_count", "boundary_calibrated_pd_cutoff",
                "approved_portfolio_mean_calibrated_pd", "cumulative_expected_defaults", "realised_approved_defaults",
                "realised_approved_default_rate",
            ],
            "risk_x_axis": "common_risk_budget_percent",
            "risk_y_axis": "maximum_feasible_approval_rate_percent",
            "risk_hover_fields": [
                "model_id", "common_risk_budget_percent", "maximum_feasible_approval_rate_percent", "approved_count",
                "approved_portfolio_mean_calibrated_pd", "boundary_calibrated_pd_cutoff", "realised_approved_defaults",
                "realised_approved_default_rate",
            ],
            "data_source_rule": "hover and plotted values come directly from saved detailed CSVs; no separately recomputed or rounded-only source",
            "smoothing": "none; a line may connect exact grid points",
            "part3_visualizations_produced": False,
        },
    }


def analysis_contract(created_at: str) -> dict[str, Any]:
    return {
        "contract_version": VERSION,
        "creation_timestamp_utc": created_at,
        "status": "FROZEN_CONTRACT_NO_FINAL_EVALUATION_CALCULATIONS",
        "frozen_foundation": {
            "membership_fingerprint": EXPECTED_FINGERPRINT,
            "population_counts": POPULATION_COUNTS,
            "model_order": MODEL_ORDER,
            "selected_calibration_methods": SELECTED_METHODS,
        },
        "subgroup_contract": {
            "assignment_boundary": PRE_MODEL_BOUNDARY,
            "dimensions_are_separate": True,
            "account": {
                "source_field": "t__observed_active_credit_count",
                "ordered_groups": ACCOUNT_GROUPS,
            },
            "bureau_history": {
                "construction": "earliest valid linked date on or before decision date across credit_bureau_b_1.contractdate_551D and credit_bureau_a_1.dateofcredstart_739D",
                "history_days_formula": "decision date - earliest valid linked date",
                "history_years_formula": "bureau_history_days / 365.25",
                "grouping_replacement": "bureau_history_years.fillna(0) directly before grouping",
                "ordered_groups": HISTORY_GROUPS,
            },
            "ad_richness": {
                "name": "observed_ad_module_count",
                "indicator_names": AD_INDICATORS,
                "formula": " + ".join(AD_INDICATORS),
                "ordered_groups": AD_GROUPS,
                "interpretation": "lower means less observed AD richness; higher means more observed AD richness",
                "exact_binary_patterns_role": "supporting Part 2 evidence only, not the primary final-evaluation grouping",
            },
            "prohibited_designs": PROHIBITED_SUBGROUP_DESIGNS,
        },
        "subgroup_metric_contract": {
            "grain": "group_dimension x group_label x model_id",
            "models": MODEL_ORDER,
            "metrics": [
                "sample_count", "positive_count", "negative_count", "observed_default_rate", "mean_calibrated_pd",
                "roc_auc", "average_precision", "log_loss", "brier_score",
            ],
            "probability_source": "selected calibrated probability only",
            "single_class_rule": "retain group and counts; set class-dependent metrics missing and record the factual reason",
            "gain_directions": {
                "roc_auc_gain": "roc_auc_T_plus_AD - roc_auc_T",
                "average_precision_gain": "average_precision_T_plus_AD - average_precision_T",
                "log_loss_gain": "log_loss_T - log_loss_T_plus_AD",
                "brier_gain": "brier_T - brier_T_plus_AD",
                "mean_calibrated_pd_change": "mean_pd_T_plus_AD - mean_pd_T; reassessment measure, not performance gain",
            },
            "algorithm_pairs": ALGORITHM_PAIRS,
            "primary_comparison": PRIMARY_COMPARISON,
            "secondary_t_plus_ad_complexity_comparison": SECONDARY_COMPARISON,
            "preserve_all_six_models": True,
        },
        "fixed_approval_contract": {
            "ranking_rule": STABLE_TIE_BREAK_RULE,
            "approved_count_rule": APPROVED_COUNT_RULE,
            "policy_meaning": "same approval rate per model, not a common individual PD cutoff",
            "external_display_points_percent": [int(value) for value in APPROVAL_ANCHORS],
            "detailed_grid": {"start_percent": "40.00", "end_percent": "80.00", "step_percentage_points": "0.25", "point_count": 161},
            "predicted_and_realised_risk_required": True,
            "mean_pd_does_not_substitute_for_realised_rate": True,
            "visualization": {
                "primary_models": PRIMARY_COMPARISON,
                "secondary_models": SECONDARY_COMPARISON,
                "interactive_models": MODEL_ORDER,
                "x_axis": "approval_rate_percent",
                "selectable_y_metrics": ["realised_approved_default_rate", "approved_portfolio_mean_calibrated_pd"],
                "default_y_metric": "realised_approved_default_rate",
                "authoritative_source": "fixed_approval_rate_curve.csv",
            },
        },
        "fixed_risk_contract": {
            "ranking_rule": STABLE_TIE_BREAK_RULE,
            "selection_rule": RISK_SELECTION_RULE,
            "constraint_type": "portfolio-average predicted-risk constraint, not an individual PD ceiling",
            "external_display_points_percent": [format(value, ".2f") for value in RISK_ANCHORS],
            "detailed_grid": {"start_percent": "0.75", "end_percent": "2.00", "step_percentage_points": "0.01", "point_count": 126},
            "zero_approval_rule": "retain row with feasibility_flag=false when no positive prefix is feasible",
            "visualization": {
                "primary_models": PRIMARY_COMPARISON,
                "secondary_models": SECONDARY_COMPARISON,
                "interactive_models": MODEL_ORDER,
                "x_axis": "common portfolio mean calibrated-PD budget",
                "y_axis": "maximum feasible approval rate",
                "smoothing": "none",
                "authoritative_source": "fixed_risk_budget_curve.csv",
            },
        },
        "normalized_economic_contract": {
            "scope": "normalized common G/L scenarios; not Home Credit actual accounting profit",
            "excluded_primary_contract_fields": [
                "credamount_770A", "annuity_780A", "eir_270L", "price_1097A", "monthsannuity_845L", "numinstls_657L",
            ],
            "normalized_G": 1,
            "loss_to_gain_ratios": GL_RATIOS,
            "expected_value_formula": "v = (1 - p) * G - p * L",
            "threshold_formula": "p_star = G / (G + L) = 1 / (1 + loss_to_gain_ratio)",
            "approval_rule": "approve if p < p_star",
            "rejected_value": 0,
            "realised_value_formula": "V_realised = approved_non_defaults * G - approved_defaults * L",
            "predicted_expected_and_target_realised_values_are_separate": True,
            "same_target_and_scenario_across_models": True,
            "incremental_realised_value_formula": "realised_value_T_plus_AD - realised_value_T",
            "display_roles": {"30": "primary", "20": "lower_loss_sensitivity", "50": "higher_loss_sensitivity", "2,5,10": "detailed_only"},
            "ad_cost_assumption": None,
        },
        "final_evaluation_data_boundary": {
            "project_wide_metadata_used_for_integrity_verification_only": True,
            "final_evaluation_feature_rows_loaded_for_analysis": False,
            "final_evaluation_targets_loaded_or_inspected": False,
            "final_evaluation_features_transformed": False,
            "final_evaluation_cases_scored": False,
            "final_evaluation_predictions_generated": False,
            "final_evaluation_subgroups_assigned": False,
            "final_evaluation_metrics_or_policy_or_economic_results_calculated": False,
            "visualizations_generated": False,
            "freeze_statement": "contract frozen without final-evaluation calculations",
        },
    }


def output_path_is_protected(data_root: Path, output_dir: Path) -> bool:
    protected_roots = [
        data_root / relative for relative in (
            "audits/task08", "audits/task08_followup", "audits/task09", "audits/task10",
            "audits/task11/part1", "audits/task11/part2", "interim/task08", "interim/task08_followup",
            "interim/task09", "interim/task10", "models/task09", "models/task10",
        )
    ]
    resolved = output_dir.resolve()
    return any(resolved == root.resolve() or root.resolve() in resolved.parents for root in protected_roots)


def protected_paths(data_root: Path, repo_root: Path) -> list[tuple[str, str, Path]]:
    records = list(part2.protected_paths(data_root, repo_root))
    records.extend([
        ("task11_part2_script", "Task 11 Part 2", repo_root / "scripts/task11_part2_design_audit.py"),
        ("task11_part2_test", "Task 11 Part 2", repo_root / "tests/test_task11_part2.py"),
    ])
    part2_dir = data_root / "audits/task11/part2"
    if not part2_dir.is_dir():
        raise Part3Blocked(f"Missing Task 11 Part 2 directory: {part2_dir}")
    if {path.name for path in part2_dir.iterdir() if path.is_file()} != PART2_OUTPUT_NAMES:
        raise Part3Blocked("Task 11 Part 2 output inventory mismatch")
    for path in sorted(part2_dir.iterdir()):
        if path.is_file():
            records.append((f"task11_part2_output::{path.name}", "Task 11 Part 2", path))
    unique: dict[str, tuple[str, str, Path]] = {str(record[2].resolve()): record for record in records}
    missing = [str(record[2]) for record in unique.values() if not record[2].is_file()]
    if missing:
        raise Part3Blocked(f"Protected artifacts missing: {missing}")
    return [unique[path] for path in sorted(unique)]


def hash_paths(records: Iterable[tuple[str, str, Path]]) -> dict[str, dict[str, str]]:
    result = {}
    for artifact, component, path in records:
        resolved = str(path.resolve())
        result[resolved] = {
            "protected_path": resolved,
            "artifact": artifact,
            "component_task": component,
            "sha256": sha256_file(path),
        }
    return result


def verify_prior_evidence(data_root: Path) -> dict[str, Any]:
    part1_dir = data_root / "audits/task11/part1"
    part2_dir = data_root / "audits/task11/part2"
    if not part1_dir.is_dir() or {p.name for p in part1_dir.iterdir() if p.is_file()} != part1.OUTPUT_NAMES:
        raise Part3Blocked("Task 11 Part 1 exact output inventory is unavailable")
    part1_validation = pd.read_csv(part1_dir / "validation_results.csv")
    part1_foundation = pd.read_csv(part1_dir / "foundation_verification.csv")
    if not part1_validation.status.eq("PASS").all() or not part1_foundation.status.eq("PASS").all():
        raise Part3Blocked("Task 11 Part 1 does not have all validations passing")
    if "Status: **COMPLETE**" not in (part1_dir / "task11_part1_report.md").read_text(encoding="utf-8"):
        raise Part3Blocked("Task 11 Part 1 report is not COMPLETE")
    part2_validation = pd.read_csv(part2_dir / "validation_results.csv")
    if not part2_validation.status.eq("PASS").all():
        raise Part3Blocked("Task 11 Part 2 does not have all validations passing")
    if "Status: **COMPLETE**" not in (part2_dir / "task11_part2_report.md").read_text(encoding="utf-8"):
        raise Part3Blocked("Task 11 Part 2 report is not COMPLETE")
    part2_config = json.loads((part2_dir / "run_config.json").read_text(encoding="utf-8"))
    if part2_config.get("final_evaluation_rows_or_results_included_in_calculations") is not False:
        raise Part3Blocked("Task 11 Part 2 boundary evidence is inconsistent")
    part2.verify_part1(data_root)
    foundation = part1.verify_foundation(data_root)
    if not all(row["status"] == "PASS" for row in foundation.rows):
        raise Part3Blocked("Task 8-10 foundation verification failed")
    saved = task10.verify_saved_run(data_root, "first_full", "first_full_v3", MODEL_ORDER)
    if saved.get("status") != "PASS" or saved.get("final_evaluation_status") != "NOT_PREDICTED":
        raise Part3Blocked("Task 10 saved-run verification failed or final evaluation was predicted")
    observed_methods = {item["model_id"]: item["selected_method"] for item in saved["models"]}
    if observed_methods != SELECTED_METHODS:
        raise Part3Blocked(f"Task 10 selected-method contradiction: {observed_methods}")
    return {
        "part1_validation_rows": len(part1_validation),
        "part1_foundation_rows": len(part1_foundation),
        "part2_validation_rows": len(part2_validation),
        "task8_10_foundation_checks": len(foundation.rows),
        "task10_saved_run": saved,
    }


def run_test(command: list[str], repo_root: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=repo_root, check=False, text=True, capture_output=True)
    output = "\n".join(value for value in (result.stdout.strip(), result.stderr.strip()) if value).strip()
    if result.returncode != 0:
        raise Part3Blocked(f"Test command failed ({' '.join(command)}):\n{output}")
    return {"command": command, "return_code": result.returncode, "result": output}


def git_state(repo_root: Path) -> tuple[str, list[str]]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, text=True, capture_output=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=repo_root, check=True, text=True, capture_output=True).stdout.splitlines()
    return commit, status


def readme_block(output_dir: Path) -> str:
    return f"""{README_START}
## Task 11 analysis design freeze

Task 11 design freeze is complete through Part 3. Account, bureau-history, and AD-richness dimensions are assigned from pre-model values; AD richness is the observed module count from 0 through 3.

Fixed-approval display points are 40%, 50%, 60%, 70%, and 80%, with a later detailed 40%–80% curve. Fixed-risk display points are 0.75%, 1.00%, 1.25%, 1.50%, 1.75%, and 2.00%, with a later detailed 0.75%–2.00% curve. Normalized G/L uses ratio 30 as primary and ratios 20 and 50 as sensitivities.

No final-evaluation result has been calculated. The Part 3 audit and contract outputs are stored privately at `{output_dir}`.
{README_END}"""


def planned_readme(before: str, block: str) -> str:
    start_count, end_count = before.count(README_START), before.count(README_END)
    if start_count != end_count or start_count > 1:
        raise Part3Blocked("README has an inconsistent Task 11 bounded section")
    if start_count == 0:
        return before.rstrip() + "\n\n" + block + "\n"
    start = before.index(README_START)
    end = before.index(README_END, start) + len(README_END)
    return before[:start] + block + before[end:]


def outside_readme_section(value: str) -> str:
    if README_START not in value and README_END not in value:
        return value.rstrip()
    if value.count(README_START) != 1 or value.count(README_END) != 1:
        raise Part3Blocked("README Task 11 markers are inconsistent")
    start = value.index(README_START)
    end = value.index(README_END, start) + len(README_END)
    return (value[:start] + value[end:]).rstrip()


def validation_row(validation_id: str, component: str, description: str, observed: Any,
                   expected: Any, passed: bool, notes: str = "") -> dict[str, Any]:
    def render(value: Any) -> str:
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value if not isinstance(value, set) else sorted(value), ensure_ascii=False, sort_keys=True)
        return str(value)
    return {
        "validation_id": validation_id,
        "component": component,
        "description": description,
        "observed_result": render(observed),
        "expected_result": render(expected),
        "status": "PASS" if passed else "FAIL",
        "notes": notes,
    }


def validate_contract(contract: dict[str, Any], approval: pd.DataFrame, risk: pd.DataFrame,
                      economics: pd.DataFrame, schema: dict[str, Any], evidence: dict[str, Any],
                      hash_frame: pd.DataFrame, readme_before: str, readme_after: str,
                      data_root: Path, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    add = rows.append
    approval_values = [Decimal(value) for value in approval.approval_rate_percent]
    risk_values = [Decimal(value) for value in risk.risk_budget_percent]
    approval_marked = [Decimal(value) for value in approval.loc[approval.is_external_display_point, "approval_rate_percent"]]
    risk_marked = [Decimal(value) for value in risk.loc[risk.is_external_display_point, "risk_budget_percent"]]
    schema_by_name = {item["file_name"]: item for item in schema["future_output_files"]}
    required_schema_keys = {"grain", "required_columns", "key_columns", "sort_order", "purpose"}
    boundary = contract["final_evaluation_data_boundary"]
    add(validation_row("task11_part1_complete", "foundation", "Part 1 inventory and evidence pass", evidence["part1_validation_rows"], ">0 and all PASS", evidence["part1_validation_rows"] > 0))
    add(validation_row("task11_part2_complete", "foundation", "Part 2 inventory and evidence pass", evidence["part2_validation_rows"], ">0 and all PASS", evidence["part2_validation_rows"] > 0))
    add(validation_row("task8_10_read_only_verification", "foundation", "Existing Task 8-10 foundation interface passes", evidence["task8_10_foundation_checks"], 27, evidence["task8_10_foundation_checks"] == 27))
    add(validation_row("task10_saved_run_verification", "foundation", "Task 10 saved run remains verified and final evaluation not predicted", {"status": evidence["task10_saved_run"]["status"], "final_evaluation_status": evidence["task10_saved_run"]["final_evaluation_status"]}, {"status": "PASS", "final_evaluation_status": "NOT_PREDICTED"}, evidence["task10_saved_run"]["status"] == "PASS" and evidence["task10_saved_run"]["final_evaluation_status"] == "NOT_PREDICTED"))
    add(validation_row("protected_hashes", "foundation", "Every protected hash is unchanged", int(hash_frame.unchanged_flag.sum()), len(hash_frame), bool(hash_frame.unchanged_flag.all())))
    add(validation_row("model_order", "models", "Exact six-model order", contract["frozen_foundation"]["model_order"], MODEL_ORDER, contract["frozen_foundation"]["model_order"] == MODEL_ORDER))
    add(validation_row("selected_methods", "models", "Exact selected calibration methods", contract["frozen_foundation"]["selected_calibration_methods"], SELECTED_METHODS, contract["frozen_foundation"]["selected_calibration_methods"] == SELECTED_METHODS))
    add(validation_row("subgroup_account", "subgroups", "Exact account source, labels, and rules", contract["subgroup_contract"]["account"], {"source_field": "t__observed_active_credit_count", "ordered_groups": ACCOUNT_GROUPS}, contract["subgroup_contract"]["account"] == {"source_field": "t__observed_active_credit_count", "ordered_groups": ACCOUNT_GROUPS}))
    add(validation_row("subgroup_history", "subgroups", "Exact two-source history grouping", contract["subgroup_contract"]["bureau_history"]["ordered_groups"], HISTORY_GROUPS, contract["subgroup_contract"]["bureau_history"]["ordered_groups"] == HISTORY_GROUPS))
    add(validation_row("subgroup_ad_richness", "subgroups", "Exact pre-model AD indicator sum and labels", {"indicators": contract["subgroup_contract"]["ad_richness"]["indicator_names"], "groups": contract["subgroup_contract"]["ad_richness"]["ordered_groups"]}, {"indicators": AD_INDICATORS, "groups": AD_GROUPS}, contract["subgroup_contract"]["ad_richness"]["indicator_names"] == AD_INDICATORS and contract["subgroup_contract"]["ad_richness"]["ordered_groups"] == AD_GROUPS))
    add(validation_row("pre_model_boundary", "subgroups", "All subgroup identities use pre-model values", contract["subgroup_contract"]["assignment_boundary"], PRE_MODEL_BOUNDARY, contract["subgroup_contract"]["assignment_boundary"] == PRE_MODEL_BOUNDARY))
    add(validation_row("prohibited_subgroup_designs", "subgroups", "No forbidden subgroup design is active", contract["subgroup_contract"]["dimensions_are_separate"], True, contract["subgroup_contract"]["dimensions_are_separate"] is True and contract["subgroup_contract"]["prohibited_designs"] == PROHIBITED_SUBGROUP_DESIGNS))
    add(validation_row("comparison_sets", "models", "Primary and secondary comparison sets", {"primary": contract["subgroup_metric_contract"]["primary_comparison"], "secondary": contract["subgroup_metric_contract"]["secondary_t_plus_ad_complexity_comparison"]}, {"primary": PRIMARY_COMPARISON, "secondary": SECONDARY_COMPARISON}, contract["subgroup_metric_contract"]["primary_comparison"] == PRIMARY_COMPARISON and contract["subgroup_metric_contract"]["secondary_t_plus_ad_complexity_comparison"] == SECONDARY_COMPARISON))
    approval_shape = len(approval_values) == 161 and approval_values[0] == Decimal("40.00") and approval_values[-1] == Decimal("80.00") and all(right - left == Decimal("0.25") for left, right in zip(approval_values, approval_values[1:])) and len(set(approval_values)) == 161
    add(validation_row("approval_grid", "policy", "Approval grid is exact, unique, and increasing", {"rows": len(approval_values), "start": str(approval_values[0]), "end": str(approval_values[-1])}, {"rows": 161, "start": "40.00", "end": "80.00", "step": "0.25"}, approval_shape))
    add(validation_row("approval_anchors", "policy", "Exactly five approval points are externally marked", [str(value) for value in approval_marked], [str(value) for value in APPROVAL_ANCHORS], approval_marked == APPROVAL_ANCHORS))
    approval_conversion = all(Decimal(row.approval_rate_probability) == Decimal(row.approval_rate_percent) / Decimal(100) for row in approval.itertuples())
    add(validation_row("approval_conversion", "policy", "Approval percent-to-probability conversion exact", approval_conversion, True, approval_conversion))
    risk_shape = len(risk_values) == 126 and risk_values[0] == Decimal("0.75") and risk_values[-1] == Decimal("2.00") and all(right - left == Decimal("0.01") for left, right in zip(risk_values, risk_values[1:])) and len(set(risk_values)) == 126
    add(validation_row("risk_grid", "policy", "Risk grid is exact, unique, and increasing", {"rows": len(risk_values), "start": str(risk_values[0]), "end": str(risk_values[-1])}, {"rows": 126, "start": "0.75", "end": "2.00", "step": "0.01"}, risk_shape))
    add(validation_row("risk_anchors", "policy", "Exactly six risk points are externally marked", [str(value) for value in risk_marked], [str(value) for value in RISK_ANCHORS], risk_marked == RISK_ANCHORS))
    risk_conversion = all(Decimal(row.risk_budget_probability) == Decimal(row.risk_budget_percent) / Decimal(100) for row in risk.itertuples())
    add(validation_row("risk_conversion", "policy", "Risk percent-to-probability conversion exact", risk_conversion, True, risk_conversion))
    add(validation_row("stable_tie_break", "policy", "Both policies use the deterministic tie rule", {"approval": contract["fixed_approval_contract"]["ranking_rule"], "risk": contract["fixed_risk_contract"]["ranking_rule"]}, STABLE_TIE_BREAK_RULE, contract["fixed_approval_contract"]["ranking_rule"] == STABLE_TIE_BREAK_RULE and contract["fixed_risk_contract"]["ranking_rule"] == STABLE_TIE_BREAK_RULE))
    add(validation_row("individual_prefix_inversion", "policy", "Risk inversion requires every individual prefix", contract["fixed_risk_contract"]["selection_rule"], RISK_SELECTION_RULE, contract["fixed_risk_contract"]["selection_rule"] == RISK_SELECTION_RULE))
    add(validation_row("economic_ratios", "economics", "Exactly six normalized G/L ratios", economics.loss_to_gain_ratio.tolist(), GL_RATIOS, economics.loss_to_gain_ratio.tolist() == GL_RATIOS))
    threshold_ok = all(abs(Decimal(row.individual_pd_threshold_probability) - (Decimal(1) / (Decimal(1) + Decimal(row.loss_to_gain_ratio)))) <= Decimal("5e-19") for row in economics.itertuples())
    add(validation_row("economic_thresholds", "economics", "Every p_star satisfies 1/(1+ratio)", threshold_ok, True, threshold_ok))
    observed_roles = dict(zip(economics.loss_to_gain_ratio, economics.display_role))
    roles_ok = observed_roles[30] == "primary" and observed_roles[20] == "lower_loss_sensitivity" and observed_roles[50] == "higher_loss_sensitivity" and all(observed_roles[value] == "detailed_only" for value in (2, 5, 10))
    add(validation_row("economic_display_roles", "economics", "Ratio 30 primary and 20/50 sensitivities", observed_roles, {20: "lower_loss_sensitivity", 30: "primary", 50: "higher_loss_sensitivity"}, roles_ok))
    add(validation_row("economic_value_separation", "economics", "Predicted expected and target-based realised values are distinct", contract["normalized_economic_contract"]["predicted_expected_and_target_realised_values_are_separate"], True, contract["normalized_economic_contract"]["predicted_expected_and_target_realised_values_are_separate"] is True))
    add(validation_row("future_output_inventory", "schemas", "Exactly nine future final-evaluation result files", list(schema_by_name), FUTURE_OUTPUT_NAMES, list(schema_by_name) == FUTURE_OUTPUT_NAMES))
    schema_shape_ok = all(required_schema_keys.issubset(item) and item["required_columns"] and item["key_columns"] and item["sort_order"] for item in schema["future_output_files"])
    add(validation_row("future_output_schema_fields", "schemas", "Every future file has grain, columns, keys, sort, and purpose", schema_shape_ok, True, schema_shape_ok))
    boundary_ok = boundary["project_wide_metadata_used_for_integrity_verification_only"] is True and all(value is False for key, value in boundary.items() if key not in {"project_wide_metadata_used_for_integrity_verification_only", "freeze_statement"})
    add(validation_row("final_evaluation_boundary", "boundary", "No final-evaluation calculation, scoring, target analysis, or visualization", boundary_ok, True, boundary_ok))
    add(validation_row("part3_inventory", "outputs", "Planned Part 3 inventory is exactly nine contract/evidence files", sorted(OUTPUT_NAMES), sorted(OUTPUT_NAMES), len(OUTPUT_NAMES) == 9))
    forbidden_suffixes = {".parquet", ".png", ".pdf", ".html", ".ipynb", ".pptx"}
    no_forbidden = not any(Path(name).suffix.lower() in forbidden_suffixes for name in OUTPUT_NAMES)
    add(validation_row("no_result_or_visual_output", "outputs", "Part 3 inventory contains no prediction/result/visualization artifact", no_forbidden, True, no_forbidden))
    add(validation_row("output_boundary", "outputs", "Part 3 output is outside every protected earlier root", str(output_dir), "unprotected Task 11 Part 3 directory", not output_path_is_protected(data_root, output_dir)))
    readme_ok = outside_readme_section(readme_before) == outside_readme_section(readme_after) and readme_after.count(README_START) == 1 and readme_after.count(README_END) == 1
    add(validation_row("readme_bounded_diff", "documentation", "README change is limited to one authorized Task 11 section", readme_ok, True, readme_ok))
    frame = pd.DataFrame(rows)
    if frame.status.eq("FAIL").any():
        raise Part3Blocked(f"Contract validation failed: {frame.loc[frame.status.eq('FAIL')].to_dict('records')}")
    return frame


def verify_completed(data_root: Path, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    if not output_dir.is_dir():
        raise Part3Blocked(f"Completed Part 3 output does not exist: {output_dir}")
    inventory = {path.name for path in output_dir.iterdir() if path.is_file()}
    if inventory != OUTPUT_NAMES:
        raise Part3Blocked(f"Completed Part 3 inventory mismatch: {sorted(inventory)}")
    recorded = pd.read_csv(output_dir / "input_hashes_before_after.csv")
    if not recorded.unchanged_flag.astype(bool).all():
        raise Part3Blocked("Recorded Part 3 protected-hash evidence contains a failure")
    current = {str(path.resolve()): sha256_file(path) for _, _, path in protected_paths(data_root, repo_root)}
    if set(recorded.protected_path) != set(current):
        raise Part3Blocked("Current protected path inventory differs from the Part 3 record")
    changed = [row.protected_path for row in recorded.itertuples() if current[row.protected_path] != row.sha256_before]
    if changed:
        raise Part3Blocked(f"Protected hashes changed after Part 3: {changed}")
    validations = pd.read_csv(output_dir / "validation_results.csv")
    if not validations.status.eq("PASS").all():
        raise Part3Blocked("Completed Part 3 validation evidence is not all PASS")
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    if readme.count(README_START) != 1 or readme.count(README_END) != 1:
        raise Part3Blocked("README Part 3 bounded section is missing or duplicated")
    return {
        "status": "PASS",
        "mode": "READ_ONLY_COMPLETED_RUN_VERIFICATION",
        "protected_files_rehashed": len(current),
        "protected_hashes_match_recorded_before": True,
        "output_inventory": sorted(inventory),
        "validation_rows": len(validations),
        "writes_performed": False,
    }


def run(data_root: Path, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    data_root, output_dir, repo_root = data_root.resolve(), output_dir.resolve(), repo_root.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Part 3 directory: {output_dir}")
    if output_path_is_protected(data_root, output_dir):
        raise Part3Blocked(f"Output path is protected: {output_dir}")
    readme_path = repo_root / "README.md"
    if not readme_path.is_file():
        raise Part3Blocked(f"README missing: {readme_path}")
    records = protected_paths(data_root, repo_root)
    before = hash_paths(records)
    evidence = verify_prior_evidence(data_root)
    created_at = now_utc()
    contract = analysis_contract(created_at)
    approval = approval_rate_grid()
    risk = portfolio_risk_budget_grid()
    economics = economic_scenario_contract()
    schema = future_output_schema()
    readme_before = readme_path.read_text(encoding="utf-8")
    focused_command = [sys.executable, "-m", "pytest", "-q", "tests/test_task11_part3.py"]
    full_command = [sys.executable, "-m", "pytest", "-q"]
    focused_test = run_test(focused_command, repo_root)
    full_test = run_test(full_command, repo_root)
    if readme_path.read_text(encoding="utf-8") != readme_before:
        raise Part3Blocked("README changed while the required tests were running")
    readme_after = planned_readme(readme_before, readme_block(output_dir))
    readme_changed = False
    temp: Path | None = None
    try:
        atomic_text(readme_path, readme_after)
        readme_changed = True
        after = hash_paths(records)
        hash_frame = pd.DataFrame([
            {
                "protected_path": path,
                "artifact": item["artifact"],
                "component_task": item["component_task"],
                "sha256_before": item["sha256"],
                "sha256_after": after[path]["sha256"],
                "unchanged_flag": item["sha256"] == after[path]["sha256"],
            }
            for path, item in before.items()
        ])
        if not hash_frame.unchanged_flag.all():
            raise Part3Blocked(f"Protected hashes changed: {hash_frame.loc[~hash_frame.unchanged_flag, 'protected_path'].tolist()}")
        validations = validate_contract(
            contract, approval, risk, economics, schema, evidence, hash_frame, readme_before, readme_after,
            data_root, output_dir,
        )
        git_commit, current_status = git_state(repo_root)
        part3_paths = {"?? scripts/freeze_task11_analysis_contract.py", "?? tests/test_task11_part3.py"}
        user_status = [line for line in current_status if line not in part3_paths]
        prompt_path = Path("/Users/haoguannan/Downloads/task11_part3_final_evaluation_contract_prompt.md")
        source_paths_and_hashes = {
            "authoritative_task_prompt": {"path": str(prompt_path), "sha256": sha256_file(prompt_path)},
            "task11_part1_run_config": {"path": str(data_root / "audits/task11/part1/run_config.json"), "sha256": sha256_file(data_root / "audits/task11/part1/run_config.json")},
            "task11_part2_run_config": {"path": str(data_root / "audits/task11/part2/run_config.json"), "sha256": sha256_file(data_root / "audits/task11/part2/run_config.json")},
            "task10_calibration_registry": {"path": str(data_root / "audits/task10/first_full_v3/calibration_registry.json"), "sha256": sha256_file(data_root / "audits/task10/first_full_v3/calibration_registry.json")},
        }
        formal_command = [
            sys.executable, "-u", "scripts/freeze_task11_analysis_contract.py", "--data-root", str(data_root),
            "--output-dir", str(output_dir),
        ]
        post_run_command = [
            sys.executable, "-u", "scripts/freeze_task11_analysis_contract.py", "--data-root", str(data_root),
            "--output-dir", str(output_dir), "--verify-completed",
        ]
        config = {
            "task_name": TASK_NAME,
            "part": 3,
            "version": VERSION,
            "timestamp_utc": created_at,
            "repository_root": str(repo_root),
            "data_root": str(data_root),
            "output_directory": str(output_dir),
            "python_executable": sys.executable,
            "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "platform": platform.platform()},
            "git_commit": git_commit,
            "pre_existing_user_worktree_status": user_status,
            "part3_new_files_visible_during_run": sorted(part3_paths.intersection(current_status)),
            "source_paths_and_hashes": source_paths_and_hashes,
            "protected_file_count": len(hash_frame),
            "foundation_verification": evidence,
            "commands": {
                "focused_tests_executed_by_freezer": focused_test,
                "full_safe_suite_executed_by_freezer": full_test,
                "formal_freeze_invocation": formal_command,
                "read_only_post_run_verification": post_run_command,
            },
            "grids": {
                "approval": {"start_percent": "40.00", "end_percent": "80.00", "step_percentage_points": "0.25", "count": 161, "anchors_percent": [format(value, ".2f") for value in APPROVAL_ANCHORS]},
                "risk": {"start_percent": "0.75", "end_percent": "2.00", "step_percentage_points": "0.01", "count": 126, "anchors_percent": [format(value, ".2f") for value in RISK_ANCHORS]},
            },
            "formulas": {
                "approval_count": APPROVED_COUNT_RULE,
                "risk_selection": RISK_SELECTION_RULE,
                "expected_value": "v = (1 - p) * G - p * L",
                "pd_threshold": "p_star = 1 / (1 + loss_to_gain_ratio)",
                "realised_value": "approved_non_defaults * G - approved_defaults * L",
            },
            "readme_update": {
                "start_marker": README_START,
                "end_marker": README_END,
                "sha256_before": sha256_text(readme_before),
                "sha256_after": sha256_text(readme_after),
                "outside_bounded_section_unchanged": outside_readme_section(readme_before) == outside_readme_section(readme_after),
            },
            "boundary_flags": contract["final_evaluation_data_boundary"],
            "random_process_used": False,
        }
        report = f"""# Task 11 Part 3 — final-evaluation analysis contract freeze

Status: **COMPLETE**

## Protected foundation

Task 11 Parts 1–2 and the existing Task 8–10 read-only verification interfaces passed. Task 10 still reports final evaluation as `NOT_PREDICTED`. All {len(hash_frame)} protected files had identical SHA-256 hashes before and after Part 3.

## Frozen owner decisions

Subgroups use separate pre-model account-count, two-source bureau-history, and observed AD-module-count dimensions. Account labels are 0, 1, 2, and 3_plus; history labels are 0_to_0_5, 0_5_to_1, 1_to_2, 2_to_3, 3_to_4, 4_to_5, and gt_5; AD richness is the sum of the three frozen numeric-content indicators with labels 0–3. No binary history threshold, missing-history group, cross-product, or composite index is active.

The primary model comparison is LightGBM T versus T+AD. The secondary T+AD complexity comparison is Logit, LightGBM, and MLP; all six model outputs remain required later.

Approval display anchors are 40%, 50%, 60%, 70%, and 80%; the detailed grid has 161 exact points from 40.00% through 80.00% in 0.25-point steps. Risk display anchors are 0.75%, 1.00%, 1.25%, 1.50%, 1.75%, and 2.00%; the detailed grid has 126 exact points from 0.75% through 2.00% in 0.01-point steps and requires individual-prefix inversion.

Normalized G/L ratios are 2, 5, 10, 20, 30, and 50. Ratio 30 is primary, 20 is the lower-loss sensitivity, and 50 is the higher-loss sensitivity. These scenarios are not Home Credit accounting-profit estimates.

## Future result contract, validation, and tests

`final_evaluation_output_schema.json` freezes nine later result files, their grains, required columns, keys, sort order, and purposes. It also freezes static and interactive views without producing them now.

All {len(validations)} substantive validations passed. Focused tests executed by the freezer: `{focused_test['result']}`. Full safe suite executed by the freezer: `{full_test['result']}`.

The freezer command was:

```text
{' '.join(formal_command)}
```

The independent read-only post-run command is:

```text
{' '.join(post_run_command)}
```

README was changed only between `{README_START}` and `{README_END}`. No final-evaluation row was used for analysis; no final-evaluation target was loaded or inspected; and no final-evaluation prediction, metric, subgroup outcome, policy outcome, economic result, or visualization was produced.
"""
        temp = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
        temp.mkdir(parents=True, exist_ok=False)
        atomic_json(temp / "analysis_contract.json", contract)
        atomic_csv(temp / "approval_rate_grid.csv", approval)
        atomic_csv(temp / "portfolio_risk_budget_grid.csv", risk)
        atomic_csv(temp / "economic_scenario_contract.csv", economics)
        atomic_json(temp / "final_evaluation_output_schema.json", schema)
        atomic_csv(temp / "input_hashes_before_after.csv", hash_frame)
        atomic_csv(temp / "validation_results.csv", validations)
        atomic_json(temp / "run_config.json", config)
        atomic_text(temp / "task11_part3_report.md", report)
        produced = {path.name for path in temp.iterdir() if path.is_file()}
        if produced != OUTPUT_NAMES:
            raise Part3Blocked(f"Part 3 output inventory mismatch: {sorted(produced)}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, output_dir)
        temp = None
        return {
            "status": "COMPLETE",
            "output_directory": str(output_dir),
            "protected_files": len(hash_frame),
            "validation_rows": len(validations),
            "focused_test_result": focused_test["result"],
            "full_test_result": full_test["result"],
            "outputs": sorted(produced),
            "final_evaluation_calculations": False,
        }
    except Exception:
        if readme_changed and readme_path.read_text(encoding="utf-8") == readme_after:
            atomic_text(readme_path, readme_before)
        if temp is not None and temp.is_dir():
            shutil.rmtree(temp)
        raise


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root, output_dir = args.data_root.resolve(), args.output_dir.resolve()
    result = verify_completed(data_root, output_dir, repo_root) if args.verify_completed else run(data_root, output_dir, repo_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
