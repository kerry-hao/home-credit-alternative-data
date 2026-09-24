from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import freeze_task11_analysis_contract as MODULE  # noqa: E402


def test_account_group_contract_and_assignment_are_exact() -> None:
    assert MODULE.ACCOUNT_GROUPS == [
        {"order": 1, "label": "0", "rule": "value == 0"},
        {"order": 2, "label": "1", "rule": "value == 1"},
        {"order": 3, "label": "2", "rule": "value == 2"},
        {"order": 4, "label": "3_plus", "rule": "value >= 3"},
    ]
    assert [MODULE.assign_account_group(value) for value in (0, 1, 2, 3, 9)] == ["0", "1", "2", "3_plus", "3_plus"]


def test_history_labels_boundaries_and_missing_to_zero_are_exact() -> None:
    assert [item["label"] for item in MODULE.HISTORY_GROUPS] == [
        "0_to_0_5", "0_5_to_1", "1_to_2", "2_to_3", "3_to_4", "4_to_5", "gt_5",
    ]
    values = [None, 0, 0.5, Decimal("0.5001"), 1, Decimal("1.0001"), 2, Decimal("2.0001"), 3,
              Decimal("3.0001"), 4, Decimal("4.0001"), 5, Decimal("5.0001")]
    expected = ["0_to_0_5", "0_to_0_5", "0_to_0_5", "0_5_to_1", "0_5_to_1", "1_to_2",
                "1_to_2", "2_to_3", "2_to_3", "3_to_4", "3_to_4", "4_to_5", "4_to_5", "gt_5"]
    assert [MODULE.assign_history_group(value) for value in values] == expected


def test_ad_richness_uses_only_exact_pre_model_indicators_and_labels() -> None:
    assert MODULE.AD_INDICATORS == [
        "ad__debit__has_numeric_content",
        "ad__deposit__has_numeric_content",
        "ad__tax__has_numeric_content",
    ]
    assert [item["label"] for item in MODULE.AD_GROUPS] == ["0", "1", "2", "3"]
    assert MODULE.assign_ad_richness_group(0, 0, 0) == "0"
    assert MODULE.assign_ad_richness_group(1, 0, 1) == "2"
    assert MODULE.assign_ad_richness_group(1, 1, 1) == "3"
    with pytest.raises(ValueError):
        MODULE.assign_ad_richness_group(1, 2, 0)


def test_pre_model_boundary_and_forbidden_subgroup_designs_are_explicit() -> None:
    contract = MODULE.analysis_contract("fixed")
    subgroup = contract["subgroup_contract"]
    assert subgroup["assignment_boundary"] == MODULE.PRE_MODEL_BOUNDARY
    assert subgroup["dimensions_are_separate"] is True
    assert "binary history-thin threshold" in subgroup["prohibited_designs"]
    assert "account-by-history cross-product" in subgroup["prohibited_designs"]
    assert "PCA or composite thickness index" in subgroup["prohibited_designs"]


def test_models_methods_pairs_and_comparison_views_are_exact() -> None:
    assert MODULE.MODEL_ORDER == [
        "logit_T", "logit_T_plus_AD", "lightgbm_T", "lightgbm_T_plus_AD", "mlp_T", "mlp_T_plus_AD",
    ]
    assert MODULE.SELECTED_METHODS == {
        "logit_T": "identity", "logit_T_plus_AD": "identity", "lightgbm_T": "identity",
        "lightgbm_T_plus_AD": "identity", "mlp_T": "logistic", "mlp_T_plus_AD": "identity",
    }
    assert MODULE.ALGORITHM_PAIRS == {
        "logit": ["logit_T", "logit_T_plus_AD"],
        "lightgbm": ["lightgbm_T", "lightgbm_T_plus_AD"],
        "mlp": ["mlp_T", "mlp_T_plus_AD"],
    }
    assert MODULE.PRIMARY_COMPARISON == ["lightgbm_T", "lightgbm_T_plus_AD"]
    assert MODULE.SECONDARY_COMPARISON == ["logit_T_plus_AD", "lightgbm_T_plus_AD", "mlp_T_plus_AD"]


def test_approval_grid_is_decimal_safe_exact_and_marks_only_anchors() -> None:
    frame = MODULE.approval_rate_grid()
    values = [Decimal(value) for value in frame.approval_rate_percent]
    assert len(frame) == 161
    assert values[0] == Decimal("40.00") and values[-1] == Decimal("80.00")
    assert all(right - left == Decimal("0.25") for left, right in zip(values, values[1:]))
    assert frame.loc[frame.is_external_display_point, "approval_rate_percent"].tolist() == ["40.00", "50.00", "60.00", "70.00", "80.00"]
    assert all(Decimal(row.approval_rate_probability) == Decimal(row.approval_rate_percent) / 100 for row in frame.itertuples())


def test_risk_grid_is_decimal_safe_exact_and_marks_only_anchors() -> None:
    frame = MODULE.portfolio_risk_budget_grid()
    values = [Decimal(value) for value in frame.risk_budget_percent]
    assert len(frame) == 126
    assert values[0] == Decimal("0.75") and values[-1] == Decimal("2.00")
    assert all(right - left == Decimal("0.01") for left, right in zip(values, values[1:]))
    assert frame.loc[frame.is_external_display_point, "risk_budget_percent"].tolist() == ["0.75", "1.00", "1.25", "1.50", "1.75", "2.00"]
    assert all(Decimal(row.risk_budget_probability) == Decimal(row.risk_budget_percent) / 100 for row in frame.itertuples())


def test_policy_tie_break_and_individual_prefix_rules_are_frozen() -> None:
    contract = MODULE.analysis_contract("fixed")
    assert contract["fixed_approval_contract"]["ranking_rule"] == MODULE.STABLE_TIE_BREAK_RULE
    assert contract["fixed_risk_contract"]["ranking_rule"] == MODULE.STABLE_TIE_BREAK_RULE
    assert "every individual ranked prefix" in contract["fixed_risk_contract"]["selection_rule"]
    assert "feasibility_flag=false" in contract["fixed_risk_contract"]["zero_approval_rule"]


def test_economic_ratios_threshold_arithmetic_roles_and_strict_rule() -> None:
    frame = MODULE.economic_scenario_contract()
    assert frame.loss_to_gain_ratio.tolist() == [2, 5, 10, 20, 30, 50]
    for row in frame.itertuples():
        threshold = Decimal(row.individual_pd_threshold_probability)
        exact = Decimal(1) / (Decimal(1) + Decimal(row.loss_to_gain_ratio))
        assert abs(threshold - exact) <= Decimal("5e-19")
        assert "p < p_star" in row.approval_rule
    roles = dict(zip(frame.loss_to_gain_ratio, frame.display_role))
    assert roles[30] == "primary"
    assert roles[20] == "lower_loss_sensitivity"
    assert roles[50] == "higher_loss_sensitivity"
    assert all(roles[value] == "detailed_only" for value in (2, 5, 10))


def test_expected_and_realised_economic_values_remain_separate() -> None:
    contract = MODULE.analysis_contract("fixed")["normalized_economic_contract"]
    assert contract["predicted_expected_and_target_realised_values_are_separate"] is True
    assert contract["same_target_and_scenario_across_models"] is True
    schema = {item["file_name"]: item for item in MODULE.future_output_schema()["future_output_files"]}
    columns = schema["economic_scenario_results.csv"]["required_columns"]
    assert "expected_scenario_value_total" in columns
    assert "realised_scenario_value_total" in columns
    assert "incremental_value_from_AD" in schema["economic_incremental_value.csv"]["required_columns"]


def test_future_result_inventory_and_contract_metadata_are_exact() -> None:
    schema = MODULE.future_output_schema()
    files = schema["future_output_files"]
    assert [item["file_name"] for item in files] == MODULE.FUTURE_OUTPUT_NAMES
    assert len(files) == 9
    for item in files:
        assert {"grain", "required_columns", "key_columns", "sort_order", "purpose"}.issubset(item)
        assert item["required_columns"] and item["key_columns"] and item["sort_order"]


def test_future_prediction_and_subgroup_required_columns_are_complete() -> None:
    schemas = {item["file_name"]: item for item in MODULE.future_output_schema()["future_output_files"]}
    prediction_columns = schemas["final_evaluation_predictions.parquet"]["required_columns"]
    assert prediction_columns[:6] == ["case_id", "base_order", "target", "account_group", "bureau_history_group", "ad_richness_group"]
    assert prediction_columns[6:] == [f"calibrated_pd__{model}" for model in MODULE.MODEL_ORDER]
    metrics = set(schemas["subgroup_model_metrics.csv"]["required_columns"])
    assert {"sample_count", "positive_count", "negative_count", "observed_default_rate", "mean_calibrated_pd",
            "roc_auc", "average_precision", "log_loss", "brier_score", "missing_metric_reason"}.issubset(metrics)


def test_future_policy_required_columns_preserve_predicted_and_realised_risk() -> None:
    schemas = {item["file_name"]: item for item in MODULE.future_output_schema()["future_output_files"]}
    approval = set(schemas["fixed_approval_rate_curve.csv"]["required_columns"])
    assert {"approved_portfolio_mean_calibrated_pd", "cumulative_expected_defaults", "realised_approved_default_rate",
            "stable_tie_break_rule"}.issubset(approval)
    risk = set(schemas["fixed_risk_budget_curve.csv"]["required_columns"])
    assert {"maximum_feasible_approval_rate_percent", "approved_portfolio_mean_calibrated_pd", "budget_slack",
            "feasibility_flag", "realised_approved_default_rate"}.issubset(risk)


def test_visualization_plan_is_schema_only_uses_saved_csvs_and_has_no_smoothing() -> None:
    plan = MODULE.future_output_schema()["visualization_plan"]
    assert plan["part3_visualizations_produced"] is False
    assert plan["approval_default_y_metric"] == "realised_approved_default_rate"
    assert "saved detailed CSVs" in plan["data_source_rule"]
    assert plan["smoothing"].startswith("none")


def test_part3_output_inventory_has_no_result_or_visual_artifact() -> None:
    assert len(MODULE.OUTPUT_NAMES) == 9
    assert MODULE.OUTPUT_NAMES == {
        "analysis_contract.json", "approval_rate_grid.csv", "portfolio_risk_budget_grid.csv",
        "economic_scenario_contract.csv", "final_evaluation_output_schema.json", "input_hashes_before_after.csv",
        "validation_results.csv", "run_config.json", "task11_part3_report.md",
    }
    forbidden = {".parquet", ".png", ".pdf", ".html", ".ipynb", ".pptx"}
    assert not any(Path(name).suffix.lower() in forbidden for name in MODULE.OUTPUT_NAMES)


def test_final_evaluation_boundary_flags_are_all_inactive() -> None:
    boundary = MODULE.analysis_contract("fixed")["final_evaluation_data_boundary"]
    assert boundary["project_wide_metadata_used_for_integrity_verification_only"] is True
    for key, value in boundary.items():
        if key not in {"project_wide_metadata_used_for_integrity_verification_only", "freeze_statement"}:
            assert value is False, key
    source = (ROOT / "scripts/freeze_task11_analysis_contract.py").read_text(encoding="utf-8")
    assert "outer_split.eq(\"final" not in source
    assert "final_evaluation_predictions.parquet" not in MODULE.OUTPUT_NAMES


def test_protected_output_boundary_rejection() -> None:
    data_root = Path("/example/data")
    assert MODULE.output_path_is_protected(data_root, data_root / "audits/task11/part1/new")
    assert MODULE.output_path_is_protected(data_root, data_root / "audits/task10/changed")
    assert not MODULE.output_path_is_protected(data_root, data_root / "audits/task11/part3")


def test_hash_helper_detects_integrity_and_change(tmp_path: Path) -> None:
    protected = tmp_path / "protected.txt"
    protected.write_text("fixed", encoding="utf-8")
    records = [("artifact", "Task", protected)]
    before = MODULE.hash_paths(records)
    unchanged = MODULE.hash_paths(records)
    assert before[str(protected.resolve())]["sha256"] == unchanged[str(protected.resolve())]["sha256"]
    protected.write_text("changed", encoding="utf-8")
    after = MODULE.hash_paths(records)
    assert before[str(protected.resolve())]["sha256"] != after[str(protected.resolve())]["sha256"]


def test_readme_update_preserves_everything_outside_bounded_section(tmp_path: Path) -> None:
    before = "# Existing\n\nUser edit.\n"
    block = MODULE.readme_block(tmp_path)
    first = MODULE.planned_readme(before, block)
    assert MODULE.outside_readme_section(first) == before.rstrip()
    updated_block = block.replace("complete through Part 3", "complete through Part 3")
    second = MODULE.planned_readme(first, updated_block)
    assert MODULE.outside_readme_section(second) == before.rstrip()
    assert second.count(MODULE.README_START) == second.count(MODULE.README_END) == 1


def test_dataframe_contracts_have_required_columns() -> None:
    approval = MODULE.approval_rate_grid()
    risk = MODULE.portfolio_risk_budget_grid()
    economic = MODULE.economic_scenario_contract()
    assert {"grid_order", "approval_rate_percent", "approval_rate_probability", "is_external_display_point", "approved_count_rule"}.issubset(approval.columns)
    assert {"grid_order", "risk_budget_percent", "risk_budget_probability", "is_external_display_point", "selection_rule"}.issubset(risk.columns)
    assert {"scenario_id", "normalized_G", "normalized_L", "loss_to_gain_ratio", "expected_value_formula",
            "individual_pd_threshold_probability", "individual_pd_threshold_percent", "approval_rule",
            "realised_value_formula", "display_role", "accounting_interpretation"}.issubset(economic.columns)
    assert isinstance(approval, pd.DataFrame) and isinstance(risk, pd.DataFrame) and isinstance(economic, pd.DataFrame)
