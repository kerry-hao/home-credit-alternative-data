from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/Users/haoguannan/Projects/home_credit/data")
TASK12_OUTPUT = DATA_ROOT / "final_evaluation/task12"
sys.path.insert(0, str(ROOT / "scripts"))
import run_task12_final_evaluation as MODULE  # noqa: E402


def small_predictions(n: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({
        "case_id": np.arange(100, 100 + n),
        "base_order": np.arange(n),
        "target": np.array([0, 1] * (n // 2), dtype=np.int8),
        "account_group": ["0"] * n,
        "bureau_history_group": ["0_to_0_5"] * n,
        "ad_richness_group": ["0"] * n,
    })
    base = np.linspace(0.01, 0.10, n)
    for index, model in enumerate(MODULE.MODEL_ORDER):
        frame[f"calibrated_pd__{model}"] = np.clip(base + index * 0.001, 0, 1)
    return frame


def test_phase_a_readme_is_complete_before_formal_output() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if TASK12_OUTPUT.exists():
        assert "Task 12 formal final evaluation is complete" in readme
    else:
        rows = MODULE.verify_phase_a_readme(readme)
        assert rows and all(row["status"] == "PASS" for row in rows)
    assert "/Users/" not in readme


def test_final_membership_count_uniqueness_order_and_split_separation() -> None:
    final, positions, keys, summary = MODULE.load_final_membership(DATA_ROOT)
    assert len(final) == MODULE.EXPECTED_FINAL == 228_999
    assert final.case_id.is_unique and final.base_order.is_unique
    assert np.array_equal(final.base_order.to_numpy(), positions)
    assert len(keys) == MODULE.EXPECTED_TOTAL
    assert summary["train_final_overlap"] == summary["validation_final_overlap"] == 0
    assert summary["membership_fingerprint"] == MODULE.EXPECTED_FINGERPRINT


def test_exact_models_selected_methods_and_comparison_sets() -> None:
    assert MODULE.MODEL_ORDER == [
        "logit_T", "logit_T_plus_AD", "lightgbm_T", "lightgbm_T_plus_AD", "mlp_T", "mlp_T_plus_AD",
    ]
    assert MODULE.SELECTED_METHODS == {
        "logit_T": "identity", "logit_T_plus_AD": "identity", "lightgbm_T": "identity",
        "lightgbm_T_plus_AD": "identity", "mlp_T": "logistic", "mlp_T_plus_AD": "identity",
    }
    assert MODULE.PRIMARY_COMPARISON == ["lightgbm_T", "lightgbm_T_plus_AD"]
    assert MODULE.SECONDARY_COMPARISON == ["logit_T_plus_AD", "lightgbm_T_plus_AD", "mlp_T_plus_AD"]


def test_no_fitting_retraining_or_recalibration_call_path() -> None:
    source = (ROOT / "scripts/run_task12_final_evaluation.py").read_text(encoding="utf-8")
    assert not re.findall(r"\.fit\s*\(", source)
    assert not re.findall(r"\.fit_transform\s*\(", source)
    with MODULE.no_fit_guard():
        with pytest.raises(MODULE.Task12Blocked):
            LogisticRegressionForTest().fit(None, None)


class LogisticRegressionForTest:
    def fit(self, *_args: object) -> None:
        # The concrete sklearn guards are checked directly below instead.
        from sklearn.linear_model import LogisticRegression
        LogisticRegression().fit([[0.0], [1.0]], [0, 1])


def test_probability_integrity_finite_and_bounds() -> None:
    result = MODULE.validate_probability(np.array([0.0, 0.25, 1.0]), 3)
    assert result["status"] == "PASS"
    with pytest.raises(MODULE.Task12Blocked):
        MODULE.validate_probability(np.array([0.0, np.nan, 1.1]), 3)
    if TASK12_OUTPUT.exists():
        saved = pd.read_parquet(TASK12_OUTPUT / "final_evaluation_predictions.parquet")
        values = saved[[f"calibrated_pd__{model}" for model in MODULE.MODEL_ORDER]].to_numpy()
        assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()


def test_prediction_alignment_contract_across_all_models() -> None:
    frame = small_predictions()
    columns = [f"calibrated_pd__{model}" for model in MODULE.MODEL_ORDER]
    assert frame.case_id.is_unique
    assert all(frame[column].notna().sum() == len(frame) for column in columns)
    if TASK12_OUTPUT.exists():
        saved = pd.read_parquet(TASK12_OUTPUT / "final_evaluation_predictions.parquet")
        assert saved.case_id.is_unique and all(saved[column].notna().sum() == 228_999 for column in columns)


def test_account_and_history_subgroup_boundary_cases() -> None:
    assert [MODULE.part3.assign_account_group(value) for value in (0, 1, 2, 3, 10)] == ["0", "1", "2", "3_plus", "3_plus"]
    values = pd.Series([np.nan, 0, 0.5, 0.5001, 1, 1.0001, 2, 2.0001, 3, 3.0001, 4, 4.0001, 5, 5.0001])
    labels, grouping = MODULE.part2_assign_history(values)
    assert grouping.iloc[0] == 0
    assert labels.tolist() == [
        "0_to_0_5", "0_to_0_5", "0_to_0_5", "0_5_to_1", "0_5_to_1", "1_to_2", "1_to_2",
        "2_to_3", "2_to_3", "3_to_4", "3_to_4", "4_to_5", "4_to_5", "gt_5",
    ]
    assert labels.notna().all() and "missing" not in labels.unique()


def test_ad_richness_uses_exact_three_numeric_content_indicators() -> None:
    assert MODULE.AD_INDICATORS == [
        "ad__debit__has_numeric_content", "ad__deposit__has_numeric_content", "ad__tax__has_numeric_content",
    ]
    assert MODULE.part3.assign_ad_richness_group(1, 0, 1) == "2"
    assert [item["label"] for item in MODULE.AD_GROUPS] == ["0", "1", "2", "3"]


def test_subgroups_remain_separate_with_no_cross_product() -> None:
    assert [item[0] for item in MODULE.subgroup_definitions(small_predictions())].count("account_count") == 4
    assert [item[0] for item in MODULE.subgroup_definitions(small_predictions())].count("bureau_history") == 7
    assert [item[0] for item in MODULE.subgroup_definitions(small_predictions())].count("ad_richness") == 4
    assert not any("cross" in column for column in small_predictions().columns)


def test_metric_formulas_and_positive_is_better_gain_orientation() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    better = np.array([0.1, 0.2, 0.8, 0.9])
    worse = np.array([0.4, 0.6, 0.4, 0.6])
    reference = pd.Series({"model_id": "x_T", "n_cases": 4, **MODULE.binary_metrics(y, worse)})
    candidate = pd.Series({"model_id": "x_T_plus_AD", "n_cases": 4, **MODULE.binary_metrics(y, better)})
    gain = MODULE.gain_row(reference, candidate, "x")
    assert gain["roc_auc_gain"] > 0 and gain["average_precision_gain"] > 0
    assert gain["log_loss_gain"] > 0 and gain["brier_gain"] > 0


def test_stable_order_rank_percentile_and_quintiles_under_ties() -> None:
    probability = np.array([0.2, 0.1, 0.1, 0.3, 0.2])
    base_order = np.array([4, 3, 1, 2, 0])
    order = MODULE.stable_order(probability, base_order)
    assert order.tolist() == [2, 1, 4, 0, 3]
    percentile = MODULE.risk_percentile(probability, base_order)
    assert np.allclose(percentile[order], np.linspace(0, 1, 5))
    quintile = MODULE.risk_quintile(probability, base_order)
    assert quintile[order].tolist() == [1, 2, 3, 4, 5]


def test_reassessment_summary_direction_and_quantile_convention() -> None:
    summary = MODULE.distribution_summary(np.array([-0.2, 0.0, 0.1, 0.3]))
    assert summary["n_cases"] == 4
    assert summary["share_below_zero"] == 0.25
    assert summary["share_exactly_zero"] == 0.25
    assert summary["share_above_zero"] == 0.5
    assert summary["quantile_convention"] == MODULE.QUANTILE_CONVENTION


def test_approval_grid_count_anchors_nesting_and_points_reconcile() -> None:
    grid = MODULE.part3.approval_rate_grid()
    curve, _ = MODULE.approval_curve(small_predictions(), grid)
    assert len(grid) == 161 and len(curve) == 161 * 6
    assert grid.approval_rate_percent.iloc[[0, -1]].tolist() == ["40.00", "80.00"]
    assert curve.groupby("model_id").approved_count.apply(lambda values: values.is_monotonic_increasing).all()
    points = curve[curve.is_external_display_point]
    assert len(points) == 30
    assert sorted(points.approval_rate_percent.unique().tolist()) == [40.0, 50.0, 60.0, 70.0, 80.0]


def test_approval_switch_four_cells_and_overall_swap_equality() -> None:
    frame = small_predictions()
    grid = MODULE.part3.approval_rate_grid()
    _curve, cache = MODULE.approval_curve(frame, grid)
    switches = MODULE.approval_switches(frame, cache, [Decimal("50")])
    overall = switches[switches.stratum_dimension.eq("overall")]
    assert set(overall.switch_category) == {"both_approved", "T_only_approved", "T_plus_AD_only_approved", "both_rejected"}
    for _, group in overall.groupby("algorithm_family"):
        counts = group.set_index("switch_category").case_count
        assert counts.sum() == len(frame)
        assert counts.T_only_approved == counts.T_plus_AD_only_approved


def test_risk_grid_monotonic_prefix_equivalence_and_feasibility() -> None:
    frame = small_predictions()
    grid = MODULE.part3.portfolio_risk_budget_grid()
    curve = MODULE.risk_budget_curve(frame, grid)
    assert len(grid) == 126 and len(curve) == 126 * 6
    assert grid.risk_budget_percent.iloc[[0, -1]].tolist() == ["0.75", "2.00"]
    assert all(group.sort_values("common_risk_budget_percent").approved_count.is_monotonic_increasing for _, group in curve.groupby("model_id"))
    feasible = curve[curve.feasibility_flag]
    assert (feasible.approved_portfolio_mean_calibrated_pd <= feasible.common_risk_budget_probability + 1e-12).all()


def test_gl_thresholds_and_strict_approval_inequality() -> None:
    scenarios = MODULE.part3.economic_scenario_contract()
    assert scenarios.loss_to_gain_ratio.tolist() == [2, 5, 10, 20, 30, 50]
    for row in scenarios.itertuples():
        threshold = Decimal(row.individual_pd_threshold_probability)
        exact = Decimal(1) / (Decimal(1) + Decimal(row.loss_to_gain_ratio))
        assert abs(threshold - exact) <= Decimal("5e-19")
        values = np.array([float(threshold) - 1e-8, float(threshold), float(threshold) + 1e-8])
        assert (values < float(threshold)).tolist() == [True, False, False]


def test_predicted_expected_and_target_realised_value_formulas() -> None:
    p = np.array([0.1, 0.2])
    y = np.array([0, 1])
    ratio = 2
    expected = (1 - p) - p * ratio
    realised = np.where(y == 0, 1, -ratio)
    assert np.allclose(expected, [0.7, 0.4])
    assert realised.tolist() == [1, -2]


def test_economic_comparison_direction_and_break_even_normalization() -> None:
    difference = 228.999
    assert difference / MODULE.EXPECTED_FINAL == pytest.approx(0.001)
    assert (-difference) / MODULE.EXPECTED_FINAL == pytest.approx(-0.001)
    source = (ROOT / "scripts/run_task12_final_evaluation.py").read_text(encoding="utf-8")
    assert "candidate.realised_scenario_value_total - reference.realised_scenario_value_total" in source


def test_exact_output_filenames_and_no_visual_artifacts() -> None:
    assert len(MODULE.CORE_OUTPUT_NAMES) == 9
    assert len(MODULE.SUPPLEMENTAL_OUTPUT_NAMES) == 7
    assert len(MODULE.AUDIT_OUTPUT_NAMES) == 5
    assert len(MODULE.OUTPUT_NAMES) == 21
    assert "subgroup_integrity.csv" in MODULE.OUTPUT_NAMES
    assert not any(Path(name).suffix.lower() in {".png", ".pdf", ".html", ".ipynb", ".pptx"} for name in MODULE.OUTPUT_NAMES)


def test_atomic_completion_refuses_existing_output(tmp_path: Path) -> None:
    existing = tmp_path / "task12"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        MODULE.run(tmp_path, existing, ROOT, 1, "test", "test")


def test_protected_hash_integrity_helper(tmp_path: Path) -> None:
    path = tmp_path / "protected.txt"
    path.write_text("fixed", encoding="utf-8")
    records = [("x", "Task", path)]
    before = MODULE.hash_paths(records)
    after = MODULE.hash_paths(records)
    assert MODULE.protected_registry_digest(before) == MODULE.protected_registry_digest(after)
    path.write_text("changed", encoding="utf-8")
    assert MODULE.protected_registry_digest(before) != MODULE.protected_registry_digest(MODULE.hash_paths(records))


def test_completed_outputs_have_required_rows_keys_and_schemas_when_present() -> None:
    if not TASK12_OUTPUT.exists():
        pytest.skip("Formal Task 12 output is not present before the run")
    predictions = pd.read_parquet(TASK12_OUTPUT / "final_evaluation_predictions.parquet")
    assert len(predictions) == 228_999 and predictions.case_id.is_unique
    expected_rows = {
        "subgroup_model_metrics.csv": 90, "subgroup_incremental_gains.csv": 45,
        "fixed_approval_rate_curve.csv": 966, "fixed_approval_rate_points.csv": 30,
        "fixed_risk_budget_curve.csv": 756, "fixed_risk_budget_points.csv": 36,
        "economic_scenario_results.csv": 36, "economic_incremental_value.csv": 18,
        "economic_complexity_comparisons.csv": 18,
        "overall_model_metrics.csv": 6, "overall_incremental_gains.csv": 3,
        "pd_reassessment_summary.csv": 96, "risk_quintile_migration.csv": 1200,
        "approval_switch_summary.csv": 960, "subgroup_integrity.csv": 15,
    }
    for name, expected in expected_rows.items():
        assert len(pd.read_csv(TASK12_OUTPUT / name)) == expected
    validation = pd.read_csv(TASK12_OUTPUT / "validation_results.csv")
    assert validation.status.eq("PASS").all()


def test_final_readme_status_and_repository_relative_links_when_complete() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if TASK12_OUTPUT.exists():
        assert "Task 12 formal final evaluation is complete" in readme
        assert "../data/final_evaluation/task12/overall_model_metrics.csv" in readme
        assert "The local showcase is complete" in readme
    else:
        assert "Task 12 has not yet been executed" in readme
    assert "/Users/" not in readme
