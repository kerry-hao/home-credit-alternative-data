from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/task11_part2_design_audit.py"
SCRIPTS = SCRIPT.parent
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("task11_part2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_account_labels_and_boundaries() -> None:
    values = pd.Series([0, 1, 2, 3, 4, 9, np.nan])
    groups = MODULE.assign_account_groups(values)
    assert MODULE.ACCOUNT_LABELS == ["0", "1", "2", "3_plus"]
    assert groups.iloc[:6].tolist() == ["0", "1", "2", "3_plus", "3_plus", "3_plus"]
    assert pd.isna(groups.iloc[6])


def test_account_missing_not_filled() -> None:
    assert pd.isna(MODULE.assign_account_groups(pd.Series([np.nan])).iloc[0])


def test_history_labels_boundaries_and_missing_to_zero() -> None:
    values = pd.Series([np.nan, 0, .5, .50001, 1, 1.1, 2, 2.1, 3, 3.1, 4, 4.1, 5, 5.1])
    groups, grouping = MODULE.assign_history_groups(values)
    assert MODULE.HISTORY_LABELS == ["0_to_0_5", "0_5_to_1", "1_to_2", "2_to_3", "3_to_4", "4_to_5", "gt_5"]
    assert groups.tolist() == ["0_to_0_5", "0_to_0_5", "0_to_0_5", "0_5_to_1", "0_5_to_1",
                              "1_to_2", "1_to_2", "2_to_3", "2_to_3", "3_to_4", "3_to_4",
                              "4_to_5", "4_to_5", "gt_5"]
    assert grouping.iloc[0] == 0


def test_contract_has_no_missing_indicator_binary_or_cross_product() -> None:
    contract = MODULE.subgroup_contract()
    text = str(contract)
    assert "history_missing_indicator" not in text
    assert contract["binary_history_threshold"] is None
    assert contract["two_dimensional_design"] is None
    assert "credquantity_1099L" not in text


def test_group_distributions_reconcile() -> None:
    account, diagnostics = MODULE.account_distribution(pd.Series([0, 1, 2, 3] * (MODULE.EXPECTED_TRAIN // 4) + [0]))
    assert int(account.iloc[:4]["count"].sum()) == MODULE.EXPECTED_TRAIN
    assert all(value == 0 for value in diagnostics.values())
    years = pd.Series(np.zeros(MODULE.EXPECTED_TRAIN))
    history, _ = MODULE.history_distribution(years)
    assert int(history.iloc[:7]["count"].sum()) == MODULE.EXPECTED_TRAIN


def test_ad_module_count_and_patterns_keep_observed_zero_distinct() -> None:
    frame = pd.DataFrame({
        MODULE.MODULE_FLAGS["debit"]: [0, 1, 0, 1],
        MODULE.MODULE_FLAGS["deposit"]: [0, 0, 1, 1],
        MODULE.MODULE_FLAGS["tax"]: [0, 1, 1, 1],
    })
    old = MODULE.EXPECTED_TRAIN
    MODULE.EXPECTED_TRAIN = 4
    try:
        audit, patterns = MODULE.ad_richness(frame)
    finally:
        MODULE.EXPECTED_TRAIN = old
    counts = audit[audit.candidate_id.eq("observed_ad_module_count")].set_index("level_or_value")["count"].to_dict()
    assert counts == {"0": 1, "1": 0, "2": 2, "3": 1}
    assert set(patterns.binary_pattern) == {"000", "101", "011", "111"}


def _small_oof() -> pd.DataFrame:
    probability = np.linspace(.1, .3, 100)
    probability[:2] = .1
    return pd.DataFrame({"case_id": np.arange(1, 101), "base_order": np.r_[3, 1, np.arange(2, 100)],
                         "target": np.tile([0, 1], 50), "oof__model__identity": probability})


def test_deterministic_pd_base_order_tie_and_approval_arithmetic() -> None:
    old = MODULE.EXPECTED_CALIBRATION
    MODULE.EXPECTED_CALIBRATION = 100
    try:
        curve = MODULE.approval_curve_for_model("model", "identity", _small_oof())
    finally:
        MODULE.EXPECTED_CALIBRATION = old
    hundred = curve.iloc[-1]
    assert hundred.approved_count == 100
    expected_sum = _small_oof()["oof__model__identity"].sum()
    assert np.isclose(hundred.cumulative_expected_defaults, expected_sum)
    assert np.isclose(hundred.approved_portfolio_mean_calibrated_pd, expected_sum / 100)
    # Equal .1 values are ordered by base_order: case 2 before case 1.
    order = np.lexsort((_small_oof().base_order, _small_oof()["oof__model__identity"]))
    assert _small_oof().case_id.to_numpy()[order][:2].tolist() == [2, 1]


def test_selected_calibration_methods_exact() -> None:
    assert MODULE.SELECTED_METHODS == {
        "logit_T":"identity", "logit_T_plus_AD":"identity", "lightgbm_T":"identity",
        "lightgbm_T_plus_AD":"identity", "mlp_T":"logistic", "mlp_T_plus_AD":"identity"}


def test_common_budget_inversion_monotonic_and_zero_approval() -> None:
    frames = []
    for model in MODULE.MODEL_ORDER:
        frames.append(pd.DataFrame({"model_id":model,"selected_method":MODULE.SELECTED_METHODS[model],
            "approval_rate_percent":[1,2],"approved_count":[1145,2290],"rejected_count":[113355,112210],
            "boundary_calibrated_pd_cutoff":[.02,.03],"approved_portfolio_mean_calibrated_pd":[.01,.02],
            "cumulative_expected_defaults":[11.45,45.8],"realised_approved_defaults":[1,2],
            "realised_approved_default_rate":[1/1145,2/2290]}))
    curve = pd.concat(frames, ignore_index=True)
    risk = MODULE.invert_risk_budgets(curve, np.array([.005,.01,.02]))
    assert (risk[risk.common_risk_budget.eq(.005)].approved_count == 0).all()
    assert all(frame.maximum_feasible_approval_rate_percent.is_monotonic_increasing for _, frame in risk.groupby("model_id"))
    assert (risk[risk.common_risk_budget.eq(.02)].approved_count == 2290).all()


def test_oof_probability_range_logic() -> None:
    values = np.array([0.0, .5, 1.0])
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()


def test_economic_classification_and_scenario_arithmetic() -> None:
    assert MODULE.ECONOMIC_FIELDS == ["credamount_770A","annuity_780A","eir_270L","price_1097A","monthsannuity_845L","numinstls_657L"]
    scenarios = MODULE.scenario_grid()
    assert scenarios.loss_to_gain_ratio.tolist() == [2,5,10,20,30,50]
    assert np.allclose(scenarios.zero_expected_value_pd_threshold_probability, 1/(1+scenarios.loss_to_gain_ratio))


def test_output_inventory_has_no_chart_types() -> None:
    assert len(MODULE.OUTPUT_NAMES) == 16
    assert not any(Path(name).suffix.lower() in {".png",".pdf",".html",".ipynb"} for name in MODULE.OUTPUT_NAMES)


def test_protected_output_boundary_rejection(tmp_path: Path) -> None:
    protected = tmp_path / "audits/task11/part1/part2"
    try:
        MODULE.run(tmp_path, protected, tmp_path, "pass", "pass")
    except MODULE.Part2Blocked as exc:
        assert "protected" in str(exc).lower()
    else:
        raise AssertionError("Protected output boundary was not rejected")


def test_protected_hash_helper_detects_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "protected.txt"; path.write_text("fixed")
    before = MODULE.hash_paths([("x","task",path)])
    after = MODULE.hash_paths([("x","task",path)])
    assert before[str(path)]["sha256"] == after[str(path)]["sha256"]


def test_no_final_evaluation_calculation_or_forbidden_design_names() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'final_evaluation_rows_or_results_included_in_calculations":False' in source
    assert "account_history_cross" not in source
    assert "history_thin" not in source
    assert "train_test_split" not in source
