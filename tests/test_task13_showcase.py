from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
TASK12_DIR = Path("/Users/haoguannan/Projects/home_credit/data/final_evaluation/task12")
SHOWCASE_DIR = ROOT / "data/showcase"
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_showcase_data as PREPARE  # noqa: E402
import run_task12_final_evaluation as TASK12  # noqa: E402
from src import showcase_charts as CHARTS  # noqa: E402


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_task12_generator_returns_separate_economic_frames() -> None:
    n = 12
    predictions = pd.DataFrame({"target": [0, 1] * (n // 2)})
    for index, model in enumerate(TASK12.MODEL_ORDER):
        predictions[f"calibrated_pd__{model}"] = np.linspace(0.01, 0.2, n) + index * 0.0001
    scenarios = pd.DataFrame([{
        "scenario_id": "normalized_gl_2", "loss_to_gain_ratio": 2,
        "individual_pd_threshold_probability": "0.333333333333333333",
        "individual_pd_threshold_percent": "33.3333333333333333", "display_role": "detailed_only",
    }])
    model_level, within, complexity = TASK12.economic_outputs(predictions, scenarios)
    assert len(model_level) == 6
    assert len(within) == 3 and set(within.comparison_type) == {"within_family_ad"}
    assert len(complexity) == 3 and set(complexity.comparison_type) == {"t_plus_ad_complexity"}
    assert not within.duplicated(["algorithm_family", "scenario_id"]).any()
    assert not complexity.duplicated(["comparison_id", "scenario_id"]).any()


def test_economic_repair_is_exact_and_idempotent(tmp_path: Path) -> None:
    repaired_within = pd.read_csv(TASK12_DIR / "economic_incremental_value.csv")
    repaired_complexity = pd.read_csv(TASK12_DIR / "economic_complexity_comparisons.csv")
    original = pd.concat([repaired_within, repaired_complexity], ignore_index=True)
    for name in TASK12.OUTPUT_NAMES - {"economic_complexity_comparisons.csv"}:
        (tmp_path / name).write_text("placeholder\n", encoding="utf-8")
    original.to_csv(tmp_path / "economic_incremental_value.csv", index=False)
    shutil.copy2(TASK12_DIR / "validation_results.csv", tmp_path / "validation_results.csv")
    validation = pd.read_csv(tmp_path / "validation_results.csv")
    validation = validation.loc[~validation.validation_id.isin([
        "row_count::economic_complexity_comparisons.csv",
        "unique_key::economic_complexity_comparisons.csv",
        "economic_split_reconciliation",
    ])]
    validation.loc[validation.validation_id.eq("row_count::economic_incremental_value.csv"), ["observed_result", "expected_result"]] = ["36", "36"]
    validation.to_csv(tmp_path / "validation_results.csv", index=False)
    (tmp_path / "run_config.json").write_text(json.dumps({"output_inventory": sorted(TASK12.OUTPUT_NAMES - {"economic_complexity_comparisons.csv"})}), encoding="utf-8")
    (tmp_path / "task12_final_evaluation_report.md").write_text("- `economic_incremental_value.csv`: 36 rows\nAll 51 Task 12 validations passed.\n", encoding="utf-8")

    first = TASK12.repair_economic_output(tmp_path)
    assert first["status"] == "REPAIRED" and first["reconciliation"].startswith("PASS")
    paths = [tmp_path / "economic_incremental_value.csv", tmp_path / "economic_complexity_comparisons.csv"]
    before = {path.name: file_hash(path) for path in paths}
    second = TASK12.repair_economic_output(tmp_path)
    after = {path.name: file_hash(path) for path in paths}
    assert second == {"status": "ALREADY_REPAIRED", "within_family_rows": 18, "complexity_rows": 18, "writes_performed": False}
    assert before == after


def test_formal_economic_split_keys_values_and_metadata() -> None:
    within = pd.read_csv(TASK12_DIR / "economic_incremental_value.csv")
    complexity = pd.read_csv(TASK12_DIR / "economic_complexity_comparisons.csv")
    TASK12._validate_economic_split(within, complexity)
    assert len(within) == len(complexity) == 18
    assert not within.duplicated(["algorithm_family", "scenario_id"]).any()
    assert not complexity.duplicated(["comparison_id", "scenario_id"]).any()
    validation = pd.read_csv(TASK12_DIR / "validation_results.csv")
    counts = validation.set_index("validation_id")
    assert int(counts.loc["row_count::economic_incremental_value.csv", "expected_result"]) == 18
    assert int(counts.loc["row_count::economic_complexity_comparisons.csv", "expected_result"]) == 18
    config = json.loads((TASK12_DIR / "run_config.json").read_text(encoding="utf-8"))
    assert "economic_complexity_comparisons.csv" in config["output_inventory"]


def test_showcase_inventory_rows_columns_and_no_row_level_identifiers() -> None:
    expected = {**PREPARE.COPY_FILES, "risk_migration_5_10_20.csv": 1575}
    assert {path.name for path in SHOWCASE_DIR.glob("*.csv")} == set(expected)
    frames = {name: pd.read_csv(SHOWCASE_DIR / name) for name in expected}
    PREPARE.validate_showcase_frames(frames)
    assert {name: len(frame) for name, frame in frames.items()} == expected
    assert not any({"case_id", "base_order", "target"} & set(frame.columns) for frame in frames.values())


def test_showcase_migration_shapes_totals_shares_and_formal_k5_reconciliation() -> None:
    migration = pd.read_csv(SHOWCASE_DIR / "risk_migration_5_10_20.csv")
    formal = pd.read_csv(TASK12_DIR / "risk_quintile_migration.csv")
    PREPARE.validate_migration(migration, formal)
    assert migration.groupby(["algorithm_family", "n_bins"]).size().to_dict() == {
        (family, bins): bins * bins for family in PREPARE.FAMILIES for bins in (5, 10, 20)
    }
    assert migration.groupby(["algorithm_family", "n_bins"]).case_count.sum().eq(228_999).all()
    assert np.allclose(
        migration.groupby(["algorithm_family", "n_bins", "t_risk_bin"]).share_within_t_bin.sum(),
        1.0, rtol=0, atol=1e-12,
    )


def test_every_chart_control_returns_plotly_figure() -> None:
    overall = pd.read_csv(SHOWCASE_DIR / "overall_model_metrics.csv")
    subgroup = pd.read_csv(SHOWCASE_DIR / "subgroup_incremental_gains.csv")
    migration = pd.read_csv(SHOWCASE_DIR / "risk_migration_5_10_20.csv")
    approval = pd.read_csv(SHOWCASE_DIR / "fixed_approval_rate_curve.csv")
    risk = pd.read_csv(SHOWCASE_DIR / "fixed_risk_budget_curve.csv")
    economic = pd.read_csv(SHOWCASE_DIR / "economic_incremental_value.csv")
    figures = []
    for metric in ("ROC AUC", "Average Precision", "Log Loss", "Brier Score"):
        figures.append(CHARTS.performance_figure(overall, metric))
    for dimension in (
        "Number of credit accounts",
        "Credit-history length (years)",
        "Alternative-data richness (number of observed sections)",
    ):
        for algorithm in ("Logit (baseline)", "LightGBM", "MLP neural network"):
            for metric in (
                "ROC AUC gain",
                "Average Precision gain",
                "Log-loss improvement",
                "Brier improvement",
            ):
                figures.append(CHARTS.heterogeneity_figure(subgroup, dimension, algorithm, metric))
    for algorithm in ("Logit (baseline)", "LightGBM", "MLP neural network"):
        for n_bins in (5, 10, 20):
            figures.append(CHARTS.reranking_figure(migration, algorithm, n_bins))
        for policy in ("Same approval rate", "Same portfolio risk"):
            figures.append(CHARTS.allocation_figure(approval, risk, policy, algorithm))
        for basis in (
            "Expected value based on model probabilities",
            "Realised value based on observed outcomes",
        ):
            figures.append(CHARTS.value_figure(economic, algorithm, basis))
    assert len(figures) == 61
    assert all(isinstance(figure, go.Figure) for figure in figures)


def test_app_runtime_is_showcase_only_and_has_five_views() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'SHOWCASE_DIR = ROOT / "data/showcase"' in source
    assert "final_evaluation_predictions" not in source
    assert 'default="Model performance"' in source
    assert 'APP_TITLE = "The Unequal Value and Cost of Alternative Data in Credit Risk Predictions"' in source
    assert "Choose a result section:" in source
    assert "Choose a results section:" not in source
    dimension_block = source[
        source.index("dimension = first.selectbox"):source.index("algorithm = second.selectbox")
    ]
    assert dimension_block.index('"Number of credit accounts"') < dimension_block.index(
        '"Credit-history length (years)"'
    )
    assert "index=0" in dimension_block
    for label in (
        "Model performance",
        "Information heterogeneity",
        "Risk re-ranking",
        "Credit re-allocation",
        "Lender-side value",
    ):
        assert f'"{label}"' in source


def test_heterogeneity_uses_human_readable_categorical_bar_labels() -> None:
    subgroup = pd.read_csv(SHOWCASE_DIR / "subgroup_incremental_gains.csv")
    cases = {
        "Number of credit accounts": ["3+ accounts", "2 accounts", "1 account", "0 accounts"],
        "Credit-history length (years)": [
            "More than 5 years", "4–5 years", "3–4 years", "2–3 years",
            "1–2 years", "0.5–1 year", "0–0.5 years",
        ],
        "Alternative-data richness (number of observed sections)": [
            "3 sections", "2 sections", "1 section", "0 sections",
        ],
    }
    raw_labels = {"3_plus", "0_to_0_5", "0_5_to_1", "1_to_2", "2_to_3", "3_to_4", "4_to_5", "gt_5"}
    for dimension, expected in cases.items():
        figure = CHARTS.heterogeneity_figure(subgroup, dimension, "LightGBM", "ROC AUC gain")
        assert len(figure.data) == 1
        assert figure.data[0].type == "bar" and figure.data[0].orientation == "h"
        assert math.isclose(float(figure.data[0].width), 0.48)
        assert list(figure.data[0].y) == expected
        assert figure.layout.yaxis.type == "category"
        assert not raw_labels.intersection(figure.data[0].y)


def test_reranking_annotations_use_complete_two_diagonal_band() -> None:
    migration = pd.read_csv(SHOWCASE_DIR / "risk_migration_5_10_20.csv")
    for n_bins in (5, 10, 20):
        figure = CHARTS.reranking_figure(migration, "LightGBM", n_bins)
        annotated = {(int(annotation.y), int(annotation.x)) for annotation in figure.layout.annotations}
        expected = {
            (origin, destination)
            for origin in range(1, n_bins + 1)
            for destination in range(1, n_bins + 1)
            if abs(destination - origin) <= 2
        }
        assert annotated == expected
        assert all(annotation.text.endswith("%") for annotation in figure.layout.annotations)
        assert "<br>" in figure.layout.xaxis.title.text
        assert "<br>" in figure.layout.yaxis.title.text
        assert f"{n_bins} = highest risk" in figure.layout.xaxis.title.text
        assert f"{n_bins} = highest risk" in figure.layout.yaxis.title.text


def _assert_linear_range_fits_values(figure: go.Figure, values: np.ndarray, *, zero_baseline: bool = False) -> None:
    low, high = map(float, figure.layout.xaxis.range)
    assert math.isfinite(low) and math.isfinite(high) and low < high
    observed_low, observed_high = float(values.min()), float(values.max())
    assert low <= observed_low <= observed_high <= high
    if zero_baseline and observed_low >= 0:
        assert low == 0
        assert high <= observed_high * 1.10 + 1e-12
    else:
        span = observed_high - observed_low
        assert observed_low - low <= span * 0.061
        assert high - observed_high <= span * 0.061


def test_non_heatmap_x_ranges_are_finite_and_selection_specific() -> None:
    overall = pd.read_csv(SHOWCASE_DIR / "overall_model_metrics.csv")
    subgroup = pd.read_csv(SHOWCASE_DIR / "subgroup_incremental_gains.csv")
    approval = pd.read_csv(SHOWCASE_DIR / "fixed_approval_rate_curve.csv")
    risk = pd.read_csv(SHOWCASE_DIR / "fixed_risk_budget_curve.csv")
    economic = pd.read_csv(SHOWCASE_DIR / "economic_incremental_value.csv")

    performance_columns = {
        "ROC AUC": "roc_auc", "Average Precision": "average_precision",
        "Log Loss": "log_loss", "Brier Score": "brier_score",
    }
    performance_ranges = []
    for metric, column in performance_columns.items():
        figure = CHARTS.performance_figure(overall, metric)
        _assert_linear_range_fits_values(figure, overall[column].to_numpy(float))
        performance_ranges.append(tuple(figure.layout.xaxis.range))
    assert len(set(performance_ranges)) == 4

    heterogeneity_ranges = []
    for dimension, dimension_id in (
        ("Number of credit accounts", "account_count"),
        ("Credit-history length (years)", "bureau_history"),
        ("Alternative-data richness (number of observed sections)", "ad_richness"),
    ):
        figure = CHARTS.heterogeneity_figure(subgroup, dimension, "LightGBM", "ROC AUC gain")
        selected = subgroup.loc[
            subgroup.group_dimension.eq(dimension_id) & subgroup.algorithm_family.eq("lightgbm"),
            "roc_auc_gain",
        ].to_numpy(float)
        _assert_linear_range_fits_values(figure, selected, zero_baseline=True)
        heterogeneity_ranges.append(tuple(figure.layout.xaxis.range))
    assert len(set(heterogeneity_ranges)) == 3

    approval_figure = CHARTS.allocation_figure(approval, risk, "Same approval rate", "LightGBM")
    _assert_linear_range_fits_values(
        approval_figure,
        approval.loc[approval.model_id.isin(["lightgbm_T", "lightgbm_T_plus_AD"]), "approval_rate_percent"].to_numpy(float),
    )
    risk_figure = CHARTS.allocation_figure(approval, risk, "Same portfolio risk", "LightGBM")
    _assert_linear_range_fits_values(
        risk_figure,
        risk.loc[risk.model_id.isin(["lightgbm_T", "lightgbm_T_plus_AD"]), "common_risk_budget_percent"].to_numpy(float),
    )
    assert tuple(approval_figure.layout.xaxis.range) != tuple(risk_figure.layout.xaxis.range)

    value = CHARTS.value_figure(
        economic, "LightGBM", "Realised value based on observed outcomes"
    )
    log_low, log_high = map(float, value.layout.xaxis.range)
    observed = economic.loc[economic.algorithm_family.eq("lightgbm"), "loss_to_gain_ratio"].to_numpy(float)
    assert math.isfinite(log_low) and math.isfinite(log_high)
    assert 10 ** log_low <= observed.min() and 10 ** log_high >= observed.max()
    assert math.log10(observed.min()) - log_low <= math.log10(observed.max() / observed.min()) * 0.061
    assert log_high - math.log10(observed.max()) <= math.log10(observed.max() / observed.min()) * 0.061


def test_public_trace_names_and_axes_use_complete_information_set_wording() -> None:
    overall = pd.read_csv(SHOWCASE_DIR / "overall_model_metrics.csv")
    migration = pd.read_csv(SHOWCASE_DIR / "risk_migration_5_10_20.csv")
    approval = pd.read_csv(SHOWCASE_DIR / "fixed_approval_rate_curve.csv")
    risk = pd.read_csv(SHOWCASE_DIR / "fixed_risk_budget_curve.csv")
    figures = [
        CHARTS.performance_figure(overall, "ROC AUC"),
        CHARTS.reranking_figure(migration, "LightGBM", 10),
        CHARTS.allocation_figure(approval, risk, "Same approval rate", "LightGBM"),
    ]
    trace_names = {trace.name for figure in figures for trace in figure.data if trace.name}
    assert "Traditional credit data" in trace_names
    assert "Traditional + alternative data" in trace_names
    assert not {"T", "T+AD"}.intersection(trace_names)
    for figure in figures:
        axis_titles = " ".join(
            axis.title.text or "" for axis in (figure.layout.xaxis, figure.layout.yaxis)
        ).lower()
        assert "risk bin" not in axis_titles
