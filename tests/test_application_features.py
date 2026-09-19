"""Focused synthetic tests for Task 08 feature construction helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("build_application_features", SCRIPT_DIR / "build_application_features.py")
assert SPEC is not None and SPEC.loader is not None
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def test_numeric_reducer_cross_batch_mean_max_and_states():
    reducer = build.NumericReducer(4, ["amount"])
    first = pd.DataFrame({"amount": [0.0, None, -2.0]})
    reducer.update(first, np.array([0, 1, 2]), np.ones(3, dtype=bool))
    second = pd.DataFrame({"amount": [4.0, 6.0, "bad"]})
    reducer.update(second, np.array([2, 2, 3]), np.ones(3, dtype=bool))
    result = reducer.result("amount")
    assert result["finite_count"].tolist() == [1, 0, 3, 0]
    assert result["nonzero_count"].tolist() == [0, 0, 3, 0]
    assert result["mean"][0] == 0
    assert np.isnan(result["mean"][1])
    assert np.isclose(result["mean"][2], 8 / 3)
    assert result["max"][2] == 6
    assert np.isnan(result["mean"][3])
    assert result["invalid_count"].tolist() == [0, 0, 0, 1]


def test_family_states_distinguish_no_evidence_all_missing_zero_and_nonzero():
    states = build.family_states(
        np.array([False, True, True, True]),
        np.array([False, False, True, True]),
        np.array([False, False, False, True]),
    )
    assert {name: np.flatnonzero(mask).tolist() for name, mask in states.items()} == {
        "NO_SOURCE_OR_EVIDENCE": [0],
        "SOURCE_OR_EVIDENCE_PRESENT_NO_FINITE_CONTENT": [1],
        "FINITE_CONTENT_ZERO_ONLY": [2],
        "FINITE_CONTENT_WITH_NONZERO": [3],
    }


def test_history_reducer_does_not_multiply_rows_across_sources():
    left = build.NumericReducer(2, ["x"])
    right = build.NumericReducer(2, ["y"])
    left.update(pd.DataFrame({"x": [1.0, 3.0]}), np.array([0, 0]), np.ones(2, bool))
    right.update(pd.DataFrame({"y": [10.0, 20.0, 30.0]}), np.array([0, 0, 1]), np.ones(3, bool))
    assert left.result("x")["mean"][0] == 2.0
    assert np.isnan(left.result("x")["mean"][1])
    assert right.result("y")["mean"].tolist() == [15.0, 30.0]
    assert left.source_count.tolist() == [2, 0]
    assert right.source_count.tolist() == [2, 1]


def test_applicant_selection_uses_only_num_group1_zero(tmp_path):
    path = tmp_path / "person.parquet"
    pd.DataFrame({
        "case_id": [1, 1, 2, 3, 3],
        "num_group1": [0, 1, 0, 0, 1],
        "personindex_1023L": [0, 1, 0, 0, 1],
        "mainoccupationinc_384A": [100.0, 999.0, 200.0, 300.0, 888.0],
        "incometype_1044T": ["A", "CONTACT", "B", "C", "CONTACT"],
        "education_927M": ["E1", "EC", "E2", "E3", "EC"],
    }).to_parquet(path, index=False)
    selected, details = build.select_applicant_rows(
        [path], pd.Index([1, 2, 3]),
        ["mainoccupationinc_384A", "incometype_1044T", "education_927M"],
    )
    assert selected["mainoccupationinc_384A"].tolist() == [100.0, 200.0, 300.0]
    assert selected["incometype_1044T"].tolist() == ["A", "B", "C"]
    assert details["selected_one"] == 3
    assert details["selected_zero"] == 0
    assert details["selected_multiple"] == 0


def test_ratio_review_keeps_invalid_denominators_undefined_and_defers():
    result = build.ratio_review(
        np.array([0.0, 2.0, 3.0, np.nan]),
        np.array([1.0, 0.0, -1.0, 2.0]),
        "ratio", "numerator / denominator",
    )
    assert result["decision"] == "DEFERRED_UNIT_UNCERTAINTY"
    assert result["eligible_count_if_materialized"] == 1
    assert result["undefined_count_if_materialized"] == 3
    assert result["denominator_zero_count"] == 1
    assert result["denominator_negative_count"] == 1


def test_correction_relabels_only_six_known_disabled_tracking_rows():
    rows = []
    for source in ("tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1"):
        for metric in sorted(build.CORRECTION_METRICS):
            rows.append({
                "section": "TAX_DETAIL_TIMING", "source": source, "field_or_comparison": "date",
                "metric": metric, "value": "0", "denominator_name": "N", "denominator_value": "3",
                "definition": "Application-level date/tie diagnostic.", "status": "MEASURED",
            })
    rows.append({"section": "OTHER", "source": "x", "field_or_comparison": "x", "metric": "kept",
                 "value": "0", "denominator_name": "N", "denominator_value": "3", "definition": "kept", "status": "MEASURED"})
    original, corrected, status = build.correction_rows(rows)
    assert len(original) == 6
    assert status == "CORRECTED_SIX_ROWS"
    changed = [row for row in corrected if row["metric"] in build.CORRECTION_METRICS]
    assert all(row["value"] == "" and row["status"] == "NOT_CHECKED" for row in changed)
    assert corrected[-1] == rows[-1]


def test_exact_duplicate_diagnostic_compares_values_and_missing_masks():
    frame = pd.DataFrame({
        "a": [1.0, np.nan, 0.0],
        "b": [1.0, np.nan, 0.0],
        "c": [1.0, 0.0, np.nan],
    })
    assert build.exact_duplicate_groups(frame, ["a", "b", "c"]) == [["a", "b"]]


def test_candidate_configuration_has_exact_approved_counts():
    assert len(build.T_CANDIDATES) == 21
    assert len(build.AD_CANDIDATES) == 20
    assert sum(row[1] == "debit" for row in build.AD_CANDIDATES) == 5
    assert sum(row[1] == "deposit" for row in build.AD_CANDIDATES) == 4
    assert sum(row[1] == "tax" for row in build.AD_CANDIDATES) == 11
