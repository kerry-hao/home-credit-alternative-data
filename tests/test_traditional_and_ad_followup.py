"""Synthetic, in-memory tests for Task 07 Batch 2 audit helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "audit_traditional_and_ad_followup",
    SCRIPT_DIR / "audit_traditional_and_ad_followup.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def run_numeric(values, positions, population, source_counts):
    accumulator = audit.NumericAccumulator(population)
    positions = np.asarray(positions, dtype=np.int64)
    matched = np.ones(len(positions), dtype=bool)
    matched_indices, unique_positions, inverse = audit.batch_groups(positions, matched)
    accumulator.update(pd.Series(values), matched_indices, unique_positions, inverse)
    return accumulator.finish(
        "source", "field", "description", "role", "population",
        len(values), np.asarray(source_counts, dtype=np.int32), False,
    )[1]


def test_numeric_application_states_keep_absent_missing_zero_and_nonzero_distinct():
    result = run_numeric(
        [None, None, 0.0, 0.0, 3.0], [1, 1, 2, 3, 3], 4, [0, 2, 1, 2]
    )
    assert {name: np.flatnonzero(mask).tolist() for name, mask in result["states"].items()} == {
        "NO_SOURCE_RECORD": [0],
        "SOURCE_PRESENT_NO_FINITE_OBSERVATION": [1],
        "FINITE_OBSERVATIONS_ZERO_ONLY": [2],
        "AT_LEAST_ONE_FINITE_NONZERO_OBSERVATION": [3],
    }


def test_repeated_summary_values_are_compared_not_summed():
    result = run_numeric([5.0, 5.0, 3.0, 4.0], [0, 0, 1, 1], 2, [2, 2])
    assert result["app_sum"].tolist() == [10.0, 7.0]
    assert result["app_min"].tolist() == [5.0, 3.0]
    assert result["app_max"].tolist() == [5.0, 4.0]
    assert result["states"]["AT_LEAST_ONE_FINITE_NONZERO_OBSERVATION"].all()


def test_shard_case_overlap_is_not_composite_duplication_and_conflicts_are_visible():
    result = audit.shard_pair_diagnostics(
        np.array([0, 1]), np.array([1, 2]),
        np.array([10, 20], dtype=np.uint64), np.array([100, 200], dtype=np.uint64),
        np.array([20, 30], dtype=np.uint64), np.array([201, 300], dtype=np.uint64),
    )
    assert result == {
        "case_id_overlap_count": 1,
        "composite_key_overlap_count": 1,
        "conflicting_selected_content_composite_keys": 1,
    }


def test_no_active_side_evidence_is_not_no_credit_history():
    states = audit.active_closed_evidence_states(
        np.array([0, 1, 1, 1]),
        np.array([False, False, True, False]),
        np.array([False, False, False, True]),
    )
    assert np.flatnonzero(states["applications_no_source_record"]).tolist() == [0]
    assert np.flatnonzero(states["applications_neither_active_nor_closed_evidence"]).tolist() == [1]
    assert np.flatnonzero(states["applications_active_evidence_only"]).tolist() == [2]
    assert np.flatnonzero(states["applications_closed_evidence_only"]).tolist() == [3]


def test_missing_bureau_b_source_is_not_observed_zero_or_thin():
    counts = {
        "finite": np.array([0, 1, 1]),
        "unparseable": np.zeros(3, dtype=np.int32),
        "nonfinite": np.zeros(3, dtype=np.int32),
    }
    state, value, consistent = audit.credit_candidate_state(
        np.array([0, 1, 1]), counts,
        np.array([np.nan, 0.0, 2.0]), np.array([np.nan, 0.0, 2.0]),
    )
    assert state.tolist() == [0, 4, 5]
    assert np.isnan(value[0])
    assert consistent.tolist() == [False, True, True]


def test_person_selection_covers_no_source_zero_one_and_multiple_without_first_record_rule():
    states = audit.person_selection_states(
        np.array([0, 2, 1, 3]), np.array([0, 0, 1, 2])
    )
    assert {name: np.flatnonzero(mask).tolist() for name, mask in states.items()} == {
        "NO_PERSON_SOURCE": [0],
        "SOURCE_PRESENT_ZERO_NUM_GROUP1_ZERO_ROWS": [1],
        "EXACTLY_ONE_NUM_GROUP1_ZERO_ROW": [2],
        "MULTIPLE_NUM_GROUP1_ZERO_ROWS": [3],
    }
    assert not hasattr(audit, "select_first_person_record")


def test_future_contractual_and_event_dates_are_measured_without_censoring():
    base_dates = np.array(["2020-01-01", "2020-01-01"], dtype="datetime64[ns]")
    for field in ("contractenddate_991D", "recorddate_4527225D"):
        accumulator = audit.DateAccumulator(2)
        accumulator.update(
            pd.Series(["2020-01-05", "2019-12-31"]),
            np.array([0, 1]), np.array([0, 1]), base_dates,
            np.array([11, 22], dtype=np.uint64),
        )
        rows, result = accumulator.finish(
            "source", field, "description", "date", "population", 2,
            np.array([1, 1], dtype=np.int32),
        )
        metrics = {row["metric"]: row["value"] for row in rows}
        assert metrics["date_offset_after_count"] == 1
        assert metrics["date_offset_before_count"] == 1
        assert result["valid_pairs"] == 2


def test_disabled_latest_tracking_is_not_reported_as_measured_zero():
    accumulator = audit.DateAccumulator(1, track_latest=False)
    accumulator.update(
        pd.Series(["2020-01-02", "2020-01-02"]), np.array([0, 1]),
        np.array([0, 0]), np.array(["2020-01-01"], dtype="datetime64[ns]"),
        np.array([10, 20], dtype=np.uint64),
    )
    rows, _ = accumulator.finish(
        "source", "event_date", "", "date", "population", 2,
        np.array([2], dtype=np.int32),
    )
    latest = {
        row["metric"]: row
        for row in rows
        if row["metric"] in {
            "applications_latest_date_tied",
            "applications_latest_tie_payload_conflict",
        }
    }
    assert set(latest) == {
        "applications_latest_date_tied",
        "applications_latest_tie_payload_conflict",
    }
    assert all(row["value"] is None for row in latest.values())
    assert all(row["status"] == "NOT_CHECKED" for row in latest.values())
    assert all("tracking was disabled" in row["definition"] for row in latest.values())


def test_tax_c_states_cover_absent_zero_all_missing_partial_and_nonzero():
    states = audit.tax_c_application_states(
        np.array([0, 1, 2, 2, 2]),
        np.array([0, 0, 2, 1, 1]),
        np.array([np.nan, np.nan, 0.0, 5.0, -2.0]),
    )
    assert {name: np.flatnonzero(mask).tolist() for name, mask in states.items()} == {
        "NO_C_SOURCE": [0],
        "C_SOURCE_ALL_AMOUNTS_MISSING_OR_INVALID": [1],
        "C_SOURCE_FINITE_ZERO_SUM": [2],
        "C_SOURCE_FINITE_NONZERO_SUM": [3, 4],
    }


def test_tax_c_count_and_float_comparison_reports_exact_and_tolerance_results():
    population = 3
    static_count = {
        "app_min": np.array([0.0, 2.0, 1.0]),
        "app": {"finite": np.ones(3, dtype=np.int32), "nonzero": np.array([0, 1, 1])},
    }
    static_sum = {
        "app_min": np.array([0.0, 0.3, 5.0]),
        "app": {"finite": np.ones(3, dtype=np.int32)},
    }
    tax_c = {
        "source_counts": np.array([0, 2, 1]),
        "amount": {
            "app": {
                "finite": np.array([0, 2, 1]), "missing": np.zeros(3, dtype=np.int32),
                "unparseable": np.zeros(3, dtype=np.int32), "nonfinite": np.zeros(3, dtype=np.int32),
            },
            "app_sum": np.array([0.0, 0.1 + 0.2, 5.0]),
        },
    }
    rows = audit.tax_followup_rows(population, static_count, static_sum, tax_c)
    metrics = {(row["field_or_comparison"], row["metric"]): row["value"] for row in rows}
    assert metrics[("pmtscount_vs_c_record_count", "exact_match_count")] == 3
    assert metrics[("pmtssum_vs_c_finite_amount_sum", "exact_match_count")] == 1
    assert metrics[("pmtssum_vs_c_finite_amount_sum", "tolerance_match_count")] == 2


def test_latest_ties_and_repeated_signatures_survive_unique_local_indices():
    accumulator = audit.DateAccumulator(1)
    accumulator.update(
        pd.Series(["2020-01-02", "2020-01-02"]), np.array([0, 1]),
        np.array([0, 0]), np.array(["2020-01-01"], dtype="datetime64[ns]"),
        np.array([10, 20], dtype=np.uint64),
    )
    _, result = accumulator.finish(
        "source", "date", "", "date", "population", 2,
        np.array([2], dtype=np.int32),
    )
    assert result["latest_count"].tolist() == [2]
    assert result["latest_conflict"].tolist() == [True]

    frame = pd.DataFrame({
        "case_id": [1, 1], "num_group1": [0, 1], "opening": ["2020-01-01"] * 2,
        "amount": [5.0, 5.0],
    })
    signature = audit.repeated_signature_metrics(
        frame, ["case_id", "opening", "amount"], ["opening"], ["amount"]
    )
    assert signature["repeated_signature_group_count"] == 1
    assert signature["repeated_signature_excess_rows"] == 1


def test_grouped_date_matching_never_creates_cartesian_rows():
    result = audit.grouped_date_comparison(
        np.array([0, 0, 1]), np.array([10, 10, 20]),
        np.array([0, 0, 1]), np.array([10, 10, 21]),
    )
    assert result["matched_group_count"] == 1
    assert result["matched_ambiguous_multirow_group_count"] == 1
    assert result["matched_equal_multiplicity_count"] == 1
    assert result["left_only_group_count"] == 1
    assert result["right_only_group_count"] == 1
