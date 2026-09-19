"""Synthetic, in-memory tests for Task 07 Batch 1 audit helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_ad_coverage.py"
SPEC = importlib.util.spec_from_file_location("audit_ad_coverage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_numeric_states_distinguish_nan_infinity_negative_zero_and_unparseable():
    series = pd.Series([None, "bad", "inf", "0", "2.5", "-3", "NaN"], dtype="object")
    result = audit.classify_numeric(series)
    assert result["missing"].tolist() == [True, False, False, False, False, False, False]
    assert result["unparseable"].tolist() == [False, True, False, False, False, False, True]
    assert result["nonfinite"].tolist() == [False, False, True, False, False, False, False]
    assert result["zero"].tolist() == [False, False, False, True, False, False, False]
    assert result["positive"].tolist() == [False, False, False, False, True, False, False]
    assert result["negative"].tolist() == [False, False, False, False, False, True, False]


def test_four_application_states_cover_absent_missing_zero_and_nonzero():
    source_counts = np.array([0, 2, 2, 2], dtype=np.int32)
    finite_counts = np.array([0, 0, 1, 2], dtype=np.int32)
    nonzero_counts = np.array([0, 0, 0, 1], dtype=np.int32)
    states = audit.numeric_application_partition(source_counts, finite_counts, nonzero_counts)
    assert {name: np.flatnonzero(mask).tolist() for name, mask in states.items()} == {
        "NO_SOURCE_RECORD": [0],
        "SOURCE_PRESENT_NO_FINITE_OBSERVATION": [1],
        "FINITE_OBSERVATIONS_ZERO_ONLY": [2],
        "AT_LEAST_ONE_FINITE_NONZERO_OBSERVATION": [3],
    }


def test_multirow_partial_missingness_and_valid_zero_are_both_retained():
    series = pd.Series([np.nan, 0.0, 4.0])
    positions = np.array([0, 0, 1])
    matched = np.ones(3, dtype=bool)
    source_counts = np.array([2, 1], dtype=np.int32)
    _, result = audit.audit_numeric_field(
        "source", "amount_1A", series, positions, matched, source_counts,
        "Amount", "AD_ECONOMIC_CONTENT_PENDING_APPROVAL"
    )
    assert result["per_app"]["missing"].tolist() == [1, 0]
    assert result["per_app"]["zero"].tolist() == [1, 0]
    assert result["per_app"]["nonzero"].tolist() == [0, 1]
    assert result["partition"]["FINITE_OBSERVATIONS_ZERO_ONLY"].tolist() == [True, False]


def test_complete_overlap_patterns_union_without_double_counting():
    debit = np.array([True, True, False, False])
    deposit = np.array([True, False, True, False])
    rows = audit.complete_pattern_rows("TEST", ["debit", "deposit"], [debit, deposit])
    counts = {row["pattern"]: row["count"] for row in rows}
    assert counts == {"00": 1, "01": 1, "10": 1, "11": 1}
    assert sum(counts.values()) == 4


def test_other_debit_and_deposit_components_remain_independent():
    blank = np.zeros(3, dtype=bool)
    debit_component = {
        "content": np.array([True, False, False]),
        "nonzero": np.array([True, False, False]),
        "auxiliary_evidence": blank.copy(),
        "field_evidence": np.array([True, False, False]),
        "invalid": blank.copy(),
    }
    deposit_component = {
        "content": np.array([False, True, False]),
        "nonzero": np.array([False, True, False]),
        "auxiliary_evidence": blank.copy(),
        "field_evidence": np.array([False, True, False]),
        "invalid": blank.copy(),
    }
    debit = audit.module_masks([debit_component], 3)
    deposit = audit.module_masks([deposit_component], 3)
    assert debit["content"].tolist() == [True, False, False]
    assert deposit["content"].tolist() == [False, True, False]


def test_static_container_presence_does_not_create_tax_content():
    blank = np.zeros(2, dtype=bool)
    component = {
        "content": blank.copy(),
        "nonzero": blank.copy(),
        "auxiliary_evidence": np.array([True, False]),
        "field_evidence": np.array([True, False]),
        "invalid": blank.copy(),
    }
    result = audit.module_masks([component], 2)
    assert result["content"].tolist() == [False, False]
    assert result["auxiliary_only"].tolist() == [True, False]


def test_coverage_measure_names_map_to_internal_masks():
    data = {
        "content": np.array([True, False]),
        "no_content": np.array([False, True]),
        "invalid": np.array([False, True]),
        "auxiliary_evidence": np.array([True, True]),
    }
    assert audit.coverage_measure_mask(data, "numeric_content").tolist() == [True, False]
    assert audit.coverage_measure_mask(data, "no_numeric_content").tolist() == [False, True]
    assert audit.coverage_measure_mask(data, "invalid_numeric").tolist() == [False, True]
    assert audit.coverage_measure_mask(data, "auxiliary_only").tolist() == [False, True]
    module_data = {**data, "auxiliary_only": np.array([True, False])}
    assert audit.coverage_measure_mask(module_data, "auxiliary_only").tolist() == [True, False]


def test_structural_audit_counts_null_orphan_and_duplicate_history_keys():
    base_index = pd.Index([1, 2, 3])
    frame = pd.DataFrame(
        {
            "case_id": [1, 1, 4, None],
            "num_group1": [0, 0, 0, 1],
        }
    )
    result, positions, matched, orphan = audit.structural_audit(
        "history_1", frame, base_index, 1
    )
    assert result["null_case_id_rows"] == 1
    assert result["orphan_rows"] == 1
    assert result["matched_base_rows"] == 2
    assert result["duplicate_case_id_excess_rows"] == 1
    assert result["duplicate_case_num_group1_excess_rows"] == 1
    assert matched.tolist() == [True, True, False, False]
    assert orphan.tolist() == [False, False, True, False]


def test_shard_overlap_reports_conflicting_selected_values():
    frame = pd.DataFrame(
        {
            "case_id": [1, 2, 2, 3],
            "value_1A": [10.0, 20.0, 21.0, 30.0],
            "__file": ["a.parquet", "a.parquet", "b.parquet", "b.parquet"],
        }
    )
    result = audit.shard_overlap_diagnostics(frame, ["value_1A"])
    assert result[0]["overlapping_distinct_case_ids"] == 1
    assert result[0]["conflicting_selected_value_keys"] == 1


def test_date_parse_future_offset_and_conflicting_latest_tie():
    series = pd.Series(["2020-01-03", "bad", None])
    positions = np.array([0, 0, 1])
    matched = np.ones(3, dtype=bool)
    base_dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-02-01"]))
    rows, _ = audit.audit_date_field(
        "deposit_1", "contractenddate_991D", series, positions, matched,
        base_dates, "End date", "AD_AUXILIARY_DATE_PENDING_APPROVAL"
    )
    metrics = {row["metric_name"]: row["value"] for row in rows}
    assert metrics["row_date_parse_failure_count"] == 1
    assert metrics["date_offset_above_zero_count"] == 1

    frame = pd.DataFrame(
        {
            "case_id": [1, 1, 2],
            "date_1D": ["2020-01-02", "2020-01-02", None],
            "amount_1A": [5.0, 7.0, 1.0],
        }
    )
    ties = audit.latest_date_tie_diagnostics(
        frame, np.ones(3, dtype=bool), "date_1D", ["amount_1A"]
    )
    assert ties["applications_multiple_rows_at_latest_date"] == 1
    assert ties["applications_latest_tie_conflicting_selected_economic_values"] == 1
    assert ties["applications_all_dates_missing_or_invalid"] == 1


def test_full_base_flags_do_not_expand_and_reject_bad_lengths():
    base = pd.Series([10, 20, 30], dtype="int64")
    flags = audit.assemble_flags(
        base,
        {
            "source__x__present": np.array([True, False, True]),
            "source__x__record_count": np.array([1, 0, 2], dtype=np.int32),
        },
    )
    assert flags["case_id"].tolist() == [10, 20, 30]
    assert len(flags) == 3
    with pytest.raises(ValueError):
        audit.assemble_flags(base, {"bad": np.array([True])})
