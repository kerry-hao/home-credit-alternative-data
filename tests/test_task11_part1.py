from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_thin_file_candidates.py"
SPEC = importlib.util.spec_from_file_location("task11_part1", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exact_candidate_list() -> None:
    assert MODULE.ACCOUNT_CANDIDATES == ["credquantity_1099L", "t__observed_active_credit_count"]


def test_consistent_count_preserves_zero_missing_and_conflict() -> None:
    train = pd.Series([1, 2, 3, 4, 5])
    source = pd.DataFrame({"case_id": [1, 1, 2, 3, 3, 5], "value": [0, 0, np.nan, 1, 2, 4]})
    values, details = MODULE.consistent_application_count(train, source, "value")
    assert values.tolist()[:1] == [0.0]
    assert values.iloc[1:4].isna().all()
    assert values.iloc[4] == 4.0
    assert details["source_present_no_finite"] == 1
    assert details["conflicting_finite_values"] == 1


def _date_aggregate(name: str, train: pd.DataFrame, source: pd.DataFrame, column: str):
    index = pd.Index(train.case_id, name="case_id")
    decisions = pd.to_datetime(train.date_decision).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    aggregate = MODULE.empty_date_aggregate(name, len(train))
    MODULE.update_date_aggregate(aggregate, source, column, index, decisions)
    counts = source.case_id.value_counts().reindex(index, fill_value=0).to_numpy()
    aggregate.unique_source_case_ids = int(np.count_nonzero(counts))
    aggregate.duplicate_source_case_ids = int(np.count_nonzero(counts > 1))
    aggregate.duplicate_source_rows = int(np.maximum(counts - 1, 0).sum())
    return aggregate, decisions


def test_earliest_valid_date_across_active_and_closed() -> None:
    train = pd.DataFrame({"case_id": [1, 2], "date_decision": ["2020-01-10", "2020-01-10"]})
    active, decisions = _date_aggregate("active", train, pd.DataFrame({"case_id": [1, 2], "d": ["2020-01-05", "2020-01-08"]}), "d")
    closed, _ = _date_aggregate("closed", train, pd.DataFrame({"case_id": [1, 2], "d": ["2019-01-01", "2020-01-03"]}), "d")
    result = MODULE.combine_bureau_history(decisions, active, closed)
    assert result["days"].tolist() == [374.0, 7.0]


def test_after_decision_excluded_and_reported() -> None:
    train = pd.DataFrame({"case_id": [1], "date_decision": ["2020-01-10"]})
    active, decisions = _date_aggregate("active", train, pd.DataFrame({"case_id": [1], "d": ["2020-01-11"]}), "d")
    closed = MODULE.empty_date_aggregate("closed", 1)
    result = MODULE.combine_bureau_history(decisions, active, closed)
    assert active.after_rows == 1
    assert not result["valid"][0]
    assert result["negative_before_exclusion"][0]


def test_missing_preserved_when_no_valid_date() -> None:
    train = pd.DataFrame({"case_id": [1], "date_decision": ["2020-01-10"]})
    active, decisions = _date_aggregate("active", train, pd.DataFrame({"case_id": [1], "d": [None]}), "d")
    result = MODULE.combine_bureau_history(decisions, active, MODULE.empty_date_aggregate("closed", 1))
    assert not result["valid"][0]
    assert np.isnan(result["days"][0])
    assert np.isnan(result["years"][0])


def test_year_conversion_uses_365_25() -> None:
    train = pd.DataFrame({"case_id": [1], "date_decision": ["2020-01-01"]})
    active, decisions = _date_aggregate("active", train, pd.DataFrame({"case_id": [1], "d": ["2019-01-01"]}), "d")
    result = MODULE.combine_bureau_history(decisions, active, MODULE.empty_date_aggregate("closed", 1))
    assert result["days"][0] == 365.0
    assert result["years"][0] == 365.0 / 365.25


def test_joint_coverage_reconciles_and_preserves_zero_missing() -> None:
    values = pd.Series([0.0, 2.0, np.nan, np.nan])
    history = np.array([True, False, True, False])
    rows = MODULE.joint_coverage("candidate", values, history, 4)
    lookup = {row["coverage_category"]: row["count"] for row in rows}
    assert sum(lookup[name] for name in ("both_valid", "count_valid_history_missing", "count_missing_history_valid", "both_missing")) == 4
    assert lookup["count_zero_history_valid"] == 1
    assert lookup["count_positive_history_missing"] == 1


def test_output_inventory_contains_no_chart_types() -> None:
    forbidden = {".png", ".pdf", ".html", ".ipynb"}
    assert not any(Path(name).suffix in forbidden for name in MODULE.OUTPUT_NAMES)


def test_output_boundary_rejects_protected_directory(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    protected = data_root / "audits/task10/first_full_v3/task11"
    try:
        MODULE.run(data_root, protected, tmp_path)
    except MODULE.AuditBlocked as exc:
        assert "protected" in str(exc).lower()
    else:
        raise AssertionError("Protected Task 8-10 output path was not rejected")


def test_script_never_reads_target_or_creates_split() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'columns=["case_id", "base_order", "outer_split", "validation_role"]' in source
    assert '"target_loaded": False' in source
    assert "train_test_split" not in source
    assert "make_splits" not in source


def test_script_refuses_existing_output_and_hashes_protected_files() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Refusing to overwrite existing Task 11 Part 1 output" in source
    assert "hash_protected(protected_paths(inputs))" in source
    assert "protected_hashes_unchanged" in source


def test_final_audit_integrity_contract() -> None:
    rows = MODULE.case_integrity_rows(
        pd.DataFrame({"case_id": [1, 2]}), 2, 2,
        {"source_rows": 0, "unique_source_case_ids": 0, "duplicate_source_case_ids": 0,
         "duplicate_source_rows": 0, "conflicting_finite_values": 0, "source_present_no_finite": 0},
        MODULE.empty_date_aggregate("active", 2), MODULE.empty_date_aggregate("closed", 2),
    )
    lookup = {(row["source"], row["metric"]): row["value"] for row in rows}
    assert lookup[("final_audit_table", "row_count")] == 2
    assert lookup[("final_audit_table", "unique_case_id_count")] == 2
    assert lookup[("final_audit_table", "join_expanded_one_row_per_application")] is False
