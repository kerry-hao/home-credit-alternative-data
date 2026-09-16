import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_base.py"
SPEC = importlib.util.spec_from_file_location("audit_base", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit_base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_base)


def test_unique_non_null_ids():
    profile = audit_base.case_id_profile(pd.DataFrame({"case_id": [1, 2, 3]}))

    assert profile["profile_completed"]["status"] == "PASS"
    assert profile["non_null_uniqueness_check"]["status"] == "PASS"
    assert profile["distinct_non_null_case_ids_occurring_more_than_once"] == 0
    assert profile["excess_rows_beyond_one_per_distinct_non_null_case_id"] == 0


def test_repeated_non_null_ids():
    profile = audit_base.case_id_profile(
        pd.DataFrame({"case_id": [1, 1, 2, 2, 2, 3]})
    )

    assert profile["profile_completed"]["status"] == "PASS"
    assert profile["non_null_uniqueness_check"]["status"] == "FAIL"
    assert profile["distinct_non_null_case_ids_occurring_more_than_once"] == 2
    assert profile["excess_rows_beyond_one_per_distinct_non_null_case_id"] == 3


def test_missing_case_id_column():
    profile = audit_base.case_id_profile(pd.DataFrame({"other": [1]}))

    assert profile["non_null_uniqueness_check"]["status"] == "NOT CHECKABLE"
    assert profile["non_null_uniqueness_check"]["reason"]


def test_all_case_id_values_missing():
    profile = audit_base.case_id_profile(
        pd.DataFrame({"case_id": [None, pd.NA, np.nan]})
    )

    assert profile["profile_completed"]["status"] == "PASS"
    assert profile["non_null_uniqueness_check"]["status"] == "NOT CHECKABLE"
    assert profile["missing_count"] == 3
    assert profile["non_null_row_count"] == 0


def test_summary_with_failed_validation_check():
    summary = audit_base.summarize_check_statuses(
        {"checks": {"failed_check": audit_base.status(False)}}
    )

    assert summary["execution_status"] == "COMPLETED"
    assert summary["validation_status"] == "FAIL"
    assert summary["failed_check_paths"] == ["checks.failed_check"]


def test_summary_with_pass_and_not_checkable_checks():
    summary = audit_base.summarize_check_statuses(
        {
            "checks": {
                "pass_check": audit_base.status(True),
                "unknown_check": audit_base.status(None, reason="synthetic"),
            }
        }
    )

    assert summary["validation_status"] == "PASS"
    assert summary["not_checkable_check_paths"] == ["checks.unknown_check"]
    assert summary["performed_validation_check_count"] == 1


def test_summary_with_only_not_checkable_checks():
    summary = audit_base.summarize_check_statuses(
        {"checks": {"unknown_check": audit_base.status(None, reason="synthetic")}}
    )

    assert summary["validation_status"] == "NOT CHECKABLE"
    assert summary["performed_validation_check_count"] == 0


def test_informational_statuses_and_plain_booleans_are_excluded():
    summary = audit_base.summarize_check_statuses(
        {
            "profile_completed": audit_base.status(True),
            "section": {
                "availability": audit_base.status(False),
                "missing_integer_week_numbers": audit_base.status(
                    True, missing_integer_week_numbers=[1]
                ),
                "plain_observation": False,
            },
        }
    )

    assert summary["validation_status"] == "NOT CHECKABLE"
    assert summary["performed_validation_check_count"] == 0
    assert summary["failed_check_paths"] == []
    assert summary["not_checkable_check_paths"] == []


def test_missing_week_enumeration_is_descriptive():
    result = audit_base.missing_integer_weeks(pd.Series([0, 2]))

    assert result["status"] == "PASS"
    assert result["missing_integer_week_numbers"] == [1]
    assert "PASS means the descriptive enumeration completed" in result["interpretation"]
    assert "does not imply that zero integer weeks are missing" in result["interpretation"]


def test_weekly_numeric_regression_and_reconciliation():
    frame = pd.DataFrame(
        {
            "WEEK_NUM": [0, 0, 0, 1, 1],
            "target": [0, 1, 1, 0, 0],
        }
    )
    dates = pd.Series(pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-08"] * 2))

    weekly, checks, _ = audit_base.weekly_summary(frame, dates)

    assert weekly["application_count"].tolist() == [3, 2]
    assert weekly["target_0_count"].tolist() == [1, 2]
    assert weekly["target_1_count"].tolist() == [2, 0]
    assert weekly["event_rate"].tolist() == [2 / 3, 0.0]
    assert checks["weekly_application_reconciliation"]["status"] == "PASS"
    assert checks["weekly_target_reconciliation"]["status"] == "PASS"


def test_weekly_zero_valid_label_denominator_is_missing():
    frame = pd.DataFrame({"WEEK_NUM": [0, 0], "target": [2, np.nan]})
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"]))

    weekly, _, _ = audit_base.weekly_summary(frame, dates)

    assert weekly.loc[0, "valid_binary_label_count"] == 0
    assert pd.isna(weekly.loc[0, "event_rate"])


def test_summary_is_json_serializable_and_paths_are_sorted():
    summary = audit_base.summarize_check_statuses(
        {
            "z": {
                "unknown": audit_base.status(None, reason="z"),
                "failed": audit_base.status(False),
            },
            "a": {
                "unknown": audit_base.status(None, reason="a"),
                "failed": audit_base.status(False),
            },
        }
    )

    json.dumps(summary)
    assert summary["failed_check_paths"] == ["a.failed", "z.failed"]
    assert summary["not_checkable_check_paths"] == ["a.unknown", "z.unknown"]
