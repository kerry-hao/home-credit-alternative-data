#!/usr/bin/env python3
"""Audit Home Credit train/test base parquet tables without modifying them."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WEEKLY_COLUMNS = [
    "WEEK_NUM",
    "week_group_is_missing",
    "application_count",
    "min_valid_date_decision",
    "max_valid_date_decision",
    "missing_invalid_date_count",
    "target_0_count",
    "target_1_count",
    "target_missing_count",
    "target_other_count",
    "valid_binary_label_count",
    "event_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--weekly-output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    return str(value)


def iso_timestamp(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def status(result: bool | None, **details: Any) -> dict[str, Any]:
    label = "NOT CHECKABLE" if result is None else ("PASS" if result else "FAIL")
    return {"status": label, **details}


def summarize_check_statuses(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize explicit validation status objects in an audit report."""
    valid_statuses = {"PASS", "FAIL", "NOT CHECKABLE"}
    informational_names = {
        "profile_completed",
        "availability",
        "missing_integer_week_numbers",
    }
    collected: list[tuple[str, str]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            status_value = value.get("status")
            path_name = path.rsplit(".", 1)[-1] if path else ""
            if status_value in valid_statuses:
                if path_name not in informational_names:
                    collected.append((path or "$", status_value))
                return
            for key in sorted(value):
                child_path = f"{path}.{key}" if path else str(key)
                visit(value[key], child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(report, "")
    failed_paths = sorted(path for path, value in collected if value == "FAIL")
    not_checkable_paths = sorted(
        path for path, value in collected if value == "NOT CHECKABLE"
    )
    performed_count = sum(value in {"PASS", "FAIL"} for _, value in collected)

    if failed_paths:
        validation_status = "FAIL"
    elif performed_count:
        validation_status = "PASS"
    else:
        validation_status = "NOT CHECKABLE"

    return {
        "execution_status": "COMPLETED",
        "validation_status": validation_status,
        "failed_check_paths": failed_paths,
        "not_checkable_check_paths": not_checkable_paths,
        "performed_validation_check_count": performed_count,
        "note": (
            "A validation PASS means that none of the performed, reported checks "
            "failed; it does not mean every possible issue was checked. "
            "NOT CHECKABLE checks are listed separately."
        ),
    }


def ensure_paths(args: argparse.Namespace) -> None:
    for input_path in (args.train, args.test):
        if not input_path.is_file():
            raise FileNotFoundError(f"Input does not exist or is not a file: {input_path}")

    input_resolved = {args.train.resolve(), args.test.resolve()}
    for output_path in (args.json_output, args.weekly_output):
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
        if output_path.resolve() in input_resolved:
            raise ValueError(f"Output path conflicts with an input: {output_path}")


def schema_profile(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "column_names": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "null_counts": {
            str(column): int(count) for column, count in frame.isna().sum().items()
        },
    }


def case_id_profile(frame: pd.DataFrame) -> dict[str, Any]:
    if "case_id" not in frame.columns:
        return {
            "profile_completed": status(None, reason="case_id column is absent"),
            "non_null_uniqueness_check": status(
                None, reason="case_id column is absent"
            ),
            "non_null_row_count": None,
            "missing_count": None,
            "distinct_non_null_case_ids": None,
            "distinct_non_null_case_ids_occurring_more_than_once": None,
            "excess_rows_beyond_one_per_distinct_non_null_case_id": None,
        }

    series = frame["case_id"]
    non_null = series.dropna()
    counts = non_null.value_counts(dropna=False)
    distinct = int(counts.size)
    repeated_distinct = int((counts > 1).sum())
    excess_rows = int((counts - 1).clip(lower=0).sum())
    if non_null.empty:
        uniqueness_check = status(
            None, reason="case_id exists but contains no non-null values"
        )
    else:
        uniqueness_check = status(repeated_distinct == 0)
    return {
        "profile_completed": status(True),
        "non_null_uniqueness_check": uniqueness_check,
        "non_null_row_count": int(non_null.size),
        "missing_count": int(series.isna().sum()),
        "distinct_non_null_case_ids": distinct,
        "distinct_non_null_case_ids_occurring_more_than_once": repeated_distinct,
        "excess_rows_beyond_one_per_distinct_non_null_case_id": excess_rows,
        "non_null_case_ids_unique": repeated_distinct == 0,
    }


def exact_duplicate_profile(frame: pd.DataFrame) -> dict[str, int]:
    repeated_after_first = frame.duplicated(keep="first")
    all_participants = frame.duplicated(keep=False)
    participant_frame = frame.loc[all_participants]
    return {
        "excess_exact_duplicate_rows_after_first_occurrence": int(
            repeated_after_first.sum()
        ),
        "rows_participating_in_an_exact_duplicate_group": int(
            all_participants.sum()
        ),
        "distinct_exact_duplicate_row_patterns": int(
            participant_frame.drop_duplicates().shape[0]
        ),
    }


def target_counts(series: pd.Series) -> dict[str, int]:
    missing_mask = series.isna()
    zero_mask = series.eq(0).fillna(False)
    one_mask = series.eq(1).fillna(False)
    other_mask = ~(missing_mask | zero_mask | one_mask)
    return {
        "target_0_count": int(zero_mask.sum()),
        "target_1_count": int(one_mask.sum()),
        "target_missing_count": int(missing_mask.sum()),
        "target_other_count": int(other_mask.sum()),
        "valid_binary_label_count": int((zero_mask | one_mask).sum()),
    }


def target_profile(frame: pd.DataFrame) -> dict[str, Any]:
    if "target" not in frame.columns:
        return {
            "exists": False,
            "strict_binary_0_1_no_missing": status(
                None, reason="target column is absent"
            ),
            "counts": None,
            "value_distribution": None,
        }

    series = frame["target"]
    counts = target_counts(series)
    distribution = []
    for value, count in series.value_counts(dropna=False, sort=False).items():
        distribution.append(
            {
                "value": json_scalar(value),
                "python_type": "missing" if pd.isna(value) else type(value).__name__,
                "count": int(count),
            }
        )
    strict = counts["target_missing_count"] == 0 and counts["target_other_count"] == 0
    return {
        "exists": True,
        "dtype": str(series.dtype),
        "strict_binary_0_1_no_missing": status(
            strict,
            allowed_values=[0, 1],
            note=(
                "No values were recoded or dropped."
                if strict
                else "Non-binary or missing labels were retained and flagged."
            ),
        ),
        "counts": counts,
        "value_distribution": distribution,
    }


def parse_decision_dates(frame: pd.DataFrame) -> tuple[pd.Series | None, dict[str, Any]]:
    if "date_decision" not in frame.columns:
        return None, {
            "exists": False,
            "check": status(None, reason="date_decision column is absent"),
        }

    original = frame["date_decision"]
    parsed = pd.to_datetime(original, errors="coerce")
    failures = original.notna() & parsed.isna()
    valid = parsed.notna()
    report = {
        "exists": True,
        "original_dtype": str(original.dtype),
        "original_non_null_count": int(original.notna().sum()),
        "original_missing_count": int(original.isna().sum()),
        "parsing_failure_count_among_non_null": int(failures.sum()),
        "valid_parsed_count": int(valid.sum()),
        "observed_valid_min": iso_timestamp(parsed.loc[valid].min()) if valid.any() else None,
        "observed_valid_max": iso_timestamp(parsed.loc[valid].max()) if valid.any() else None,
        "preservation_note": (
            "Parsing was performed in a separate series for audit calculations; "
            "the original field and input file were not changed."
        ),
    }
    return parsed, report


def range_profile(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame.columns:
        return {"exists": False, "check": status(None, reason=f"{column} is absent")}
    series = frame[column]
    non_null = series.dropna()
    try:
        observed_min = json_scalar(non_null.min()) if len(non_null) else None
        observed_max = json_scalar(non_null.max()) if len(non_null) else None
        ordered = True
    except TypeError:
        observed_min = None
        observed_max = None
        ordered = False
    return {
        "exists": True,
        "dtype": str(series.dtype),
        "missing_count": int(series.isna().sum()),
        "distinct_non_null_count": int(non_null.nunique(dropna=True)),
        "observed_non_null_min": observed_min,
        "observed_non_null_max": observed_max,
        "min_max_order_comparable": ordered,
    }


def numeric_characteristics(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    if non_null.empty:
        return {
            "numerically_sortable": False,
            "integer_valued": False,
            "numeric_values": None,
        }
    numeric = pd.to_numeric(non_null, errors="coerce")
    numeric_array = numeric.to_numpy(dtype=float)
    sortable = bool(np.isfinite(numeric_array).all())
    integer_valued = sortable and bool(np.equal(numeric_array, np.floor(numeric_array)).all())
    return {
        "numerically_sortable": sortable,
        "integer_valued": integer_valued,
        "numeric_values": numeric if sortable else None,
    }


def ordered_non_null_values(series: pd.Series) -> tuple[list[Any], dict[str, Any]]:
    values = list(pd.unique(series.dropna()))
    characteristics = numeric_characteristics(series)
    if characteristics["numerically_sortable"]:
        values.sort(key=lambda value: float(pd.to_numeric(value)))
    else:
        values.sort(key=lambda value: (type(value).__name__, str(value)))
    return values, characteristics


def value_mask(series: pd.Series, value: Any) -> pd.Series:
    return series.eq(value).fillna(False)


def weekly_summary(
    frame: pd.DataFrame, parsed_dates: pd.Series | None
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    if "WEEK_NUM" not in frame.columns:
        empty = pd.DataFrame(columns=WEEKLY_COLUMNS)
        return (
            empty,
            {
                "availability": status(None, reason="WEEK_NUM column is absent"),
                "weekly_application_reconciliation": status(
                    None, reason="WEEK_NUM column is absent"
                ),
                "weekly_target_reconciliation": status(
                    None, reason="WEEK_NUM column is absent"
                ),
            },
            [],
        )

    week_series = frame["WEEK_NUM"]
    week_values, characteristics = ordered_non_null_values(week_series)
    groups: list[tuple[Any, pd.Series, bool]] = [
        (value, value_mask(week_series, value), False) for value in week_values
    ]
    if week_series.isna().any():
        groups.append((None, week_series.isna(), True))

    rows: list[dict[str, Any]] = []
    internal_ranges: list[dict[str, Any]] = []
    target_exists = "target" in frame.columns

    for week_value, mask, is_missing_group in groups:
        group_size = int(mask.sum())
        if parsed_dates is None:
            valid_dates = None
            valid_min = None
            valid_max = None
            missing_invalid = group_size
        else:
            valid_dates = parsed_dates.loc[mask].dropna()
            valid_min = valid_dates.min() if not valid_dates.empty else None
            valid_max = valid_dates.max() if not valid_dates.empty else None
            missing_invalid = group_size - int(valid_dates.size)

        if target_exists:
            counts = target_counts(frame.loc[mask, "target"])
            denominator = counts["valid_binary_label_count"]
            event_rate = (
                counts["target_1_count"] / denominator if denominator else None
            )
        else:
            counts = {
                "target_0_count": None,
                "target_1_count": None,
                "target_missing_count": None,
                "target_other_count": None,
                "valid_binary_label_count": None,
            }
            event_rate = None

        output_week = json_scalar(week_value)
        if characteristics["integer_valued"] and week_value is not None:
            output_week = int(float(pd.to_numeric(week_value)))
        row = {
            "WEEK_NUM": output_week,
            "week_group_is_missing": is_missing_group,
            "application_count": group_size,
            "min_valid_date_decision": iso_timestamp(valid_min),
            "max_valid_date_decision": iso_timestamp(valid_max),
            "missing_invalid_date_count": missing_invalid,
            **counts,
            "event_rate": event_rate,
        }
        rows.append(row)
        if not is_missing_group:
            internal_ranges.append(
                {
                    "week": output_week,
                    "sort_value": (
                        float(pd.to_numeric(week_value))
                        if characteristics["numerically_sortable"]
                        else None
                    ),
                    "min": valid_min,
                    "max": valid_max,
                }
            )

    summary = pd.DataFrame(rows, columns=WEEKLY_COLUMNS)
    application_total = int(summary["application_count"].sum()) if len(summary) else 0
    application_reconciles = application_total == len(frame)

    if target_exists:
        overall = target_counts(frame["target"])
        weekly_totals = {
            key: int(summary[key].sum())
            for key in (
                "target_0_count",
                "target_1_count",
                "target_missing_count",
                "target_other_count",
                "valid_binary_label_count",
            )
        }
        target_reconciles = weekly_totals == overall
        target_reconciliation = status(
            target_reconciles,
            overall_counts=overall,
            weekly_summed_counts=weekly_totals,
        )
    else:
        target_reconciliation = status(None, reason="target column is absent")

    checks = {
        "availability": status(True),
        "weekly_application_reconciliation": status(
            application_reconciles,
            train_row_count=int(len(frame)),
            weekly_application_count_sum=application_total,
        ),
        "weekly_target_reconciliation": target_reconciliation,
        "week_sorting": {
            "numeric_sorting_applied": characteristics["numerically_sortable"],
            "missing_group_placed_last": True,
        },
    }
    return summary, checks, internal_ranges


def missing_integer_weeks(
    week_series: pd.Series,
) -> dict[str, Any]:
    characteristics = numeric_characteristics(week_series)
    if not characteristics["integer_valued"]:
        return status(
            None,
            reason="WEEK_NUM is absent, empty, non-numeric, non-finite, or not integer-valued",
        )

    values = sorted(
        {int(value) for value in characteristics["numeric_values"].to_numpy(dtype=float)}
    )
    missing: list[int] = []
    for left, right in zip(values, values[1:]):
        if right > left + 1:
            missing.extend(range(left + 1, right))
    return status(
        True,
        observed_min=values[0],
        observed_max=values[-1],
        missing_integer_week_count=len(missing),
        missing_integer_week_numbers=missing,
        interpretation=(
            "Missing integer week numbers are reported descriptively and are not "
            "automatically treated as data errors or ISO calendar weeks. PASS means "
            "the descriptive enumeration completed; it does not imply that zero "
            "integer weeks are missing."
        ),
    )


def temporal_checks(
    frame: pd.DataFrame,
    parsed_dates: pd.Series | None,
    internal_ranges: list[dict[str, Any]],
) -> dict[str, Any]:
    if "WEEK_NUM" not in frame.columns:
        absent = status(None, reason="WEEK_NUM column is absent")
        return {
            "valid_date_maps_to_single_week": absent,
            "dates_move_chronologically_as_week_increases": absent,
            "adjacent_observed_week_date_ranges": absent,
            "missing_integer_week_numbers": absent,
        }

    week_series = frame["WEEK_NUM"]
    if parsed_dates is None or not parsed_dates.notna().any():
        date_mapping = status(None, reason="No valid decision dates are available")
    else:
        mapping_frame = pd.DataFrame(
            {
                "date_decision_parsed": parsed_dates,
                "WEEK_NUM": week_series,
            }
        ).loc[parsed_dates.notna()]
        distinct_weeks = mapping_frame.groupby("date_decision_parsed", dropna=False)[
            "WEEK_NUM"
        ].nunique(dropna=False)
        conflict_count = int((distinct_weeks > 1).sum())
        date_mapping = status(
            conflict_count == 0,
            valid_distinct_decision_date_count=int(distinct_weeks.size),
            decision_dates_mapping_to_multiple_week_values=conflict_count,
        )

    characteristics = numeric_characteristics(week_series)
    valid_ranges = [
        item for item in internal_ranges if item["min"] is not None and item["max"] is not None
    ]
    missing_range_count = len(internal_ranges) - len(valid_ranges)
    if not characteristics["numerically_sortable"]:
        chronological = status(
            None, reason="WEEK_NUM values cannot all be sorted numerically"
        )
        adjacent = status(None, reason="WEEK_NUM values cannot all be sorted numerically")
    elif len(valid_ranges) < 2:
        chronological = status(
            None, reason="Fewer than two non-missing week groups have valid date ranges"
        )
        adjacent = status(
            None, reason="Fewer than two non-missing week groups have valid date ranges"
        )
    else:
        valid_ranges.sort(key=lambda item: item["sort_value"])
        min_backward = []
        max_backward = []
        overlaps = []
        any_backward = []
        for previous, current in zip(valid_ranges, valid_ranges[1:]):
            pair = [previous["week"], current["week"]]
            if current["min"] < previous["min"]:
                min_backward.append(pair)
            if current["max"] < previous["max"]:
                max_backward.append(pair)
            if current["min"] <= previous["max"]:
                overlaps.append(pair)
            if current["min"] < previous["min"] or current["max"] < previous["max"]:
                any_backward.append(pair)

        chronological = status(
            not min_backward and not max_backward,
            adjacent_observed_week_pairs_compared=len(valid_ranges) - 1,
            week_groups_without_valid_date_range=missing_range_count,
            pairs_with_backward_min_date=min_backward,
            pairs_with_backward_max_date=max_backward,
        )
        adjacent = status(
            not overlaps and not any_backward,
            adjacent_definition="successive observed non-missing WEEK_NUM groups in numeric order",
            adjacent_observed_week_pairs_compared=len(valid_ranges) - 1,
            overlapping_date_range_pairs=overlaps,
            backward_date_range_pairs=any_backward,
            note="Overlap/backward findings are reported without repairing or reclassifying the data.",
        )

    return {
        "valid_date_maps_to_single_week": date_mapping,
        "dates_move_chronologically_as_week_increases": chronological,
        "adjacent_observed_week_date_ranges": adjacent,
        "missing_integer_week_numbers": missing_integer_weeks(week_series),
    }


def month_date_mapping(
    frame: pd.DataFrame, parsed_dates: pd.Series | None
) -> dict[str, Any]:
    explanation = (
        "MONTH is treated only as an observed code. The mappings below compare it "
        "descriptively with calendar year-month values derived from valid "
        "date_decision values; no encoding is assumed."
    )
    if "MONTH" not in frame.columns:
        return {
            "availability": status(None, reason="MONTH column is absent"),
            "explanation": explanation,
            "mappings": [],
        }
    if parsed_dates is None or not parsed_dates.notna().any():
        return {
            "availability": status(None, reason="No valid decision dates are available"),
            "explanation": explanation,
            "mappings": [],
        }

    month_series = frame["MONTH"]
    month_values, characteristics = ordered_non_null_values(month_series)
    groups: list[tuple[Any, pd.Series, bool]] = [
        (value, value_mask(month_series, value), False) for value in month_values
    ]
    if month_series.isna().any():
        groups.append((None, month_series.isna(), True))

    mappings = []
    for month_value, mask, is_missing in groups:
        valid_dates = parsed_dates.loc[mask].dropna()
        calendar_values = sorted(valid_dates.dt.strftime("%Y-%m").unique().tolist())
        output_month = json_scalar(month_value)
        if characteristics["integer_valued"] and month_value is not None:
            output_month = int(float(pd.to_numeric(month_value)))
        mappings.append(
            {
                "MONTH": output_month,
                "month_group_is_missing": is_missing,
                "application_count": int(mask.sum()),
                "min_valid_date_decision": iso_timestamp(
                    valid_dates.min() if not valid_dates.empty else None
                ),
                "max_valid_date_decision": iso_timestamp(
                    valid_dates.max() if not valid_dates.empty else None
                ),
                "missing_invalid_date_count": int(mask.sum()) - int(valid_dates.size),
                "distinct_calendar_year_month_count": len(calendar_values),
                "calendar_year_month_values": calendar_values,
            }
        )

    nonmissing_mappings = [item for item in mappings if not item["month_group_is_missing"]]
    month_to_one_calendar = all(
        item["distinct_calendar_year_month_count"] <= 1 for item in nonmissing_mappings
    )

    valid_frame = pd.DataFrame(
        {
            "calendar_year_month": parsed_dates.dt.strftime("%Y-%m"),
            "MONTH": month_series,
        }
    ).loc[parsed_dates.notna()]
    calendar_to_month_counts = valid_frame.groupby("calendar_year_month")["MONTH"].nunique(
        dropna=False
    )
    calendar_to_one_month = bool((calendar_to_month_counts <= 1).all())

    return {
        "availability": status(True),
        "explanation": explanation,
        "each_nonmissing_MONTH_maps_to_at_most_one_calendar_year_month": status(
            month_to_one_calendar
        ),
        "each_calendar_year_month_maps_to_at_most_one_MONTH_value": status(
            calendar_to_one_month,
            distinct_calendar_year_month_count=int(calendar_to_month_counts.size),
            calendar_year_months_mapping_to_multiple_MONTH_values=int(
                (calendar_to_month_counts > 1).sum()
            ),
        ),
        "mappings": mappings,
    }


def case_id_overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    if "case_id" not in train.columns or "case_id" not in test.columns:
        return status(
            None,
            reason="case_id must exist in both train and test to check overlap",
            overlapping_distinct_non_null_case_id_count=None,
        )
    train_ids = pd.Index(train["case_id"].dropna().unique())
    test_ids = pd.Index(test["case_id"].dropna().unique())
    overlap_count = int(train_ids.intersection(test_ids).size)
    return status(
        overlap_count == 0,
        overlapping_distinct_non_null_case_id_count=overlap_count,
        identifiers_printed=False,
    )


def main() -> int:
    args = parse_args()
    ensure_paths(args)

    before_hashes = {
        "train": sha256_file(args.train),
        "test": sha256_file(args.test),
    }
    input_sizes = {
        "train": int(args.train.stat().st_size),
        "test": int(args.test.stat().st_size),
    }

    train = pd.read_parquet(args.train)
    test = pd.read_parquet(args.test)

    train_dates, train_date_report = parse_decision_dates(train)
    weekly, weekly_checks, internal_ranges = weekly_summary(train, train_dates)
    train_target = target_profile(train)

    report: dict[str, Any] = {
        "audit_scope": {
            "train_input": str(args.train),
            "test_input": str(args.test),
            "json_output": str(args.json_output),
            "weekly_output": str(args.weekly_output),
            "rows_audited": "all rows in both inputs",
            "individual_records_or_identifiers_output": False,
        },
        "input_integrity": {
            "train": {
                "path": str(args.train),
                "size_bytes": input_sizes["train"],
                "sha256_before": before_hashes["train"],
            },
            "test": {
                "path": str(args.test),
                "size_bytes": input_sizes["test"],
                "sha256_before": before_hashes["test"],
            },
        },
        "train_base": {
            **schema_profile(train),
            "case_id": case_id_profile(train),
            "target": train_target,
            "date_decision": train_date_report,
            "WEEK_NUM": range_profile(train, "WEEK_NUM"),
            "MONTH": range_profile(train, "MONTH"),
            "exact_duplicate_rows": exact_duplicate_profile(train),
        },
        "weekly_temporal_summary": {
            "row_count": int(len(weekly)),
            "columns": WEEKLY_COLUMNS,
            "checks": weekly_checks,
            "temporal_checks": temporal_checks(train, train_dates, internal_ranges),
        },
        "month_to_decision_date_mapping": month_date_mapping(train, train_dates),
        "test_base": {
            **schema_profile(test),
            "case_id": case_id_profile(test),
            "target_exists": "target" in test.columns,
        },
        "train_test_case_id_overlap": case_id_overlap(train, test),
        "interpretation_boundaries": [
            "The label is described neutrally as an event; its formal definition and horizon remain unverified.",
            "WEEK_NUM is not assumed to be an ISO calendar week.",
            "Missing calendar days or integer week numbers are not automatically classified as data errors.",
            "MONTH encoding is not inferred from the observed aggregate correspondence.",
            "No borrower identity is inferred from case_id uniqueness or overlap alone.",
            "No temporal split, group cutoff, or policy threshold is selected by this audit.",
        ],
    }

    after_hashes = {
        "train": sha256_file(args.train),
        "test": sha256_file(args.test),
    }
    integrity_pass = before_hashes == after_hashes
    for name in ("train", "test"):
        report["input_integrity"][name]["sha256_after"] = after_hashes[name]
        report["input_integrity"][name]["unchanged"] = (
            before_hashes[name] == after_hashes[name]
        )
    report["input_integrity"]["overall_check"] = status(integrity_pass)
    audit_summary = summarize_check_statuses(report)
    report["audit_summary"] = audit_summary

    if not integrity_pass:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "reason": "An input SHA-256 hash changed during the audit; outputs were not written.",
                    "before": before_hashes,
                    "after": after_hashes,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.weekly_output.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.weekly_output, index=False, date_format="%Y-%m-%d")
    args.json_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                **audit_summary,
                "json_output": str(args.json_output),
                "weekly_output": str(args.weekly_output),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "weekly_rows": int(len(weekly)),
                "input_hashes_unchanged": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
