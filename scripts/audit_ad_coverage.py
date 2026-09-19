#!/usr/bin/env python3
"""Audit Batch 1 alternative-data coverage and record structure.

This script reads only the explicitly scoped TRAIN columns.  It produces
aggregate audit reports plus a base-keyed audit-flags table; it does not build
model features, use the target, or make T/AD eligibility decisions.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow
import pyarrow as pa
import pyarrow.parquet as pq


TASK_ID = "TASK_07_BATCH1"
SCRIPT_VERSION = "1.0.0"
QUANTILE_METHOD = "linear"
FLOAT_EQUAL_RTOL = 1e-9
FLOAT_EQUAL_ATOL = 1e-12

OUTPUT_NAMES = (
    "source_structure.csv",
    "ad_coverage.csv",
    "ad_overlap.csv",
    "field_value_audit.csv",
    "coverage_by_month.csv",
    "feature_review.csv",
    "application_ad_audit_flags.parquet",
    "audit_summary.json",
    "audit_report.md",
)

CONTACT_FIELDS = (
    "clientscnt_1022L",
    "clientscnt_304L",
    "clientscnt3m_3712950L",
    "clientscnt6m_3712949L",
    "clientscnt12m_3712952L",
    "applicationcnt_361L",
    "applicationscnt_867L",
    "clientscnt_136L",
    "mobilephncnt_593L",
    "sellerplacescnt_216L",
)
TAX_SUMMARY_FIELDS = (
    "pmtscount_423L",
    "pmtssum_45A",
    "pmtcount_693L",
    "pmtcount_4527229L",
    "pmtcount_4955617L",
    "pmtaverage_3A",
    "pmtaverage_4527227A",
    "pmtaverage_4955615A",
)
COUNT_FIELDS = set(CONTACT_FIELDS) | {
    "pmtscount_423L",
    "pmtcount_693L",
    "pmtcount_4527229L",
    "pmtcount_4955617L",
}

SOURCE_SPECS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "base",
            {
                "task06_group": "train_base",
                "depth": 0,
                "numeric": (),
                "dates": ("date_decision",),
                "text": (),
                "structural": ("case_id", "MONTH", "WEEK_NUM"),
            },
        ),
        (
            "static_0",
            {
                "task06_group": "train_static_0",
                "depth": 0,
                "numeric": CONTACT_FIELDS,
                "dates": (),
                "text": (),
                "structural": ("case_id",),
            },
        ),
        (
            "static_cb_0",
            {
                "task06_group": "train_static_cb_0",
                "depth": 0,
                "numeric": TAX_SUMMARY_FIELDS,
                "dates": (
                    "assignmentdate_238D",
                    "assignmentdate_4527235D",
                    "assignmentdate_4955616D",
                    "responsedate_1012D",
                    "responsedate_4527233D",
                    "responsedate_4917613D",
                ),
                "text": ("requesttype_4525192L",),
                "structural": ("case_id",),
            },
        ),
        (
            "debitcard_1",
            {
                "task06_group": "train_debitcard_1",
                "depth": 1,
                "numeric": (
                    "last30dayturnover_651A",
                    "last180dayturnover_1134A",
                    "last180dayaveragebalance_704A",
                ),
                "dates": ("openingdate_857D",),
                "text": (),
                "structural": ("case_id", "num_group1"),
            },
        ),
        (
            "deposit_1",
            {
                "task06_group": "train_deposit_1",
                "depth": 1,
                "numeric": ("amount_416A",),
                "dates": ("openingdate_313D", "contractenddate_991D"),
                "text": (),
                "structural": ("case_id", "num_group1"),
            },
        ),
        (
            "other_1",
            {
                "task06_group": "train_other_1",
                "depth": 1,
                "numeric": (
                    "amtdebitincoming_4809443A",
                    "amtdebitoutgoing_4809440A",
                    "amtdepositbalance_4809441A",
                    "amtdepositincoming_4809444A",
                    "amtdepositoutgoing_4809442A",
                ),
                "dates": (),
                "text": (),
                "structural": ("case_id", "num_group1"),
            },
        ),
        (
            "tax_registry_a_1",
            {
                "task06_group": "train_tax_registry_a_1",
                "depth": 1,
                "numeric": ("amount_4527230A",),
                "dates": ("recorddate_4527225D",),
                "text": (),
                "structural": ("case_id", "num_group1"),
            },
        ),
        (
            "tax_registry_b_1",
            {
                "task06_group": "train_tax_registry_b_1",
                "depth": 1,
                "numeric": ("amount_4917619A",),
                "dates": ("deductiondate_4917603D",),
                "text": (),
                "structural": ("case_id", "num_group1"),
            },
        ),
        (
            "tax_registry_c_1",
            {
                "task06_group": "train_tax_registry_c_1",
                "depth": 1,
                "numeric": ("pmtamount_36A",),
                "dates": ("processingdate_168D",),
                "text": (),
                "structural": ("case_id", "num_group1"),
            },
        ),
    ]
)

COMPONENT_SPECS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "debitcard__debitcard_1",
            {
                "module": "debitcard",
                "source": "debitcard_1",
                "economic": SOURCE_SPECS["debitcard_1"]["numeric"],
                "auxiliary": SOURCE_SPECS["debitcard_1"]["dates"],
            },
        ),
        (
            "debitcard__other_1",
            {
                "module": "debitcard",
                "source": "other_1",
                "economic": (
                    "amtdebitincoming_4809443A",
                    "amtdebitoutgoing_4809440A",
                ),
                "auxiliary": (),
            },
        ),
        (
            "deposit__deposit_1",
            {
                "module": "deposit",
                "source": "deposit_1",
                "economic": ("amount_416A",),
                "auxiliary": SOURCE_SPECS["deposit_1"]["dates"],
            },
        ),
        (
            "deposit__other_1",
            {
                "module": "deposit",
                "source": "other_1",
                "economic": (
                    "amtdepositbalance_4809441A",
                    "amtdepositincoming_4809444A",
                    "amtdepositoutgoing_4809442A",
                ),
                "auxiliary": (),
            },
        ),
        (
            "tax__static_cb_0",
            {
                "module": "tax",
                "source": "static_cb_0",
                "economic": TAX_SUMMARY_FIELDS,
                "auxiliary": SOURCE_SPECS["static_cb_0"]["dates"]
                + SOURCE_SPECS["static_cb_0"]["text"],
            },
        ),
        (
            "tax__tax_registry_a_1",
            {
                "module": "tax",
                "source": "tax_registry_a_1",
                "economic": ("amount_4527230A",),
                "auxiliary": ("recorddate_4527225D",),
            },
        ),
        (
            "tax__tax_registry_b_1",
            {
                "module": "tax",
                "source": "tax_registry_b_1",
                "economic": ("amount_4917619A",),
                "auxiliary": ("deductiondate_4917603D",),
            },
        ),
        (
            "tax__tax_registry_c_1",
            {
                "module": "tax",
                "source": "tax_registry_c_1",
                "economic": ("pmtamount_36A",),
                "auxiliary": ("processingdate_168D",),
            },
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--dictionary", required=True, type=Path)
    parser.add_argument("--task06-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: json_safe(row.get(name)) for name in headers})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_file_state(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths, key=lambda item: str(item.resolve())):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.resolve()),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return rows


def compare_file_state(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> dict[str, Any]:
    left = {row["path"]: row for row in before}
    right = {row["path"]: row for row in after}
    paths = sorted(set(left) | set(right))
    details = [
        {
            "path": path,
            "unchanged": left.get(path) == right.get(path),
            "before": left.get(path),
            "after": right.get(path),
        }
        for path in paths
    ]
    return {
        "method": "file size and mtime_ns; not cryptographic proof of unchanged bytes",
        "all_unchanged": all(row["unchanged"] for row in details),
        "details": details,
    }


def refuse_existing_outputs(output_dir: Path) -> dict[str, Path]:
    targets = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing Batch 1 outputs: {existing}")
    return targets


def classify_numeric(series: pd.Series) -> dict[str, np.ndarray]:
    """Partition values into the six mutually exclusive audit states."""
    native_missing = series.isna().to_numpy(dtype=bool)
    numeric_dtype = pd.api.types.is_numeric_dtype(series.dtype)
    if numeric_dtype:
        parsed = pd.to_numeric(series, errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        missing = native_missing | np.isnan(parsed)
        unparseable = np.zeros(len(series), dtype=bool)
    else:
        parsed_series = pd.to_numeric(series, errors="coerce")
        parsed = parsed_series.to_numpy(dtype=np.float64, na_value=np.nan)
        missing = native_missing
        unparseable = ~native_missing & np.isnan(parsed)
    nonfinite = ~missing & ~unparseable & ~np.isfinite(parsed)
    finite = ~missing & ~unparseable & np.isfinite(parsed)
    zero = finite & (parsed == 0.0)
    positive = finite & (parsed > 0.0)
    negative = finite & (parsed < 0.0)
    assert np.all(
        missing.astype(np.int8)
        + unparseable.astype(np.int8)
        + nonfinite.astype(np.int8)
        + zero.astype(np.int8)
        + positive.astype(np.int8)
        + negative.astype(np.int8)
        == 1
    )
    return {
        "values": parsed,
        "missing": missing,
        "unparseable": unparseable,
        "nonfinite": nonfinite,
        "finite": finite,
        "zero": zero,
        "positive": positive,
        "negative": negative,
        "nonzero": positive | negative,
        "populated": ~missing,
    }


def parse_date_values(series: pd.Series) -> dict[str, Any]:
    native_missing = series.isna().to_numpy(dtype=bool)
    if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
        as_text = series.astype("string")
        empty = (~series.isna() & as_text.str.strip().eq("")).fillna(False).to_numpy(bool)
    else:
        empty = np.zeros(len(series), dtype=bool)
    candidate = ~(native_missing | empty)
    parsed = pd.to_datetime(series.where(candidate), errors="coerce", format="mixed")
    parseable = candidate & parsed.notna().to_numpy(bool)
    failures = candidate & ~parsed.notna().to_numpy(bool)
    return {
        "parsed": parsed,
        "native_missing": native_missing,
        "empty": empty,
        "parseable": parseable,
        "parse_failure": failures,
        "populated": candidate,
    }


def app_counts(mask: np.ndarray, positions: np.ndarray, population: int) -> np.ndarray:
    selected = mask & (positions >= 0)
    if not np.any(selected):
        return np.zeros(population, dtype=np.int32)
    counts = np.bincount(positions[selected], minlength=population)
    if counts.max(initial=0) <= np.iinfo(np.int32).max:
        return counts.astype(np.int32)
    return counts.astype(np.int64)


def numeric_application_partition(
    source_record_counts: np.ndarray,
    finite_counts: np.ndarray,
    nonzero_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    source = source_record_counts > 0
    finite = finite_counts > 0
    nonzero = nonzero_counts > 0
    return {
        "NO_SOURCE_RECORD": ~source,
        "SOURCE_PRESENT_NO_FINITE_OBSERVATION": source & ~finite,
        "FINITE_OBSERVATIONS_ZERO_ONLY": finite & ~nonzero,
        "AT_LEAST_ONE_FINITE_NONZERO_OBSERVATION": finite & nonzero,
    }


def exact_quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            key: None
            for key in ("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max", "mean")
        }
    probs = np.quantile(
        finite,
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
        method=QUANTILE_METHOD,
    )
    return {
        "min": float(np.min(finite)),
        "p01": float(probs[0]),
        "p05": float(probs[1]),
        "p25": float(probs[2]),
        "median": float(probs[3]),
        "p75": float(probs[4]),
        "p95": float(probs[5]),
        "p99": float(probs[6]),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def complete_pattern_rows(
    pattern_type: str,
    bit_order: list[str],
    masks: list[np.ndarray],
) -> list[dict[str, Any]]:
    if not masks:
        raise ValueError("At least one mask is required")
    population = len(masks[0])
    if any(len(mask) != population for mask in masks):
        raise ValueError("Pattern masks must have equal length")
    rows = []
    for bits in itertools.product((0, 1), repeat=len(masks)):
        selected = np.ones(population, dtype=bool)
        for bit, mask in zip(bits, masks):
            selected &= mask if bit else ~mask
        count = int(np.count_nonzero(selected))
        rows.append(
            {
                "pattern_type": pattern_type,
                "bit_order": json.dumps(bit_order, separators=(",", ":")),
                "pattern": "".join(str(bit) for bit in bits),
                "count": count,
                "denominator_name": "verified_base_applications",
                "denominator_value": population,
                "rate": count / population if population else None,
                "definition": "Complete mutually exclusive application pattern.",
            }
        )
    return rows


def module_masks(
    components: list[dict[str, np.ndarray]], population: int
) -> dict[str, np.ndarray]:
    def union(name: str) -> np.ndarray:
        result = np.zeros(population, dtype=bool)
        for component in components:
            result |= component[name]
        return result

    content = union("content")
    nonzero = union("nonzero")
    auxiliary = union("auxiliary_evidence")
    field_evidence = union("field_evidence")
    invalid = union("invalid")
    return {
        "content": content,
        "nonzero": content & nonzero,
        "zero_only": content & ~nonzero,
        "auxiliary_only": auxiliary & ~content,
        "field_evidence": field_evidence,
        "no_content": ~content,
        "invalid": invalid,
    }


def assemble_flags(base_case_ids: pd.Series, columns: dict[str, np.ndarray]) -> pd.DataFrame:
    if base_case_ids.isna().any() or base_case_ids.duplicated().any():
        raise ValueError("Verified base keys must be unique and non-null")
    population = len(base_case_ids)
    bad = [name for name, values in columns.items() if len(values) != population]
    if bad:
        raise ValueError(f"Flag columns have wrong length: {bad}")
    data = {"case_id": base_case_ids.to_numpy(copy=True)}
    data.update({name: columns[name] for name in sorted(columns)})
    frame = pd.DataFrame(data, copy=False)
    if len(frame) != population or frame["case_id"].nunique(dropna=False) != population:
        raise AssertionError("Audit flag construction expanded or lost base rows")
    return frame


def source_positions(case_ids: pd.Series, base_index: pd.Index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonnull = case_ids.notna().to_numpy(bool)
    positions = np.full(len(case_ids), -1, dtype=np.int64)
    if np.any(nonnull):
        positions[nonnull] = base_index.get_indexer(case_ids[nonnull])
    matched = nonnull & (positions >= 0)
    orphan = nonnull & (positions < 0)
    return positions, matched, orphan


def record_count_distribution(counts: np.ndarray) -> dict[str, Any]:
    positive = counts[counts > 0]
    if positive.size == 0:
        return {
            "min": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "applications_1_row": 0,
            "applications_2_rows": 0,
            "applications_3_5_rows": 0,
            "applications_6_10_rows": 0,
            "applications_gt10_rows": 0,
        }
    q = np.quantile(positive, [0.5, 0.9, 0.95, 0.99], method=QUANTILE_METHOD)
    return {
        "min": int(positive.min()),
        "median": float(q[0]),
        "p90": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
        "max": int(positive.max()),
        "applications_1_row": int(np.count_nonzero(positive == 1)),
        "applications_2_rows": int(np.count_nonzero(positive == 2)),
        "applications_3_5_rows": int(np.count_nonzero((positive >= 3) & (positive <= 5))),
        "applications_6_10_rows": int(np.count_nonzero((positive >= 6) & (positive <= 10))),
        "applications_gt10_rows": int(np.count_nonzero(positive > 10)),
    }


def structural_audit(
    source: str,
    frame: pd.DataFrame,
    base_index: pd.Index,
    depth: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    positions, matched, orphan = source_positions(frame["case_id"], base_index)
    nonnull = frame["case_id"].notna().to_numpy(bool)
    source_nonnull = frame.loc[nonnull, "case_id"]
    matched_counts = app_counts(np.ones(len(frame), dtype=bool), positions, len(base_index))
    duplicate_case_id_distinct = int(
        (source_nonnull.value_counts(dropna=False) > 1).sum()
    )
    duplicate_case_id_excess = int(len(source_nonnull) - source_nonnull.nunique())
    result: dict[str, Any] = {
        "source": source,
        "physical_rows": int(len(frame)),
        "null_case_id_rows": int(np.count_nonzero(~nonnull)),
        "distinct_nonnull_case_ids": int(source_nonnull.nunique()),
        "duplicate_case_id_distinct_count": duplicate_case_id_distinct,
        "duplicate_case_id_excess_rows": duplicate_case_id_excess,
        "matched_base_rows": int(np.count_nonzero(matched)),
        "orphan_rows": int(np.count_nonzero(orphan)),
        "matched_distinct_case_ids": int(np.count_nonzero(matched_counts > 0)),
        "orphan_distinct_case_ids": int(frame.loc[orphan, "case_id"].nunique()),
        "missing_base_case_ids": int(len(base_index) - np.count_nonzero(matched_counts > 0)),
        **record_count_distribution(matched_counts),
        "null_num_group1_rows": None,
        "duplicate_case_num_group1_distinct_count": None,
        "duplicate_case_num_group1_excess_rows": None,
        "join_readiness": "",
    }
    if depth == 1:
        group_null = frame["num_group1"].isna().to_numpy(bool)
        valid_pair = nonnull & ~group_null
        pairs = frame.loc[valid_pair, ["case_id", "num_group1"]]
        pair_counts = pairs.value_counts(sort=False)
        result["null_num_group1_rows"] = int(np.count_nonzero(group_null))
        result["duplicate_case_num_group1_distinct_count"] = int(
            np.count_nonzero(pair_counts.to_numpy() > 1)
        )
        result["duplicate_case_num_group1_excess_rows"] = int(
            (pair_counts - 1).clip(lower=0).sum()
        )
        if result["null_case_id_rows"] or result["null_num_group1_rows"]:
            result["join_readiness"] = "KEY_DEFECTS_REQUIRE_REVIEW"
        elif duplicate_case_id_excess:
            result["join_readiness"] = "ONE_TO_MANY_REQUIRES_AGGREGATION"
        else:
            result["join_readiness"] = "OBSERVED_ONE_ROW_PER_MATCHED_APPLICATION"
    elif source == "base":
        result["join_readiness"] = (
            "VERIFIED_BASE_KEY"
            if not result["null_case_id_rows"] and not duplicate_case_id_excess
            else "FAILED_BASE_KEY_PREREQUISITE"
        )
    else:
        result["join_readiness"] = (
            "READY_LEFT_JOIN_ONE_TO_ONE"
            if not result["null_case_id_rows"] and not duplicate_case_id_excess
            else "NOT_READY_ONE_TO_ONE"
        )
    return result, positions, matched, orphan


def shard_overlap_diagnostics(
    frame: pd.DataFrame, selected_fields: list[str]
) -> list[dict[str, Any]]:
    files = sorted(frame["__file"].unique().tolist())
    rows: list[dict[str, Any]] = []
    for left_name, right_name in itertools.combinations(files, 2):
        left = frame.loc[frame["__file"] == left_name]
        right = frame.loc[frame["__file"] == right_name]
        left_keys = pd.Index(left["case_id"].dropna().unique())
        right_keys = pd.Index(right["case_id"].dropna().unique())
        overlap = left_keys.intersection(right_keys, sort=False)
        detail: dict[str, Any] = {
            "left_file": left_name,
            "right_file": right_name,
            "left_rows": int(len(left)),
            "right_rows": int(len(right)),
            "left_distinct_nonnull_case_ids": int(len(left_keys)),
            "right_distinct_nonnull_case_ids": int(len(right_keys)),
            "overlapping_distinct_case_ids": int(len(overlap)),
            "conflicting_selected_value_keys": 0,
            "conflict_check_status": "PASS_NO_OVERLAP" if len(overlap) == 0 else "CHECKED",
        }
        if len(overlap):
            left_overlap = left[left["case_id"].isin(overlap)]
            right_overlap = right[right["case_id"].isin(overlap)]
            if left_overlap["case_id"].duplicated().any() or right_overlap["case_id"].duplicated().any():
                detail["conflicting_selected_value_keys"] = None
                detail["conflict_check_status"] = "NOT_CHECKED_DUPLICATE_KEYS_AVOIDS_CARTESIAN_JOIN"
            else:
                merged = left_overlap[["case_id", *selected_fields]].merge(
                    right_overlap[["case_id", *selected_fields]],
                    on="case_id",
                    how="inner",
                    suffixes=("__left", "__right"),
                    validate="one_to_one",
                )
                conflict = np.zeros(len(merged), dtype=bool)
                for field in selected_fields:
                    a = merged[f"{field}__left"]
                    b = merged[f"{field}__right"]
                    equal = (a.eq(b) | (a.isna() & b.isna())).fillna(False).to_numpy(bool)
                    conflict |= ~equal
                detail["conflicting_selected_value_keys"] = int(np.count_nonzero(conflict))
        rows.append(detail)
    return rows


def read_group(
    train_dir: Path,
    file_names: list[str],
    columns: list[str],
) -> pd.DataFrame:
    frames = []
    for file_name in file_names:
        path = train_dir / file_name
        table = pq.read_table(path, columns=columns, use_threads=True)
        frame = table.to_pandas(split_blocks=True)
        frame["__file"] = file_name
        frames.append(frame)
    if not frames:
        raise ValueError("No physical files supplied")
    return pd.concat(frames, ignore_index=True, copy=False) if len(frames) > 1 else frames[0]


def metric_row(
    source: str,
    field: str,
    description: str,
    role: str,
    metric_name: str,
    value: Any,
    denominator_name: str,
    denominator_value: int | None,
    definition: str,
    status: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "field": field,
        "dictionary_description": description,
        "role": role,
        "metric_name": metric_name,
        "value": value,
        "denominator_name": denominator_name,
        "denominator_value": denominator_value,
        "definition": definition,
        "status": status,
    }


def audit_numeric_field(
    source: str,
    field: str,
    series: pd.Series,
    positions: np.ndarray,
    matched: np.ndarray,
    source_record_counts: np.ndarray,
    description: str,
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    population = len(source_record_counts)
    classified = classify_numeric(series)
    matched_rows = int(np.count_nonzero(matched))
    rows: list[dict[str, Any]] = []
    states = ("missing", "unparseable", "nonfinite", "zero", "positive", "negative")
    for state in states:
        count = int(np.count_nonzero(classified[state] & matched))
        rows.append(
            metric_row(
                source,
                field,
                description,
                role,
                f"row_state_{state}_count",
                count,
                "matched_source_rows",
                matched_rows,
                "Mutually exclusive state on rows matched to a valid base key.",
                "MEASURED",
            )
        )
        rows.append(
            metric_row(
                source,
                field,
                description,
                role,
                f"row_state_{state}_rate",
                count / matched_rows if matched_rows else None,
                "matched_source_rows",
                matched_rows,
                "Rate for the mutually exclusive matched-row numeric state.",
                "MEASURED" if matched_rows else "UNDEFINED_ZERO_DENOMINATOR",
            )
        )

    finite_values = classified["values"][matched & classified["finite"]]
    finite_count = int(finite_values.size)
    rows.append(
        metric_row(source, field, description, role, "finite_observation_count", finite_count,
                   "matched_source_rows", matched_rows,
                   "Finite numeric observations; zeros and negative values are included.", "MEASURED")
    )
    rows.append(
        metric_row(source, field, description, role, "distinct_finite_value_count",
                   int(pd.Series(finite_values).nunique()) if finite_count else 0,
                   "finite_observations", finite_count, "Exact distinct count of finite values.", "MEASURED")
    )
    for name, value in exact_quantiles(finite_values).items():
        rows.append(
            metric_row(source, field, description, role, f"finite_{name}", value,
                       "finite_observations", finite_count,
                       f"Exact finite-value statistic; quantiles use NumPy method={QUANTILE_METHOD}.",
                       "MEASURED" if value is not None else "UNDEFINED_NO_FINITE_VALUES")
        )

    if field in COUNT_FIELDS:
        nonnegative = finite_values >= 0
        integer_like = np.isclose(
            finite_values, np.rint(finite_values), rtol=0.0, atol=FLOAT_EQUAL_ATOL
        )
        for name, mask in (
            ("finite_nonnegative", nonnegative),
            ("finite_integer_like", integer_like),
            ("finite_positive_integer_like", (finite_values > 0) & integer_like),
        ):
            count = int(np.count_nonzero(mask))
            rows.append(
                metric_row(source, field, description, role, f"{name}_count", count,
                           "finite_observations", finite_count,
                           f"Count-field diagnostic; integer tolerance atol={FLOAT_EQUAL_ATOL}, rtol=0.",
                           "MEASURED")
            )
            rows.append(
                metric_row(source, field, description, role, f"{name}_rate",
                           count / finite_count if finite_count else None,
                           "finite_observations", finite_count,
                           "Count-field diagnostic rate; no values were changed.",
                           "MEASURED" if finite_count else "UNDEFINED_ZERO_DENOMINATOR")
            )

    per_app = {
        name: app_counts(classified[name], positions, population)
        for name in ("missing", "unparseable", "nonfinite", "finite", "zero", "nonzero", "populated")
    }
    partition = numeric_application_partition(
        source_record_counts, per_app["finite"], per_app["nonzero"]
    )
    source_applications = int(np.count_nonzero(source_record_counts > 0))
    for state, mask in partition.items():
        count = int(np.count_nonzero(mask))
        rows.append(
            metric_row(source, field, description, role, f"application_state_{state}_count", count,
                       "verified_base_applications", population,
                       "One of four mutually exclusive application states for this field.", "MEASURED")
        )
        rows.append(
            metric_row(source, field, description, role, f"application_state_{state}_population_rate",
                       count / population if population else None,
                       "verified_base_applications", population,
                       "Population rate over all verified base applications.", "MEASURED")
        )
        conditional_value = None if state == "NO_SOURCE_RECORD" or not source_applications else count / source_applications
        rows.append(
            metric_row(source, field, description, role, f"application_state_{state}_conditional_source_rate",
                       conditional_value, "applications_with_source_records", source_applications,
                       "Conditional rate among applications with a source record; not applicable to NO_SOURCE_RECORD.",
                       "MEASURED" if conditional_value is not None else "NOT_APPLICABLE")
        )

    diagnostics = {
        "applications_finite_and_missing": (per_app["finite"] > 0) & (per_app["missing"] > 0),
        "applications_zero_and_nonzero": (per_app["zero"] > 0) & (per_app["nonzero"] > 0),
        "applications_any_invalid": (per_app["unparseable"] > 0) | (per_app["nonfinite"] > 0),
    }
    if source_record_counts.max() <= 1:
        variation_count = 0
    else:
        finite_frame = pd.DataFrame(
            {
                "position": positions[matched & classified["finite"]],
                "value": classified["values"][matched & classified["finite"]],
            }
        )
        if len(finite_frame):
            variation_positions = finite_frame.groupby("position", sort=False)["value"].nunique()
            variation_count = int(np.count_nonzero(variation_positions.to_numpy() > 1))
        else:
            variation_count = 0
    diagnostic_counts = {
        **{name: int(np.count_nonzero(mask)) for name, mask in diagnostics.items()},
        "applications_finite_variation_across_rows": variation_count,
    }
    for name, count in diagnostic_counts.items():
        rows.append(
            metric_row(source, field, description, role, name, count,
                       "verified_base_applications", population,
                       "Overlapping application-level diagnostic; not part of the four-state partition.", "MEASURED")
        )

    return rows, {"classified": classified, "per_app": per_app, "partition": partition}


def audit_date_field(
    source: str,
    field: str,
    series: pd.Series,
    positions: np.ndarray,
    matched: np.ndarray,
    base_dates: pd.Series,
    description: str,
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = parse_date_values(series)
    population = len(base_dates)
    matched_rows = int(np.count_nonzero(matched))
    rows = []
    for state in ("native_missing", "empty", "parseable", "parse_failure"):
        count = int(np.count_nonzero(parsed[state] & matched))
        rows.append(
            metric_row(source, field, description, role, f"row_date_{state}_count", count,
                       "matched_source_rows", matched_rows,
                       "Date audit state; parse failures are separate from native missing values.", "MEASURED")
        )
        rows.append(
            metric_row(source, field, description, role, f"row_date_{state}_rate",
                       count / matched_rows if matched_rows else None,
                       "matched_source_rows", matched_rows,
                       "Date-state rate on matched source rows.",
                       "MEASURED" if matched_rows else "UNDEFINED_ZERO_DENOMINATOR")
        )

    usable = matched & parsed["parseable"]
    usable_indices = np.flatnonzero(usable)
    if usable_indices.size:
        source_dates = parsed["parsed"].iloc[usable_indices].reset_index(drop=True)
        base_for_rows = base_dates.iloc[positions[usable_indices]].reset_index(drop=True)
        pair_valid = base_for_rows.notna().to_numpy(bool)
        offsets = np.full(len(usable_indices), np.nan, dtype=np.float64)
        if np.any(pair_valid):
            delta = source_dates[pair_valid] - base_for_rows[pair_valid]
            offsets[pair_valid] = delta.dt.total_seconds().to_numpy() / 86400.0
    else:
        pair_valid = np.zeros(0, dtype=bool)
        offsets = np.zeros(0, dtype=np.float64)
    valid_offsets = offsets[np.isfinite(offsets)]
    date_pair_count = int(valid_offsets.size)
    rows.append(
        metric_row(source, field, description, role, "date_offset_valid_pair_count", date_pair_count,
                   "parseable_source_dates_on_matched_rows", int(usable_indices.size),
                   "Pairs with parseable source date and usable base date_decision.", "MEASURED")
    )
    rows.append(
        metric_row(source, field, description, role, "date_source_without_usable_base_date_count",
                   int(usable_indices.size - date_pair_count),
                   "parseable_source_dates_on_matched_rows", int(usable_indices.size),
                   "Parseable source dates whose matched base date_decision is unavailable.", "MEASURED")
    )
    for name, mask in (
        ("below_zero", valid_offsets < 0),
        ("equal_zero", valid_offsets == 0),
        ("above_zero", valid_offsets > 0),
    ):
        count = int(np.count_nonzero(mask))
        rows.append(
            metric_row(source, field, description, role, f"date_offset_{name}_count", count,
                       "valid_date_pairs", date_pair_count,
                       "Calendar-day source_date minus base date_decision.", "MEASURED")
        )
        rows.append(
            metric_row(source, field, description, role, f"date_offset_{name}_rate",
                       count / date_pair_count if date_pair_count else None,
                       "valid_date_pairs", date_pair_count,
                       "Rate of calendar-day source-date offset sign.",
                       "MEASURED" if date_pair_count else "UNDEFINED_ZERO_DENOMINATOR")
        )
    if date_pair_count:
        q = np.quantile(valid_offsets, [0.05, 0.5, 0.95], method=QUANTILE_METHOD)
        date_stats = {
            "min": float(np.min(valid_offsets)),
            "p05": float(q[0]),
            "median": float(q[1]),
            "p95": float(q[2]),
            "max": float(np.max(valid_offsets)),
        }
    else:
        date_stats = {name: None for name in ("min", "p05", "median", "p95", "max")}
    for name, value in date_stats.items():
        rows.append(
            metric_row(source, field, description, role, f"date_offset_days_{name}", value,
                       "valid_date_pairs", date_pair_count,
                       f"Calendar-day offset statistic; quantiles use method={QUANTILE_METHOD}.",
                       "MEASURED" if value is not None else "UNDEFINED_NO_VALID_PAIRS")
        )
    return rows, {
        "parsed": parsed,
        "per_app_populated": app_counts(parsed["populated"], positions, population),
        "per_app_parseable": app_counts(parsed["parseable"], positions, population),
    }


def audit_text_field(
    source: str,
    field: str,
    series: pd.Series,
    positions: np.ndarray,
    matched: np.ndarray,
    population: int,
    description: str,
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    native_missing = series.isna().to_numpy(bool)
    text = series.astype("string")
    empty = (~series.isna() & text.str.strip().eq("")).fillna(False).to_numpy(bool)
    populated = ~(native_missing | empty)
    matched_rows = int(np.count_nonzero(matched))
    rows = []
    for name, mask in (
        ("native_missing", native_missing),
        ("empty_or_whitespace", empty),
        ("populated", populated),
    ):
        count = int(np.count_nonzero(mask & matched))
        rows.append(
            metric_row(source, field, description, role, f"row_text_{name}_count", count,
                       "matched_source_rows", matched_rows,
                       "Text auxiliary-field state on matched source rows.", "MEASURED")
        )
        rows.append(
            metric_row(source, field, description, role, f"row_text_{name}_rate",
                       count / matched_rows if matched_rows else None,
                       "matched_source_rows", matched_rows,
                       "Text auxiliary-field state rate.",
                       "MEASURED" if matched_rows else "UNDEFINED_ZERO_DENOMINATOR")
        )
    distinct = int(text[matched & populated].nunique())
    rows.append(
        metric_row(source, field, description, role, "distinct_populated_token_count", distinct,
                   "populated_matched_rows", int(np.count_nonzero(matched & populated)),
                   "Exact distinct populated-token count; tokens are not printed.", "MEASURED")
    )
    return rows, {"per_app_populated": app_counts(populated, positions, population)}


def latest_date_tie_diagnostics(
    frame: pd.DataFrame,
    matched: np.ndarray,
    date_field: str,
    economic_fields: list[str],
) -> dict[str, Any]:
    parsed = parse_date_values(frame[date_field])
    work = frame.loc[matched, ["case_id", *economic_fields]].copy()
    work["__date"] = parsed["parsed"].loc[matched].to_numpy()
    source_apps = int(work["case_id"].nunique())
    parseable = work["__date"].notna()
    apps_with_date = int(work.loc[parseable, "case_id"].nunique())
    if not parseable.any():
        return {
            "source_applications": source_apps,
            "applications_all_dates_missing_or_invalid": source_apps,
            "applications_multiple_rows_at_latest_date": 0,
            "applications_latest_tie_conflicting_selected_economic_values": 0,
        }
    valid = work.loc[parseable].copy()
    valid["__latest"] = valid.groupby("case_id", sort=False)["__date"].transform("max")
    latest = valid.loc[valid["__date"] == valid["__latest"]]
    latest_counts = latest.groupby("case_id", sort=False).size()
    tied_ids = latest_counts[latest_counts > 1].index
    tied = latest[latest["case_id"].isin(tied_ids)]
    conflict_count = 0
    if len(tied):
        for _, group in tied.groupby("case_id", sort=False):
            canonical = group[economic_fields].astype(object).where(group[economic_fields].notna(), "__MISSING__")
            if len(canonical.drop_duplicates()) > 1:
                conflict_count += 1
    return {
        "source_applications": source_apps,
        "applications_all_dates_missing_or_invalid": source_apps - apps_with_date,
        "applications_multiple_rows_at_latest_date": int(len(tied_ids)),
        "applications_latest_tie_conflicting_selected_economic_values": int(conflict_count),
    }


def pair_diagnostic_rows(
    source: str,
    fields: tuple[str, ...],
    cache: dict[str, dict[str, Any]],
    descriptions: dict[str, str],
    role: str,
) -> list[dict[str, Any]]:
    rows = []
    for left, right in itertools.combinations(fields, 2):
        left_data = cache[left]["classified"]
        right_data = cache[right]["classified"]
        matched = cache[left]["matched"] & cache[right]["matched"]
        finite_pair = matched & left_data["finite"] & right_data["finite"]
        both_missing = matched & left_data["missing"] & right_data["missing"]
        pair_count = int(np.count_nonzero(finite_pair))
        left_values = left_data["values"][finite_pair]
        right_values = right_data["values"][finite_pair]
        equal = np.isclose(
            left_values,
            right_values,
            rtol=FLOAT_EQUAL_RTOL,
            atol=FLOAT_EQUAL_ATOL,
            equal_nan=False,
        )
        correlation = None
        if pair_count >= 2 and np.std(left_values) > 0 and np.std(right_values) > 0:
            correlation = float(np.corrcoef(left_values, right_values)[0, 1])
        pair_name = f"{left}|{right}"
        pair_description = f"{descriptions.get(left, '')} | {descriptions.get(right, '')}"
        metrics = (
            ("joint_finite_row_count", pair_count, int(np.count_nonzero(matched))),
            ("joint_missing_row_count", int(np.count_nonzero(both_missing)), int(np.count_nonzero(matched))),
            ("finite_pair_equal_count", int(np.count_nonzero(equal)), pair_count),
            ("finite_pair_disagreement_count", int(pair_count - np.count_nonzero(equal)), pair_count),
            ("finite_pair_equality_rate", int(np.count_nonzero(equal)) / pair_count if pair_count else None, pair_count),
            ("finite_pair_pearson_correlation", correlation, pair_count),
        )
        for metric_name, value, denominator in metrics:
            rows.append(
                metric_row(source, pair_name, pair_description, role, metric_name, value,
                           "joint_matched_rows" if "joint_" in metric_name else "finite_value_pairs",
                           denominator,
                           f"Pair diagnostic; equality uses rtol={FLOAT_EQUAL_RTOL}, atol={FLOAT_EQUAL_ATOL}.",
                           "MEASURED" if value is not None else "UNDEFINED")
            )
    return rows


def coverage_row(
    scope_type: str,
    scope_name: str,
    measure: str,
    definition: str,
    mask: np.ndarray,
    population: int,
    status: str,
) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    return {
        "scope_type": scope_type,
        "scope_name": scope_name,
        "measure": measure,
        "definition": definition,
        "population_count": count,
        "denominator_name": "verified_base_applications",
        "denominator_value": population,
        "rate": count / population if population else None,
        "scope_status": status,
    }


def coverage_measure_mask(data: dict[str, np.ndarray], measure: str) -> np.ndarray:
    if measure == "auxiliary_only":
        if "auxiliary_only" in data:
            return data["auxiliary_only"]
        return data["auxiliary_evidence"] & ~data["content"]
    key = {
        "numeric_content": "content",
        "no_numeric_content": "no_content",
        "invalid_numeric": "invalid",
    }.get(measure, measure)
    return data[key]


def feature_review_rows(
    descriptions: dict[str, str],
    structures: dict[str, dict[str, Any]],
    date_ties: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    economic_fields = {
        field
        for component in COMPONENT_SPECS.values()
        for field in component["economic"]
    }
    auxiliary_fields = {
        field
        for component in COMPONENT_SPECS.values()
        for field in component["auxiliary"]
    }
    for source, spec in SOURCE_SPECS.items():
        if source == "base":
            continue
        print(f"Auditing source {source}...", flush=True)
        structure = structures[source]
        structure_text = (
            f"{structure['physical_rows']} physical rows; "
            f"{structure['matched_distinct_case_ids']} matched applications; "
            f"multiplicity median={structure['median']}, p99={structure['p99']}, max={structure['max']}; "
            f"join status={structure['join_readiness']}."
        )
        for field in (*spec["numeric"], *spec["dates"], *spec["text"]):
            is_contact = field in CONTACT_FIELDS
            if source == "static_cb_0" and field in TAX_SUMMARY_FIELDS:
                row_unit = "Dictionary supports a precomputed tax summary; underlying window/source relationship remains undocumented."
                possible = "Retain as a direct summary candidate only after timing/source validation; do not sum summary fields together."
            elif source in {"tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1"}:
                row_unit = "Dictionary supports a tax deduction amount/date record, but no cross-source matching identity is established."
                possible = "Possible per-application count/sum/date-window aggregates; window and cross-source overlap remain unresolved."
            elif source == "debitcard_1":
                row_unit = "Dictionary supports debit-card turnover/balance/opening-date concepts; exact row unit is UNKNOWN_FROM_AVAILABLE_DOCUMENTATION."
                possible = "Possible per-application summaries or latest-date use, subject to card multiplicity and tied-date review."
            elif source == "deposit_1":
                row_unit = "'Deposit amount' and contract dates do not establish whether rows are accounts, transactions, or snapshots."
                possible = "Possible per-application or dated summaries; summing balances/snapshots and latest-record selection remain unresolved."
            elif source == "other_1":
                row_unit = "Dictionary describes debit/deposit amounts, but exact record unit and window are UNKNOWN_FROM_AVAILABLE_DOCUMENTATION."
                possible = "Possible independent debit/deposit per-application summaries; do not merge module meanings or assume additivity."
            elif is_contact:
                row_unit = "Static application-level count description; shared contact does not establish person identity, ownership, fraud, or borrowing."
                possible = "Audit only; no threshold or predictive use is approved."
            else:
                row_unit = "UNKNOWN_FROM_AVAILABLE_DOCUMENTATION."
                possible = "Pending researcher review."

            if field.startswith("openingdate"):
                timing = "Opening date describes relationship timing, not necessarily transaction timing or availability proof."
            elif field == "contractenddate_991D":
                timing = "Potentially a known future contractual date; a positive decision-date offset is not automatically leakage."
            elif field.startswith(("recorddate", "deductiondate")):
                timing = "Potential economic-event date, subject to source documentation and availability verification."
            elif field.startswith(("processingdate", "assignmentdate", "responsedate")):
                timing = "Operational date; it does not automatically establish economic-event or data-availability timing."
            elif field == "requesttype_4525192L":
                timing = "A populated/default request code does not prove that tax records were returned or available before decision."
            else:
                timing = "Point-in-time availability is not established by this distribution audit."
            tie_key = f"{source}.{field}"
            if tie_key in date_ties:
                timing += " Latest-date tie diagnostics are recorded in audit_summary.json."
            if field in economic_fields:
                missing_zero = (
                    "Retain documented finite zeros. An all-missing group must remain missing; "
                    "a partial observed sum is not automatically complete."
                )
            elif field in auxiliary_fields:
                missing_zero = "Keep absent, empty, parse-failed, and populated states distinct; auxiliary evidence is not economic content."
            else:
                missing_zero = "Do not infer absence of a person/account/activity from a missing value."
            rows.append(
                {
                    "source": source,
                    "field_or_module": field,
                    "dictionary_meaning": descriptions.get(field, ""),
                    "observed_structure": structure_text,
                    "row_unit_evidence": row_unit,
                    "timing_concerns": timing,
                    "possible_future_use_or_aggregation": possible,
                    "missing_zero_considerations": missing_zero,
                    "unresolved_questions": (
                        "Confirm row unit, observation window, pre-decision availability, and whether repeated records are additive or snapshots."
                    ),
                    "researcher_review_status": (
                        "AUDIT_ONLY_NOT_APPROVED_FOR_MODEL"
                        if is_contact
                        else "PENDING_FINAL_FEATURE_APPROVAL"
                    ),
                }
            )
    for module in ("debitcard", "deposit", "tax"):
        rows.append(
            {
                "source": "MULTIPLE_SOURCES",
                "field_or_module": module,
                "dictionary_meaning": "Module assembled only for audit coverage from explicitly assigned fields.",
                "observed_structure": "Independent per-application component flags were unioned; raw detail tables were not cross-joined.",
                "row_unit_evidence": "Component row units differ or remain unresolved; module union does not establish additivity.",
                "timing_concerns": "Each component requires separate point-in-time verification.",
                "possible_future_use_or_aggregation": "Discuss a small, nonredundant set of source/content indicators and defensible summaries after review.",
                "missing_zero_considerations": "Absent source, present-but-unobserved, finite zero, finite nonzero, and invalid values remain distinct.",
                "unresolved_questions": "Resolve source overlap, windows, aggregation, timing, and imputation using training data only in a later task.",
                "researcher_review_status": "PENDING_FINAL_FEATURE_APPROVAL",
            }
        )
    return rows


def markdown_report(
    population: int,
    module_data: dict[str, dict[str, np.ndarray]],
    components: dict[str, dict[str, np.ndarray]],
    structures: dict[str, dict[str, Any]],
    field_rows: list[dict[str, Any]],
    input_unchanged: bool,
    validation_checks: list[dict[str, str]],
) -> str:
    def count(module: str, key: str) -> int:
        return int(np.count_nonzero(module_data[module][key]))

    any_content = module_data["debitcard"]["content"] | module_data["deposit"]["content"] | module_data["tax"]["content"]
    all_content = module_data["debitcard"]["content"] & module_data["deposit"]["content"] & module_data["tax"]["content"]

    negative = []
    nonfinite = []
    zero_heavy = []
    for row in field_rows:
        metric = row["metric_name"]
        value = row["value"]
        if metric == "row_state_negative_count" and value:
            negative.append((int(value), row["source"], row["field"]))
        elif metric in {"row_state_nonfinite_count", "row_state_unparseable_count"} and value:
            nonfinite.append((int(value), row["source"], row["field"], metric))
        elif metric == "row_state_zero_rate" and value is not None:
            zero_heavy.append((float(value), row["source"], row["field"]))
    negative.sort(reverse=True)
    nonfinite.sort(reverse=True)
    zero_heavy.sort(reverse=True)

    lines = [
        "# Task 07 Batch 1 — Alternative-data coverage and record-structure audit",
        "",
        "## Measured findings",
        "",
        f"The verified denominator is **{population:,} unique, non-null base application keys**. Counts refer to applications, not natural persons.",
        "",
        "Observed numeric content means at least one finite value in the explicitly listed economic fields; finite zero and negative values count as observed content.",
        "",
        "| Module | Numeric content | Rate | Field evidence | Auxiliary only | Zero only | Nonzero |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for module in ("debitcard", "deposit", "tax"):
        numeric = count(module, "content")
        lines.append(
            f"| {module} | {numeric:,} | {numeric / population:.4%} | "
            f"{count(module, 'field_evidence'):,} | {count(module, 'auxiliary_only'):,} | "
            f"{count(module, 'zero_only'):,} | {count(module, 'nonzero'):,} |"
        )
    lines.extend(
        [
            "",
            f"At least one AD module has numeric content for **{np.count_nonzero(any_content):,}** applications ({np.count_nonzero(any_content)/population:.4%}); all three do so for **{np.count_nonzero(all_content):,}** ({np.count_nonzero(all_content)/population:.4%}).",
            "",
            "Container presence is not content evidence. In particular, static_cb_0 presence is not tax-content coverage and other_1 presence is not automatically debit-card or deposit coverage.",
            "",
            "## Source and join structure",
            "",
            "| Source | Physical rows | Matched applications | Orphan rows | Missing base keys | Multiplicity max | Join assessment |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for source, row in structures.items():
        lines.append(
            f"| {source} | {row['physical_rows']:,} | {row['matched_distinct_case_ids']:,} | "
            f"{row['orphan_rows']:,} | {row['missing_base_case_ids']:,} | {row['max']} | {row['join_readiness']} |"
        )
    lines.extend(
        [
            "",
            "Detail sources were summarized independently by base case_id; no depth-1 tables were joined to one another and num_group1 was not treated as a chronological or cross-table identifier.",
            "",
            "## Missing, zero, and invalid values",
            "",
        ]
    )
    if nonfinite:
        lines.append("Fields with non-finite or unparseable matched-row values: " + "; ".join(f"{source}.{field} ({metric}={value:,})" for value, source, field, metric in nonfinite[:10]) + ".")
    else:
        lines.append("No non-finite or unparseable matched-row numeric values were observed in the selected fields.")
    if negative:
        lines.append("Largest negative-value counts: " + "; ".join(f"{source}.{field}={value:,}" for value, source, field in negative[:10]) + ". Negative values were retained as observed content, not corrected.")
    else:
        lines.append("No finite negative values were observed in the selected fields.")
    if zero_heavy:
        lines.append("Highest matched-row zero rates: " + "; ".join(f"{source}.{field}={rate:.2%}" for rate, source, field in zero_heavy[:10]) + ".")
    lines.extend(
        [
            "",
            "Absent source records, present records with no finite field observation, zero-only observations, and finite nonzero observations are reported separately. An all-missing future aggregation must remain missing; a default sum of zero is not acceptable.",
            "",
            "## Aggregation and timing limits",
            "",
            "- `num_group1` is only an observed local record index here. It is not established as chronological, globally unique, or a cross-table join key.",
            "- Repeated amount/date combinations were diagnosed but not removed. A source-record count is not an account or contract count without documentation.",
            "- Summing balances, snapshots, deduction records with unknown windows, or tax sources that may overlap remains unresolved.",
            "- Opening, contract-end, record, processing, assignment, and response dates have different possible meanings. Positive offsets are reported but are not automatically labeled leakage.",
            "- Contact-sharing fields remain AUDIT_ONLY_NOT_APPROVED_FOR_MODEL and do not contribute to any AD flag or overlap pattern.",
            "- Future imputation or model preprocessing was not performed. Missing/source indicators and training-only imputation remain researcher-review topics.",
            "",
            "## Verification",
            "",
            f"Lightweight input size/mtime comparison: **{'PASS' if input_unchanged else 'FAIL'}**. This is not cryptographic proof of unchanged bytes.",
            f"Validation checks: **{sum(row['status'] == 'PASS' for row in validation_checks)} PASS**, **{sum(row['status'] == 'FAIL' for row in validation_checks)} FAIL**, **{sum(row['status'] == 'NOT_CHECKED' for row in validation_checks)} NOT_CHECKED**.",
            "",
            "This audit does not approve fields for modeling and does not establish point-in-time eligibility, row semantics, source completeness, or missingness mechanisms.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    for label, path in (
        ("TRAIN directory", args.train_dir),
        ("dictionary", args.dictionary),
        ("Task 06 directory", args.task06_dir),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required {label} is unavailable: {path}")
    targets = refuse_existing_outputs(args.output_dir)
    task06_paths = {
        name: args.task06_dir / name
        for name in (
            "table_inventory.csv",
            "field_inventory.csv",
            "proposal_candidates.csv",
            "inventory_summary.json",
        )
    }
    missing_task06 = [str(path) for path in task06_paths.values() if not path.is_file()]
    if missing_task06:
        raise FileNotFoundError(f"Required Task 06 reports are unavailable: {missing_task06}")

    table_inventory = read_csv(task06_paths["table_inventory.csv"])
    field_inventory = read_csv(task06_paths["field_inventory.csv"])
    task06_summary = json.loads(task06_paths["inventory_summary.json"].read_text(encoding="utf-8"))
    dictionary_rows = read_csv(args.dictionary)
    if not dictionary_rows or not {"Variable", "Description"}.issubset(dictionary_rows[0]):
        raise ValueError("Dictionary must contain Variable and Description columns")
    dictionary_names = [row["Variable"] for row in dictionary_rows]
    if any(not name.strip() for name in dictionary_names) or len(dictionary_names) != len(set(dictionary_names)):
        raise ValueError("Dictionary has blank or duplicate variable names")
    descriptions = {row["Variable"]: row["Description"] for row in dictionary_rows}

    group_to_files: dict[str, list[str]] = {}
    for source, spec in SOURCE_SPECS.items():
        records = sorted(
            (row for row in table_inventory if row["provisional_table_group"] == spec["task06_group"]),
            key=lambda row: row["file_name"],
        )
        if not records:
            raise FileNotFoundError(
                f"Task 06 inventory has no physical files for required source {source} ({spec['task06_group']})"
            )
        group_to_files[source] = [row["file_name"] for row in records]
        for file_name in group_to_files[source]:
            if not (args.train_dir / file_name).is_file():
                raise FileNotFoundError(f"Inventoried TRAIN file is unavailable: {args.train_dir / file_name}")

    schema_by_group_file: dict[tuple[str, str], dict[str, str]] = {}
    for row in field_inventory:
        schema_by_group_file.setdefault(
            (row["provisional_table_group"], row["file_name"]), {}
        )[row["field_name"]] = row["arrow_type"]

    schema_validation: list[dict[str, Any]] = []
    unavailable_fields: set[tuple[str, str]] = set()
    for source, spec in SOURCE_SPECS.items():
        selected = list(dict.fromkeys((*spec["structural"], *spec["numeric"], *spec["dates"], *spec["text"])))
        for file_name in group_to_files[source]:
            schema = schema_by_group_file.get((spec["task06_group"], file_name), {})
            for field in selected:
                arrow_type = schema.get(field)
                if arrow_type is None:
                    unavailable_fields.add((source, field))
                    status = "MISSING"
                elif field in spec["numeric"] and not any(
                    arrow_type.startswith(prefix)
                    for prefix in ("int", "uint", "float", "double", "decimal")
                ):
                    status = "UNEXPECTED_NUMERIC_TYPE_AUDIT_PARSE_REQUIRED"
                elif field in spec["dates"] and not (
                    arrow_type.startswith("string")
                    or arrow_type.startswith("date")
                    or arrow_type.startswith("timestamp")
                ):
                    status = "UNEXPECTED_DATE_TYPE_AUDIT_PARSE_REQUIRED"
                else:
                    status = "VERIFIED_PRESENT"
                schema_validation.append(
                    {
                        "source": source,
                        "file_name": file_name,
                        "field": field,
                        "arrow_type": arrow_type,
                        "status": status,
                    }
                )

    input_paths = [args.dictionary, *task06_paths.values()]
    input_paths.extend(
        args.train_dir / file_name
        for files in group_to_files.values()
        for file_name in files
    )
    before_state = collect_file_state(input_paths)

    validation_checks: list[dict[str, str]] = []

    def add_check(name: str, passed: bool | None, detail: str) -> None:
        validation_checks.append(
            {
                "check": name,
                "status": "NOT_CHECKED" if passed is None else ("PASS" if passed else "FAIL"),
                "detail": detail,
            }
        )

    base_columns = list(dict.fromkeys((*SOURCE_SPECS["base"]["structural"], *SOURCE_SPECS["base"]["dates"])))
    if any(("base", field) in unavailable_fields for field in base_columns):
        raise RuntimeError("Base prerequisite fields are missing; dependent outputs cannot be produced")
    base = read_group(args.train_dir, group_to_files["base"], base_columns)
    base_null = int(base["case_id"].isna().sum())
    base_distinct = int(base["case_id"].nunique())
    base_duplicate_excess = int(base["case_id"].notna().sum() - base_distinct)
    base_valid = base_null == 0 and base_duplicate_excess == 0
    if not base_valid:
        raise RuntimeError(
            "Base key prerequisite failed: "
            f"null case_id rows={base_null}, duplicate excess rows={base_duplicate_excess}. "
            "Application-level outputs were not produced."
        )
    base_index = pd.Index(base["case_id"])
    population = len(base_index)
    base_structure, base_positions, base_matched, _ = structural_audit(
        "base", base, base_index, 0
    )
    base_date_info = parse_date_values(base["date_decision"])
    base_dates = base_date_info["parsed"].reset_index(drop=True)
    add_check(
        "base_key_prerequisite",
        base_valid,
        f"rows={len(base)}, distinct_nonnull={base_distinct}, null={base_null}, duplicate_excess={base_duplicate_excess}",
    )
    add_check(
        "base_date_parseability",
        int(np.count_nonzero(base_date_info["parse_failure"])) == 0,
        (
            f"native_missing={np.count_nonzero(base_date_info['native_missing'])}, "
            f"empty={np.count_nonzero(base_date_info['empty'])}, "
            f"parse_failures={np.count_nonzero(base_date_info['parse_failure'])}"
        ),
    )

    field_audit_rows: list[dict[str, Any]] = []
    base_matched_count = population
    for state in ("native_missing", "empty", "parseable", "parse_failure"):
        count_value = int(np.count_nonzero(base_date_info[state]))
        field_audit_rows.append(
            metric_row(
                "base", "date_decision", descriptions.get("date_decision", ""), "BASE_TIME_METADATA",
                f"row_date_{state}_count", count_value, "base_rows", base_matched_count,
                "Base decision-date parse audit; failures are not merged into missing.", "MEASURED"
            )
        )

    structures: dict[str, dict[str, Any]] = {"base": base_structure}
    source_record_counts: dict[str, np.ndarray] = {
        "base": np.ones(population, dtype=np.int32)
    }
    flag_columns: dict[str, np.ndarray] = {
        "source__base__present": np.ones(population, dtype=bool),
        "source__base__record_count": np.ones(population, dtype=np.int32),
    }
    field_app: dict[str, dict[str, Any]] = {}
    shard_diagnostics: dict[str, list[dict[str, Any]]] = {}
    projection_duplicates: dict[str, int] = {}
    date_tie_diagnostics: dict[str, dict[str, Any]] = {}

    for source, spec in SOURCE_SPECS.items():
        if source == "base":
            continue
        selected = list(dict.fromkeys((*spec["structural"], *spec["numeric"], *spec["dates"], *spec["text"])))
        available = [field for field in selected if (source, field) not in unavailable_fields]
        if "case_id" not in available or (spec["depth"] == 1 and "num_group1" not in available):
            raise RuntimeError(f"Required structural fields missing for {source}")
        frame = read_group(args.train_dir, group_to_files[source], available)
        structure, positions, matched, orphan = structural_audit(
            source, frame, base_index, spec["depth"]
        )
        structures[source] = structure
        counts = app_counts(np.ones(len(frame), dtype=bool), positions, population)
        source_record_counts[source] = counts
        flag_columns[f"source__{source}__present"] = counts > 0
        flag_columns[f"source__{source}__record_count"] = counts

        add_check(
            f"{source}_row_key_partition",
            structure["matched_base_rows"] + structure["orphan_rows"] + structure["null_case_id_rows"] == structure["physical_rows"],
            "matched-key rows + orphan-key rows + null-key rows must equal physical rows",
        )
        add_check(
            f"{source}_distinct_key_partition",
            structure["matched_distinct_case_ids"] + structure["orphan_distinct_case_ids"] == structure["distinct_nonnull_case_ids"],
            "matched and orphan distinct non-null keys must equal all distinct non-null keys",
        )
        shard_diagnostics[source] = shard_overlap_diagnostics(
            frame, [field for field in available if field not in {"case_id", "num_group1"}]
        )
        projected = [field for field in available if field != "__file"]
        projection_duplicates[source] = int(frame[projected].duplicated().sum())

        per_source_numeric_cache: dict[str, dict[str, Any]] = {}
        for field in spec["numeric"]:
            if field not in frame:
                field_audit_rows.append(
                    metric_row(source, field, descriptions.get(field, ""), "UNAVAILABLE",
                               "field_availability", None, "not_applicable", None,
                               "Exact field absent from at least one required source schema.", "NOT_AVAILABLE")
                )
                continue
            role = (
                "CONTACT_AUDIT_ONLY_NOT_APPROVED_FOR_MODEL"
                if field in CONTACT_FIELDS
                else "AD_ECONOMIC_CONTENT_PENDING_APPROVAL"
            )
            rows, result = audit_numeric_field(
                source, field, frame[field], positions, matched, counts,
                descriptions.get(field, ""), role
            )
            field_audit_rows.extend(rows)
            result["matched"] = matched
            field_app[field] = result["per_app"]
            per_source_numeric_cache[field] = result
            state_sum = sum(
                np.count_nonzero(result["classified"][name] & matched)
                for name in ("missing", "unparseable", "nonfinite", "zero", "positive", "negative")
            )
            add_check(
                f"numeric_row_states::{source}.{field}",
                state_sum == int(np.count_nonzero(matched)),
                f"state_sum={state_sum}, matched_rows={np.count_nonzero(matched)}",
            )
            partition_sum = sum(np.count_nonzero(mask) for mask in result["partition"].values())
            add_check(
                f"numeric_application_states::{source}.{field}",
                partition_sum == population,
                f"partition_sum={partition_sum}, N={population}",
            )
            if field not in CONTACT_FIELDS:
                flag_columns[f"economic_field__{field}__finite_row_count"] = result["per_app"]["finite"]
                flag_columns[f"economic_field__{field}__missing_row_count"] = result["per_app"]["missing"]
                flag_columns[f"economic_field__{field}__zero_row_count"] = result["per_app"]["zero"]

        for field in spec["dates"]:
            if field not in frame:
                continue
            role = "AD_AUXILIARY_DATE_PENDING_APPROVAL"
            rows, result = audit_date_field(
                source, field, frame[field], positions, matched, base_dates,
                descriptions.get(field, ""), role
            )
            field_audit_rows.extend(rows)
            field_app[field] = {"populated": result["per_app_populated"], "parseable": result["per_app_parseable"]}
            if spec["depth"] == 1:
                key = f"{source}.{field}"
                date_tie_diagnostics[key] = latest_date_tie_diagnostics(
                    frame, matched, field, [name for name in spec["numeric"] if name in frame]
                )

        for field in spec["text"]:
            if field not in frame:
                continue
            rows, result = audit_text_field(
                source, field, frame[field], positions, matched, population,
                descriptions.get(field, ""), "AD_AUXILIARY_TEXT_PENDING_APPROVAL"
            )
            field_audit_rows.extend(rows)
            field_app[field] = {"populated": result["per_app_populated"]}

        if source == "static_0":
            field_audit_rows.extend(
                pair_diagnostic_rows(source, CONTACT_FIELDS, per_source_numeric_cache,
                                     descriptions, "CONTACT_REDUNDANCY_DIAGNOSTIC_AUDIT_ONLY")
            )
        if source == "static_cb_0":
            field_audit_rows.extend(
                pair_diagnostic_rows(source, TAX_SUMMARY_FIELDS, per_source_numeric_cache,
                                     descriptions, "TAX_SUMMARY_REDUNDANCY_DIAGNOSTIC")
            )
        print(f"Completed source {source}.", flush=True)
        del frame, positions, matched, orphan, per_source_numeric_cache

    source_structure_rows: list[dict[str, Any]] = []
    for source, structure in structures.items():
        overlaps = shard_diagnostics.get(source, [])
        source_structure_rows.append(
            {
                **structure,
                "physical_file_count": len(group_to_files[source]),
                "physical_files_json": json.dumps(group_to_files[source], separators=(",", ":")),
                "shard_pair_count": len(overlaps),
                "shard_overlap_distinct_case_ids_sum": sum(
                    row["overlapping_distinct_case_ids"] for row in overlaps
                ),
                "shard_conflicting_selected_value_keys_sum": (
                    sum(row["conflicting_selected_value_keys"] or 0 for row in overlaps)
                    if all(row["conflicting_selected_value_keys"] is not None for row in overlaps)
                    else None
                ),
                "identical_selected_projection_excess_rows": projection_duplicates.get(source, 0),
                "record_unit_interpretation": (
                    "APPLICATION_RECORD"
                    if source == "base"
                    else (
                        "OBSERVED_ONE_ROW_PER_CASE_ID; SUBSTANTIVE UNIT REQUIRES DOCUMENTATION"
                        if SOURCE_SPECS[source]["depth"] == 0
                        else "LOCAL_INDEXED_RECORD; NOT ESTABLISHED AS ACCOUNT/CONTRACT/CHRONOLOGY"
                    )
                ),
            }
        )

    components: dict[str, dict[str, np.ndarray]] = {}
    for component_name, spec in COMPONENT_SPECS.items():
        content = np.zeros(population, dtype=bool)
        nonzero = np.zeros(population, dtype=bool)
        invalid = np.zeros(population, dtype=bool)
        economic_evidence = np.zeros(population, dtype=bool)
        auxiliary = np.zeros(population, dtype=bool)
        for field in spec["economic"]:
            if field not in field_app:
                continue
            data = field_app[field]
            content |= data["finite"] > 0
            nonzero |= data["nonzero"] > 0
            invalid |= (data["unparseable"] > 0) | (data["nonfinite"] > 0)
            economic_evidence |= data["populated"] > 0
        for field in spec["auxiliary"]:
            if field in field_app:
                auxiliary |= field_app[field]["populated"] > 0
        components[component_name] = {
            "content": content,
            "nonzero": content & nonzero,
            "zero_only": content & ~nonzero,
            "invalid": invalid,
            "auxiliary_evidence": auxiliary,
            "field_evidence": economic_evidence | auxiliary,
            "source_present": source_record_counts[spec["source"]] > 0,
        }
        for key in ("content", "nonzero", "zero_only", "invalid", "auxiliary_evidence", "field_evidence"):
            flag_columns[f"component__{component_name}__{key}"] = components[component_name][key]

    module_data: dict[str, dict[str, np.ndarray]] = {}
    for module in ("debitcard", "deposit", "tax"):
        module_components = [
            data for name, data in components.items() if COMPONENT_SPECS[name]["module"] == module
        ]
        module_data[module] = module_masks(module_components, population)
        for key, values in module_data[module].items():
            flag_columns[f"module__{module}__{key}"] = values
        expected_content = np.zeros(population, dtype=bool)
        for data in module_components:
            expected_content |= data["content"]
        add_check(
            f"module_content_or::{module}",
            np.array_equal(expected_content, module_data[module]["content"]),
            "Module numeric-content flag equals OR across assigned source/field content flags; finite zero contributes.",
        )
        add_check(
            f"module_zero_nonzero_partition::{module}",
            np.array_equal(
                module_data[module]["content"],
                module_data[module]["zero_only"] | module_data[module]["nonzero"],
            ) and not np.any(module_data[module]["zero_only"] & module_data[module]["nonzero"]),
            "Zero-only and nonzero states partition applications with module numeric content.",
        )
        add_check(
            f"module_content_within_field_evidence::{module}",
            not np.any(module_data[module]["content"] & ~module_data[module]["field_evidence"]),
            "Numeric content cannot exceed module field evidence.",
        )

    coverage_rows: list[dict[str, Any]] = []
    for source in SOURCE_SPECS:
        coverage_rows.append(
            coverage_row(
                "SOURCE_CONTAINER", source, "matched_source_footprint",
                "Distinct base applications with at least one matched source row; container presence is not content evidence.",
                source_record_counts[source] > 0, population,
                "CONTAINER_ONLY" if source in {"static_cb_0", "other_1"} else "MEASURED",
            )
        )
    for name, data in components.items():
        for measure, definition in (
            ("field_evidence", "Any populated assigned economic or auxiliary field."),
            ("numeric_content", "At least one finite assigned economic value, including zero or negative."),
            ("auxiliary_only", "Auxiliary field evidence without finite assigned economic content."),
            ("zero_only", "Finite assigned economic content exists and all finite values equal zero."),
            ("nonzero", "At least one finite nonzero assigned economic value."),
            ("invalid_numeric", "At least one non-finite or unparseable assigned economic value."),
        ):
            mask = coverage_measure_mask(data, measure)
            coverage_rows.append(
                coverage_row("MODULE_COMPONENT", name, measure, definition, mask, population, "MEASURED")
            )
        other_content = np.zeros(population, dtype=bool)
        for other_name, other_data in components.items():
            if other_name != name and COMPONENT_SPECS[other_name]["module"] == COMPONENT_SPECS[name]["module"]:
                other_content |= other_data["content"]
        coverage_rows.append(
            coverage_row(
                "MODULE_COMPONENT", name, "unique_numeric_content_vs_other_module_components",
                "Numeric-content applications contributed by this component and by no other component in the same module.",
                data["content"] & ~other_content, population, "MEASURED",
            )
        )
    for module, data in module_data.items():
        for measure, definition in (
            ("field_evidence", "Any populated listed economic or auxiliary field across module sources."),
            ("numeric_content", "At least one finite listed economic value across module sources."),
            ("auxiliary_only", "Auxiliary evidence but no observed numeric economic content."),
            ("zero_only", "Observed module numeric content with all finite values equal zero."),
            ("nonzero", "At least one finite nonzero module economic value."),
            ("no_numeric_content", "No finite listed module economic value."),
            ("invalid_numeric", "Any non-finite or unparseable module economic value."),
        ):
            coverage_rows.append(
                coverage_row("MODULE", module, measure, definition,
                             coverage_measure_mask(data, measure), population,
                             "PENDING_FINAL_FEATURE_APPROVAL")
            )

    module_content_masks = [module_data[name]["content"] for name in ("debitcard", "deposit", "tax")]
    content_count = sum(mask.astype(np.int8) for mask in module_content_masks)
    for measure, mask in (
        ("at_least_one_module_numeric_content", content_count >= 1),
        ("exactly_one_module_numeric_content", content_count == 1),
        ("exactly_two_modules_numeric_content", content_count == 2),
        ("all_three_modules_numeric_content", content_count == 3),
    ):
        coverage_rows.append(
            coverage_row("CROSS_MODULE", "debitcard_deposit_tax", measure,
                         "Cross-module count under the observed finite numeric-content definition.",
                         mask, population, "MEASURED")
        )

    overlap_rows = complete_pattern_rows(
        "MODULE_NUMERIC_CONTENT", ["debitcard", "deposit", "tax"], module_content_masks
    )
    overlap_rows.extend(
        complete_pattern_rows(
            "MODULE_FIELD_EVIDENCE", ["debitcard", "deposit", "tax"],
            [module_data[name]["field_evidence"] for name in ("debitcard", "deposit", "tax")],
        )
    )
    overlap_rows.extend(
        complete_pattern_rows(
            "DEBITCARD_COMPONENT_NUMERIC_CONTENT",
            ["debitcard_1", "other_1_debitcard"],
            [components["debitcard__debitcard_1"]["content"], components["debitcard__other_1"]["content"]],
        )
    )
    overlap_rows.extend(
        complete_pattern_rows(
            "DEPOSIT_COMPONENT_NUMERIC_CONTENT",
            ["deposit_1", "other_1_deposit"],
            [components["deposit__deposit_1"]["content"], components["deposit__other_1"]["content"]],
        )
    )
    tax_component_order = ["static_cb_0", "tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1"]
    tax_component_masks = [components[f"tax__{name}"]["content"] for name in tax_component_order]
    overlap_rows.extend(
        complete_pattern_rows("TAX_COMPONENT_NUMERIC_CONTENT", tax_component_order, tax_component_masks)
    )
    detail_tax = tax_component_masks[1] | tax_component_masks[2] | tax_component_masks[3]
    overlap_rows.extend(
        complete_pattern_rows(
            "TAX_STATIC_SUMMARY_VS_ANY_DETAIL_NUMERIC_CONTENT",
            ["static_cb_0_tax_summary", "any_tax_detail"],
            [tax_component_masks[0], detail_tax],
        )
    )
    for pattern_type in sorted({row["pattern_type"] for row in overlap_rows}):
        pattern_sum = sum(row["count"] for row in overlap_rows if row["pattern_type"] == pattern_type)
        add_check(
            f"complete_pattern_sum::{pattern_type}", pattern_sum == population,
            f"pattern_sum={pattern_sum}, N={population}",
        )

    month_labels = base_dates.dt.strftime("%Y-%m").fillna("UNKNOWN_DATE")
    month_rows: list[dict[str, Any]] = []
    for month in sorted(month_labels.unique(), key=lambda value: (value == "UNKNOWN_DATE", value)):
        month_mask = month_labels.eq(month).to_numpy(bool)
        month_n = int(np.count_nonzero(month_mask))
        debit = int(np.count_nonzero(month_mask & module_data["debitcard"]["content"]))
        deposit = int(np.count_nonzero(month_mask & module_data["deposit"]["content"]))
        tax = int(np.count_nonzero(month_mask & module_data["tax"]["content"]))
        any_ad_mask = module_data["debitcard"]["content"] | module_data["deposit"]["content"] | module_data["tax"]["content"]
        all_ad_mask = module_data["debitcard"]["content"] & module_data["deposit"]["content"] & module_data["tax"]["content"]
        any_ad = int(np.count_nonzero(month_mask & any_ad_mask))
        all_ad = int(np.count_nonzero(month_mask & all_ad_mask))
        month_rows.append(
            {
                "calendar_month": month,
                "base_applications": month_n,
                "debitcard_numeric_content_count": debit,
                "debitcard_numeric_content_rate": debit / month_n if month_n else None,
                "deposit_numeric_content_count": deposit,
                "deposit_numeric_content_rate": deposit / month_n if month_n else None,
                "tax_numeric_content_count": tax,
                "tax_numeric_content_rate": tax / month_n if month_n else None,
                "any_module_numeric_content_count": any_ad,
                "any_module_numeric_content_rate": any_ad / month_n if month_n else None,
                "all_three_numeric_content_count": all_ad,
                "all_three_numeric_content_rate": all_ad / month_n if month_n else None,
                "definition": "Calendar month of parseable base date_decision; UNKNOWN_DATE retains missing/unparseable dates.",
            }
        )
    add_check(
        "calendar_month_population_reconciliation",
        sum(row["base_applications"] for row in month_rows) == population,
        "Monthly base-application counts must sum to N.",
    )

    add_check(
        "contact_fields_excluded_from_ad_flags",
        not any(any(field in name for field in CONTACT_FIELDS) for name in flag_columns),
        "No contact-derived field contributes a flag column or module pattern.",
    )
    flags = assemble_flags(base["case_id"], flag_columns)
    add_check(
        "application_flags_base_key_reconciliation",
        len(flags) == population
        and flags["case_id"].notna().all()
        and flags["case_id"].nunique() == population
        and pd.Index(flags["case_id"]).equals(base_index),
        "Flags table must have exactly the ordered unique non-null base key set without row expansion.",
    )
    coverage_bounds_ok = all(
        0 <= row["population_count"] <= population for row in coverage_rows
    )
    add_check(
        "coverage_counts_within_population", coverage_bounds_ok,
        "Every source/module/component coverage count must be between zero and N.",
    )

    review_rows = feature_review_rows(descriptions, structures, date_tie_diagnostics)
    after_state = collect_file_state(input_paths)
    input_comparison = compare_file_state(before_state, after_state)
    add_check(
        "input_size_mtime_unchanged", input_comparison["all_unchanged"],
        "Scoped raw files, dictionary, and Task 06 report sizes/mtimes compared before and after.",
    )
    add_check(
        "traditional_history_and_person_tables",
        None,
        "NOT_CHECKED_IN_BATCH1: credit_bureau, applprev, person, and depth-2 row values were not read.",
    )
    add_check(
        "point_in_time_eligibility",
        None,
        "NOT_CHECKED: observed dates and offsets do not prove data availability at decision time.",
    )
    add_check(
        "substantive_feature_approval",
        None,
        "NOT_CHECKED: this audit does not approve model fields, aggregations, or preprocessing.",
    )

    failures = [row for row in validation_checks if row["status"] == "FAIL"]
    execution_status = "COMPLETED" if not failures else "COMPLETED_WITH_CHECK_FAILURES"
    flags_schema = [
        {"name": field.name, "arrow_type": str(field.type), "description": (
            "Base application join key" if field.name == "case_id" else "Audit count or boolean flag; not an approved model feature"
        )}
        for field in pa.Table.from_pandas(flags, preserve_index=False).schema
    ]
    report_text = markdown_report(
        population, module_data, components, structures, field_audit_rows,
        input_comparison["all_unchanged"], validation_checks,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    refuse_existing_outputs(args.output_dir)
    write_csv(targets["source_structure.csv"], list(source_structure_rows[0]), source_structure_rows)
    write_csv(targets["ad_coverage.csv"], list(coverage_rows[0]), coverage_rows)
    write_csv(targets["ad_overlap.csv"], list(overlap_rows[0]), overlap_rows)
    write_csv(targets["field_value_audit.csv"], list(field_audit_rows[0]), field_audit_rows)
    write_csv(targets["coverage_by_month.csv"], list(month_rows[0]), month_rows)
    write_csv(targets["feature_review.csv"], list(review_rows[0]), review_rows)
    pq.write_table(
        pa.Table.from_pandas(flags, preserve_index=False),
        targets["application_ad_audit_flags.parquet"],
        compression="zstd",
    )
    targets["audit_report.md"].write_text(report_text, encoding="utf-8")

    output_manifest = []
    for name in OUTPUT_NAMES:
        path = targets[name]
        output_manifest.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "status": "PRODUCED" if path.exists() else ("PENDING_SELF_WRITE" if name == "audit_summary.json" else "NOT_PRODUCED"),
            }
        )
    summary = {
        "task_identifier": TASK_ID,
        "script_version": SCRIPT_VERSION,
        "execution_status": execution_status,
        "utc_execution_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "paths": {
            "train_directory": str(args.train_dir.resolve()),
            "dictionary": str(args.dictionary.resolve()),
            "task06_directory": str(args.task06_dir.resolve()),
            "output_directory": str(args.output_dir.resolve()),
        },
        "verified_base": {
            "physical_rows": int(len(base)),
            "null_case_id_rows": base_null,
            "distinct_nonnull_case_ids": base_distinct,
            "duplicate_case_id_excess_rows": base_duplicate_excess,
            "coverage_denominator_N": population,
            "date_decision_native_missing": int(np.count_nonzero(base_date_info["native_missing"])),
            "date_decision_empty": int(np.count_nonzero(base_date_info["empty"])),
            "date_decision_parse_failures": int(np.count_nonzero(base_date_info["parse_failure"])),
            "date_decision_min": base_dates.min().isoformat() if base_dates.notna().any() else None,
            "date_decision_max": base_dates.max().isoformat() if base_dates.notna().any() else None,
        },
        "definitions": {
            "unit": "application record keyed by case_id; not a distinct natural person",
            "observed_numeric_content": "At least one finite observation in assigned economic fields; includes finite zero and negative values.",
            "field_evidence": "At least one populated assigned economic or auxiliary field; not confirmed economic activity.",
            "numeric_states": ["missing", "unparseable", "nonfinite", "zero", "positive", "negative"],
            "quantile_method": QUANTILE_METHOD,
            "float_equality": {"rtol": FLOAT_EQUAL_RTOL, "atol": FLOAT_EQUAL_ATOL},
            "num_group1": "Observed local record index only; not assumed chronological, globally unique, or cross-table joinable.",
        },
        "discovered_inputs": {
            "task06_report_status": task06_summary.get("report_status"),
            "source_files": group_to_files,
            "schema_validation": schema_validation,
        },
        "source_structures": structures,
        "shard_diagnostics": shard_diagnostics,
        "identical_selected_projection_excess_rows": projection_duplicates,
        "date_latest_tie_diagnostics": date_tie_diagnostics,
        "module_counts": {
            module: {
                name: int(np.count_nonzero(mask))
                for name, mask in data.items()
            }
            for module, data in module_data.items()
        },
        "component_counts": {
            name: {
                key: int(np.count_nonzero(mask))
                for key, mask in data.items()
            }
            for name, data in components.items()
        },
        "cross_module_counts": {
            "at_least_one_numeric_content": int(np.count_nonzero(content_count >= 1)),
            "exactly_one_numeric_content": int(np.count_nonzero(content_count == 1)),
            "exactly_two_numeric_content": int(np.count_nonzero(content_count == 2)),
            "all_three_numeric_content": int(np.count_nonzero(content_count == 3)),
            "none_numeric_content": int(np.count_nonzero(content_count == 0)),
        },
        "application_flags_schema": flags_schema,
        "application_flags_note": "Audit-only base-keyed counts/flags; no target, imputed predictors, contact-derived model features, or model-ready table.",
        "validation_results": validation_checks,
        "validation_status_counts": dict(Counter(row["status"] for row in validation_checks)),
        "input_before_after_comparison": input_comparison,
        "caveats": [
            "Source-container presence is not economic-content evidence.",
            "Finite content is a reproducible observation rule, not proof of business validity or feature approval.",
            "Absent records do not establish no account, no income, no activity, or no credit history.",
            "Row unit and aggregation meaning remain unresolved for several detail sources.",
            "Dates do not by themselves prove point-in-time availability or absence of leakage.",
            "Application case_id is not interpreted as natural-person identity.",
            "All-missing groups must remain missing in future aggregation; partial sums are not automatically complete totals.",
        ],
        "deferred_checks": [
            "Traditional credit-bureau and internal previous-application row-value audit (Batch 2)",
            "Person-role and depth-2 table audit (Batch 2)",
            "Final field eligibility and T/AD feature registry",
            "Feature aggregation construction and training-only preprocessing",
            "Data partitions, model training, calibration, approval, and economic analyses",
        ],
        "output_manifest": output_manifest,
    }
    targets["audit_summary.json"].write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "execution_status": execution_status,
                "verified_base_applications": population,
                "module_numeric_content_counts": {
                    name: int(np.count_nonzero(data["content"]))
                    for name, data in module_data.items()
                },
                "any_module_numeric_content": int(np.count_nonzero(content_count >= 1)),
                "all_three_numeric_content": int(np.count_nonzero(content_count == 3)),
                "validation_status_counts": dict(Counter(row["status"] for row in validation_checks)),
                "outputs": [str(targets[name]) for name in OUTPUT_NAMES],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
