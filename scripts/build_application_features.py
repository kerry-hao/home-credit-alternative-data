#!/usr/bin/env python3
"""Task 08: construct the approved application-level T and AD feature table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shlex
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from audit_ad_coverage import classify_numeric, parse_date_values


VERSION = "1.0.0"
EXPECTED_N = 1_526_659
BATCH_SIZE = 200_000
CORRECTION_METRICS = {
    "applications_latest_date_tied",
    "applications_latest_tie_payload_conflict",
}

AUDIT_OUTPUTS = (
    "raw_candidate_decisions.csv",
    "feature_registry.csv",
    "feature_sets.json",
    "feature_value_audit.csv",
    "ad_coverage.csv",
    "ad_overlap.csv",
    "feature_examples.csv",
    "aggregation_examples.csv",
    "validation_results.csv",
    "batch2_correction_original_rows.csv",
    "build_summary.json",
    "build_report.md",
)

T_CANDIDATES = [
    ("T01", "static_0", "maininc_215A", "t__recorded_primary_income", "Numeric", "Recorded primary income"),
    ("T02", "person_1", "mainoccupationinc_384A", "t__applicant_main_income", "Numeric", "Recorded applicant main income"),
    ("T03", "static_0", "credamount_770A", "t__requested_credit_amount", "Numeric", "Requested loan amount or credit-card limit"),
    ("T04", "static_0", "annuity_780A", "t__current_application_annuity", "Numeric", "Monthly annuity of the current application"),
    ("T05", "static_0", "credtype_322L", "t__credit_product_type", "Categorical", "Credit product type"),
    ("T06", "person_1", "incometype_1044T", "t__applicant_income_type", "Categorical", "Applicant income type"),
    ("T07", "person_1", "education_927M", "t__applicant_education", "Categorical", "Applicant education category"),
    ("T08", "static_0", "currdebt_22A", "t__current_debt", "Numeric", "Recorded current debt"),
    ("T09", "static_0", "totaldebt_9A", "t__total_debt", "Numeric", "Recorded total debt"),
    ("T10", "static_0", "numactivecreds_622L", "t__observed_active_credit_count", "Numeric count", "Observed active-credit count"),
    ("T11", "static_0", "maxdpdlast3m_392P", "t__max_dpd_last_3m", "Numeric", "Maximum DPD in the documented 3-month window"),
    ("T12", "static_0", "maxdpdlast12m_727P", "t__max_dpd_last_12m", "Numeric", "Maximum DPD in the documented 12-month window"),
    ("T13", "static_0", "maxdpdlast24m_143P", "t__max_dpd_last_24m", "Numeric", "Maximum DPD in the documented 24-month window"),
    ("T14", "static_cb_0", "days30_165L", "t__bureau_queries_30d", "Numeric count", "Bureau query count in the documented 30-day window"),
    ("T15", "static_cb_0", "days360_512L", "t__bureau_queries_360d", "Numeric count", "Bureau query count in the documented 360-day window"),
    ("T16", "static_0", "pmtnum_254L", "t__client_loan_payment_count", "Numeric count", "Client loan-payment count"),
    ("T17", "static_0", "applications30d_658L", "t__applications_30d", "Numeric count", "Client application count in the documented 30-day window"),
    ("T18", "static_0", "numcontrs3months_479L", "t__contracts_3m", "Numeric count", "Contract count in the documented 3-month window"),
    ("T19", "static_0", "numactiverelcontr_750L", "t__active_revolving_credit_count", "Numeric count", "Active revolving-credit count"),
    ("T20", "static_0", "lastapprcredamount_781A", "t__last_approved_credit_amount", "Numeric", "Previous application credit amount"),
    ("T21", "static_0", "numinstpaidlastcontr_4325080L", "t__paid_installments_last_contract", "Numeric count", "Paid installments on the last contract"),
]

AD_CANDIDATES = [
    ("AD_D01", "debit", "debitcard_1", "last30dayturnover_651A", "aggregation"),
    ("AD_D02", "debit", "debitcard_1", "last180dayturnover_1134A", "aggregation"),
    ("AD_D03", "debit", "debitcard_1", "last180dayaveragebalance_704A", "aggregation"),
    ("AD_D04", "debit", "other_1", "amtdebitincoming_4809443A", "direct"),
    ("AD_D05", "debit", "other_1", "amtdebitoutgoing_4809440A", "direct"),
    ("AD_P01", "deposit", "deposit_1", "amount_416A", "aggregation"),
    ("AD_P02", "deposit", "other_1", "amtdepositbalance_4809441A", "direct"),
    ("AD_P03", "deposit", "other_1", "amtdepositincoming_4809444A", "direct"),
    ("AD_P04", "deposit", "other_1", "amtdepositoutgoing_4809442A", "direct"),
    ("AD_T01", "tax", "static_cb_0", "pmtscount_423L", "direct"),
    ("AD_T02", "tax", "static_cb_0", "pmtssum_45A", "direct"),
    ("AD_T03", "tax", "static_cb_0", "pmtcount_693L", "direct"),
    ("AD_T04", "tax", "static_cb_0", "pmtcount_4527229L", "direct"),
    ("AD_T05", "tax", "static_cb_0", "pmtcount_4955617L", "direct"),
    ("AD_T06", "tax", "static_cb_0", "pmtaverage_3A", "direct"),
    ("AD_T07", "tax", "static_cb_0", "pmtaverage_4527227A", "direct"),
    ("AD_T08", "tax", "static_cb_0", "pmtaverage_4955615A", "direct"),
    ("AD_T09", "tax", "tax_registry_a_1", "amount_4527230A", "aggregation"),
    ("AD_T10", "tax", "tax_registry_b_1", "amount_4917619A", "aggregation"),
    ("AD_T11", "tax", "tax_registry_c_1", "pmtamount_36A", "aggregation"),
]

GROUPS = {
    "base": "train_base",
    "static_0": "train_static_0",
    "static_cb_0": "train_static_cb_0",
    "person_1": "train_person_1",
    "debitcard_1": "train_debitcard_1",
    "deposit_1": "train_deposit_1",
    "other_1": "train_other_1",
    "tax_registry_a_1": "train_tax_registry_a_1",
    "tax_registry_b_1": "train_tax_registry_b_1",
    "tax_registry_c_1": "train_tax_registry_c_1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--dictionary", required=True, type=Path)
    parser.add_argument("--task06-dir", required=True, type=Path)
    parser.add_argument("--batch1-dir", required=True, type=Path)
    parser.add_argument("--batch2-dir", required=True, type=Path)
    parser.add_argument("--audit-output-dir", required=True, type=Path)
    parser.add_argument("--feature-output", required=True, type=Path)
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
    if value is None:
        return None
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str] | None = None) -> None:
    if headers is None:
        if not rows:
            raise ValueError(f"Cannot infer headers for empty CSV: {path}")
        headers = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in headers})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_state(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result = []
    for path in sorted(set(paths), key=lambda item: str(item.resolve())):
        stat = path.stat()
        result.append({"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return result


def compare_states(before: list[dict[str, Any]], after: list[dict[str, Any]], expected_changed: set[str]) -> dict[str, Any]:
    left = {row["path"]: row for row in before}
    right = {row["path"]: row for row in after}
    details = []
    for path in sorted(set(left) | set(right)):
        changed = left.get(path) != right.get(path)
        expected = path in expected_changed
        details.append({"path": path, "changed": changed, "expected_changed": expected,
                        "status": "PASS" if changed == expected else "FAIL",
                        "before": left.get(path), "after": right.get(path)})
    return {"method": "file size and mtime_ns; lightweight, non-cryptographic",
            "all_changes_as_expected": all(row["status"] == "PASS" for row in details), "details": details}


def numeric_clean(series: pd.Series) -> tuple[np.ndarray, dict[str, int], dict[str, np.ndarray]]:
    classified = classify_numeric(series)
    values = np.full(len(series), np.nan, dtype=np.float64)
    values[classified["finite"]] = classified["values"][classified["finite"]]
    counts = {name: int(np.count_nonzero(classified[name])) for name in
              ("missing", "unparseable", "nonfinite", "zero", "positive", "negative", "finite", "nonzero")}
    return values, counts, classified


class NumericReducer:
    """Finite-only application reducer that preserves source and numeric states."""

    def __init__(self, population: int, fields: list[str]) -> None:
        self.population = population
        self.fields = fields
        self.source_count = np.zeros(population, dtype=np.int32)
        self.finite_count = {field: np.zeros(population, dtype=np.int32) for field in fields}
        self.nonzero_count = {field: np.zeros(population, dtype=np.int32) for field in fields}
        self.missing_count = {field: np.zeros(population, dtype=np.int32) for field in fields}
        self.invalid_count = {field: np.zeros(population, dtype=np.int32) for field in fields}
        self.sums = {field: np.zeros(population, dtype=np.float64) for field in fields}
        self.maxima = {field: np.full(population, -np.inf, dtype=np.float64) for field in fields}
        self.row_states = {field: Counter() for field in fields}

    @staticmethod
    def _add(target: np.ndarray, positions: np.ndarray, mask: np.ndarray) -> None:
        if mask.any():
            np.add.at(target, positions[mask], 1)

    def update(self, frame: pd.DataFrame, positions: np.ndarray, matched: np.ndarray) -> None:
        np.add.at(self.source_count, positions[matched], 1)
        for field in self.fields:
            classified = classify_numeric(frame[field])
            for state in ("missing", "unparseable", "nonfinite", "zero", "positive", "negative"):
                self.row_states[field][state] += int(np.count_nonzero(matched & classified[state]))
            finite = matched & classified["finite"]
            nonzero = matched & classified["nonzero"]
            missing = matched & classified["missing"]
            invalid = matched & (classified["unparseable"] | classified["nonfinite"])
            self._add(self.finite_count[field], positions, finite)
            self._add(self.nonzero_count[field], positions, nonzero)
            self._add(self.missing_count[field], positions, missing)
            self._add(self.invalid_count[field], positions, invalid)
            if finite.any():
                vals = classified["values"][finite]
                pos = positions[finite]
                np.add.at(self.sums[field], pos, vals)
                np.maximum.at(self.maxima[field], pos, vals)

    def result(self, field: str) -> dict[str, Any]:
        count = self.finite_count[field]
        mean = np.full(self.population, np.nan, dtype=np.float64)
        maximum = np.full(self.population, np.nan, dtype=np.float64)
        has = count > 0
        mean[has] = self.sums[field][has] / count[has]
        maximum[has] = self.maxima[field][has]
        return {"mean": mean, "max": maximum, "finite_count": count,
                "nonzero_count": self.nonzero_count[field], "missing_count": self.missing_count[field],
                "invalid_count": self.invalid_count[field], "row_states": dict(self.row_states[field])}


def positions_for(case_ids: pd.Series, base_index: pd.Index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonnull = case_ids.notna().to_numpy(bool)
    positions = np.full(len(case_ids), -1, dtype=np.int64)
    if nonnull.any():
        positions[nonnull] = base_index.get_indexer(case_ids[nonnull])
    return positions, nonnull & (positions >= 0), nonnull & (positions < 0)


def aggregate_history(
    paths: list[Path], fields: list[str], base_index: pd.Index,
) -> tuple[NumericReducer, dict[str, Any]]:
    reducer = NumericReducer(len(base_index), fields)
    codes: list[np.ndarray] = []
    physical_rows = null_case = orphan = null_index = invalid_index = 0
    projections = ["case_id", "num_group1", *fields]
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=projections, use_threads=True):
            frame = batch.to_pandas()
            physical_rows += len(frame)
            positions, matched, orphan_mask = positions_for(frame["case_id"], base_index)
            null_case += int(frame["case_id"].isna().sum())
            orphan += int(orphan_mask.sum())
            idx = pd.to_numeric(frame["num_group1"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
            null_index += int(frame["num_group1"].isna().sum())
            valid_index = np.isfinite(idx) & (idx == np.floor(idx)) & (idx >= 0) & (idx <= np.iinfo(np.uint32).max)
            invalid_index += int(np.count_nonzero(~frame["num_group1"].isna().to_numpy(bool) & ~valid_index))
            valid_code = matched & valid_index
            if valid_code.any():
                codes.append((positions[valid_code].astype(np.uint64) << np.uint64(32)) | idx[valid_code].astype(np.uint64))
            reducer.update(frame, positions, matched)
    all_codes = np.concatenate(codes) if codes else np.empty(0, dtype=np.uint64)
    _, counts = np.unique(all_codes, return_counts=True)
    structure = {
        "physical_rows": physical_rows,
        "matched_rows": int(reducer.source_count.sum()),
        "matched_applications": int(np.count_nonzero(reducer.source_count)),
        "null_case_id_rows": null_case,
        "orphan_rows": orphan,
        "null_num_group1_rows": null_index,
        "invalid_num_group1_rows": invalid_index,
        "duplicate_composite_key_distinct_count": int(np.count_nonzero(counts > 1)),
        "duplicate_composite_key_excess_rows": int(np.sum(np.clip(counts - 1, 0, None))) if counts.size else 0,
        "projected_columns": projections,
    }
    return reducer, structure


def select_applicant_rows(paths: list[Path], base_index: pd.Index, fields: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    population = len(base_index)
    numeric_fields = [field for field in fields if field == "mainoccupationinc_384A"]
    categorical_fields = [field for field in fields if field not in numeric_fields]
    data: dict[str, Any] = {field: np.full(population, np.nan) for field in numeric_fields}
    data.update({field: np.full(population, None, dtype=object) for field in categorical_fields})
    selected_count = np.zeros(population, dtype=np.int32)
    source_count = np.zeros(population, dtype=np.int32)
    personindex_zero = personindex_nonzero = personindex_missing = 0
    codes: list[np.ndarray] = []
    columns = ["case_id", "num_group1", "personindex_1023L", *fields]
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_SIZE, columns=columns):
            frame = batch.to_pandas()
            positions, matched, orphan = positions_for(frame["case_id"], base_index)
            if orphan.any() or frame["case_id"].isna().any():
                raise RuntimeError("person_1 contains null or orphan case_id values")
            np.add.at(source_count, positions[matched], 1)
            idx = pd.to_numeric(frame["num_group1"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
            valid = matched & np.isfinite(idx) & (idx == np.floor(idx)) & (idx >= 0)
            codes.append((positions[valid].astype(np.uint64) << np.uint64(32)) | idx[valid].astype(np.uint64))
            selected = matched & np.isfinite(idx) & (idx == 0)
            np.add.at(selected_count, positions[selected], 1)
            pi = pd.to_numeric(frame["personindex_1023L"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
            personindex_zero += int(np.count_nonzero(selected & (pi == 0)))
            personindex_nonzero += int(np.count_nonzero(selected & np.isfinite(pi) & (pi != 0)))
            personindex_missing += int(np.count_nonzero(selected & ~np.isfinite(pi)))
            pos = positions[selected]
            for field in numeric_fields:
                cleaned, _, _ = numeric_clean(frame.loc[selected, field].reset_index(drop=True))
                data[field][pos] = cleaned
            for field in categorical_fields:
                values = frame.loc[selected, field].astype(object).to_numpy()
                data[field][pos] = values
    all_codes = np.concatenate(codes)
    _, code_counts = np.unique(all_codes, return_counts=True)
    result = pd.DataFrame(data, index=base_index)
    details = {
        "source_applications": int(np.count_nonzero(source_count)),
        "selected_zero": int(np.count_nonzero(selected_count == 0)),
        "selected_one": int(np.count_nonzero(selected_count == 1)),
        "selected_multiple": int(np.count_nonzero(selected_count > 1)),
        "personindex_zero": personindex_zero,
        "personindex_nonzero": personindex_nonzero,
        "personindex_missing": personindex_missing,
        "duplicate_composite_key_distinct_count": int(np.count_nonzero(code_counts > 1)),
        "selected_count": selected_count,
    }
    return result, details


def ratio_review(numerator: np.ndarray, denominator: np.ndarray, name: str, formula: str) -> dict[str, Any]:
    numerator_finite = np.isfinite(numerator)
    denominator_finite = np.isfinite(denominator)
    eligible = numerator_finite & denominator_finite & (denominator > 0)
    return {
        "candidate": name, "formula": formula, "decision": "DEFERRED_UNIT_UNCERTAINTY",
        "eligible_count_if_materialized": int(eligible.sum()),
        "undefined_count_if_materialized": int((~eligible).sum()),
        "denominator_missing_or_nonfinite_count": int((~denominator_finite).sum()),
        "denominator_zero_count": int(np.count_nonzero(denominator_finite & (denominator == 0))),
        "denominator_negative_count": int(np.count_nonzero(denominator_finite & (denominator < 0))),
        "evidence": "Dictionary describes applicant main income as an amount but gives no income period or cross-field transformation comparability.",
        "reason": "Existing evidence does not establish comparable monetary scale; no unit conversion or ratio was materialized.",
    }


def family_states(evidence: np.ndarray, content: np.ndarray, nonzero: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "NO_SOURCE_OR_EVIDENCE": ~evidence,
        "SOURCE_OR_EVIDENCE_PRESENT_NO_FINITE_CONTENT": evidence & ~content,
        "FINITE_CONTENT_ZERO_ONLY": content & ~nonzero,
        "FINITE_CONTENT_WITH_NONZERO": nonzero,
    }


def exact_duplicate_groups(frame: pd.DataFrame, columns: list[str]) -> list[list[str]]:
    signatures: dict[str, list[str]] = defaultdict(list)
    for column in columns:
        hashed = pd.util.hash_pandas_object(frame[column], index=False, categorize=True).to_numpy(np.uint64)
        signature = hashlib.sha256(hashed.tobytes()).hexdigest()
        signatures[signature].append(column)
    groups = []
    for candidates in signatures.values():
        if len(candidates) < 2:
            continue
        verified = []
        while candidates:
            first, *rest = candidates
            same = [item for item in rest if frame[first].equals(frame[item])]
            if same:
                groups.append([first, *same])
            candidates = [item for item in rest if item not in same]
    return groups


def correction_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    selected = [row for row in rows if row["section"] == "TAX_DETAIL_TIMING" and row["metric"] in CORRECTION_METRICS]
    if len(selected) != 6:
        raise RuntimeError(f"Expected six Batch 2 correction rows, found {len(selected)}")
    original = [dict(row) for row in selected]
    if all(row["status"] == "NOT_CHECKED" and row["value"] == "" for row in selected):
        return original, rows, "ALREADY_CORRECTED"
    if not all(row["status"] == "MEASURED" and row["value"] == "0" for row in selected):
        raise RuntimeError("Batch 2 correction rows are neither the known erroneous state nor the corrected state")
    corrected = []
    for row in rows:
        updated = dict(row)
        if row in selected:
            updated["value"] = ""
            updated["status"] = "NOT_CHECKED"
            updated["definition"] = "Latest-date tracking was disabled for this audit; no tie value was computed."
        corrected.append(updated)
    return original, corrected, "CORRECTED_SIX_ROWS"


def feature_audit_rows(frame: pd.DataFrame, registry: list[dict[str, Any]], invalid_counts: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for spec in registry:
        if spec["feature_role"] == "METADATA":
            continue
        name = spec["feature_name"]
        series = frame[name]
        categorical = spec["dtype"] == "category_string"
        row = {
            "feature_name": name, "family": spec["feature_family"], "role": spec["feature_role"],
            "type": "categorical" if categorical else "numeric", "N": len(frame),
            "missing_count": int(series.isna().sum()), "missing_rate": float(series.isna().mean()),
            "finite_count": None, "finite_rate": None, "zero_count": None, "zero_rate": None,
            "positive_count": None, "negative_count": None,
            "invalid_source_observation_count": invalid_counts.get(name, 0),
            "nonfinite_materialized_count": None, "min": None, "p01": None, "median": None,
            "p99": None, "max": None, "populated_count": None, "populated_rate": None,
            "distinct_category_count": None, "top_categories_json": None,
            "denominator_definition": "Full verified base application population",
            "status": "MEASURED", "undefined_reason": "",
        }
        if categorical:
            populated = series.notna()
            frequencies = series[populated].astype(str).value_counts(dropna=False)
            row.update({
                "populated_count": int(populated.sum()), "populated_rate": float(populated.mean()),
                "distinct_category_count": int(frequencies.size),
                "top_categories_json": json.dumps(
                    [{"value": str(key), "count": int(value)} for key, value in frequencies.head(20).items()],
                    ensure_ascii=False, separators=(",", ":"),
                ),
                "undefined_reason": "Numeric statistics do not apply to categorical features.",
            })
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(np.float64)
            materialized_nonfinite = np.isinf(values)
            finite = np.isfinite(values)
            finite_values = values[finite]
            row.update({
                "finite_count": int(finite.sum()), "finite_rate": float(finite.mean()),
                "zero_count": int(np.count_nonzero(finite_values == 0)),
                "zero_rate": float(np.count_nonzero(finite_values == 0) / len(frame)),
                "positive_count": int(np.count_nonzero(finite_values > 0)),
                "negative_count": int(np.count_nonzero(finite_values < 0)),
                "nonfinite_materialized_count": int(materialized_nonfinite.sum()),
            })
            if finite_values.size:
                q = np.quantile(finite_values, [0.01, 0.5, 0.99], method="linear")
                row.update({"min": float(finite_values.min()), "p01": float(q[0]),
                            "median": float(q[1]), "p99": float(q[2]), "max": float(finite_values.max())})
            else:
                row["status"] = "UNDEFINED_NO_FINITE_VALUES"
                row["undefined_reason"] = "No finite materialized values."
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    output_paths = {name: args.audit_output_dir / name for name in AUDIT_OUTPUTS}
    existing = [str(path) for path in [*output_paths.values(), args.feature_output] if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing Task 08 outputs: {existing}")
    required = [args.train_dir, args.dictionary, args.task06_dir, args.batch1_dir, args.batch2_dir]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    table_inventory_path = args.task06_dir / "table_inventory.csv"
    field_inventory_path = args.task06_dir / "field_inventory.csv"
    flags_path = args.batch1_dir / "application_ad_audit_flags.parquet"
    b1_summary_path = args.batch1_dir / "audit_summary.json"
    b2_csv_path = args.batch2_dir / "ad_followup_results.csv"
    b2_summary_path = args.batch2_dir / "audit_summary.json"
    for path in (table_inventory_path, field_inventory_path, flags_path, b1_summary_path, b2_csv_path, b2_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    inventory = read_csv(table_inventory_path)
    fields_inventory = read_csv(field_inventory_path)
    dictionary_rows = read_csv(args.dictionary)
    descriptions = {row["Variable"]: row["Description"] for row in dictionary_rows}
    group_files: dict[str, list[str]] = {}
    for source, group in GROUPS.items():
        group_files[source] = sorted(row["file_name"] for row in inventory if row["provisional_table_group"] == group)
        if not group_files[source]:
            raise RuntimeError(f"Task 06 inventory has no files for {source}")
        for file_name in group_files[source]:
            if not (args.train_dir / file_name).is_file():
                raise FileNotFoundError(args.train_dir / file_name)

    required_fields: dict[str, set[str]] = defaultdict(set)
    required_fields["base"].update({"case_id", "date_decision", "WEEK_NUM", "MONTH"})
    for _, source, raw, _, _, _ in T_CANDIDATES:
        required_fields[source].add(raw)
    required_fields["person_1"].update({"case_id", "num_group1", "personindex_1023L"})
    for _, _, source, raw, use in AD_CANDIDATES:
        required_fields[source].add(raw)
        required_fields[source].add("case_id")
        if use == "aggregation":
            required_fields[source].add("num_group1")
    required_fields["static_0"].add("case_id")
    required_fields["static_cb_0"].add("case_id")
    required_fields["other_1"].add("case_id")
    inventory_schema: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in fields_inventory:
        inventory_schema[(row["provisional_table_group"], row["file_name"])].add(row["field_name"])
    missing_schema = []
    for source, required_source_fields in required_fields.items():
        group = GROUPS[source]
        for file_name in group_files[source]:
            missing_schema.extend((source, file_name, field) for field in required_source_fields
                                  if field not in inventory_schema[(group, file_name)])
    if missing_schema:
        raise RuntimeError(f"Required schema fields missing: {missing_schema}")

    raw_paths = [args.train_dir / name for source in GROUPS for name in group_files[source]]
    prior_paths = [args.dictionary, table_inventory_path, field_inventory_path, flags_path,
                   b1_summary_path, b2_csv_path, b2_summary_path]
    before_state = file_state([*raw_paths, *prior_paths])

    validation: list[dict[str, Any]] = []
    def check(check_id: str, passed: bool | None, observed: Any, expected: Any, explanation: str) -> None:
        validation.append({"check_id": check_id, "status": "NOT_CHECKED" if passed is None else ("PASS" if passed else "FAIL"),
                           "observed": json.dumps(json_safe(observed), ensure_ascii=False, separators=(",", ":")) if isinstance(observed, (dict, list)) else observed,
                           "expected": json.dumps(json_safe(expected), ensure_ascii=False, separators=(",", ":")) if isinstance(expected, (dict, list)) else expected,
                           "explanation": explanation})

    base_path = args.train_dir / group_files["base"][0]
    base = pq.read_table(base_path, columns=["case_id", "date_decision", "WEEK_NUM", "MONTH"]).to_pandas()
    base_index = pd.Index(base["case_id"])
    population = len(base)
    check("base_population", population == EXPECTED_N, population, EXPECTED_N, "Actual TRAIN base rows.")
    check("base_case_id_nonnull", not base["case_id"].isna().any(), int(base["case_id"].isna().sum()), 0, "Base key null count.")
    check("base_case_id_unique", base["case_id"].is_unique, int(base["case_id"].duplicated().sum()), 0, "Base key duplicate count.")
    if not base["case_id"].is_unique or base["case_id"].isna().any():
        raise RuntimeError("Base keys are not unique and non-null")

    flag_columns = [
        "case_id",
        *[f"module__{family}__{name}" for family in ("debitcard", "deposit", "tax")
          for name in ("content", "nonzero", "field_evidence")],
        *[f"source__{source}__present" for source in ("debitcard_1", "deposit_1", "other_1", "tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1")],
    ]
    flags = pq.read_table(flags_path, columns=flag_columns).to_pandas()
    order_match = np.array_equal(flags["case_id"].to_numpy(), base["case_id"].to_numpy())
    check("batch1_exact_key_order", order_match, order_match, True, "Batch 1 flags preserve the exact base key order.")
    if not order_match:
        raise RuntimeError("Batch 1 flags do not match base key order")

    features = base[["case_id", "date_decision", "WEEK_NUM", "MONTH"]].copy()
    registry: list[dict[str, Any]] = []
    invalid_counts: dict[str, int] = {}
    raw_masks: dict[str, dict[str, np.ndarray]] = {
        family: {"content": np.zeros(population, bool), "nonzero": np.zeros(population, bool)}
        for family in ("debit", "deposit", "tax")
    }
    source_presence: dict[str, np.ndarray] = {}
    source_structures: dict[str, Any] = {}
    projections: dict[str, list[str]] = {"base": ["case_id", "date_decision", "WEEK_NUM", "MONTH"]}

    def add_registry(name: str, family: str, role: str, source_fields: list[str], dtype: str,
                     construction: str, selection: str, missing: str, zero: str,
                     finite_rule: str, description: str, limitations: str,
                     include_t: bool, include_plus: bool) -> None:
        registry.append({
            "feature_name": name, "feature_family": family, "feature_role": role,
            "source_fields": json.dumps(source_fields, separators=(",", ":")), "dtype": dtype,
            "construction_or_formula": construction, "row_selection_rule": selection,
            "missing_rule": missing, "zero_rule": zero, "finite_eligibility_rule": finite_rule,
            "economic_description": description, "interpretation_limitations": limitations,
            "include_in_T": include_t, "include_in_T_plus_AD": include_plus,
        })

    for name, dtype, description in (
        ("case_id", "int64", "Application record join key"),
        ("date_decision", "datetime64[ns]", "Decision-date metadata"),
        ("WEEK_NUM", str(base["WEEK_NUM"].dtype), "Competition week metadata"),
        ("MONTH", str(base["MONTH"].dtype), "Competition month metadata"),
    ):
        add_registry(name, "METADATA", "METADATA", [f"base.{name}"], dtype, "Direct metadata copy",
                     "All verified base rows in original order", "Preserve source missingness", "Not applicable",
                     "Not a model feature", description, "Excluded from model whitelists.", False, False)

    def load_direct(source: str, fields: list[str]) -> pd.DataFrame:
        columns = ["case_id", *fields]
        projections[source] = columns
        frames = [pq.read_table(args.train_dir / name, columns=columns).to_pandas() for name in group_files[source]]
        frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        nulls = int(frame["case_id"].isna().sum())
        duplicates = int(frame["case_id"].duplicated().sum())
        orphans = int((~frame["case_id"].isin(base_index) & frame["case_id"].notna()).sum())
        source_structures[source] = {"physical_rows": len(frame), "null_case_id_rows": nulls,
                                     "duplicate_case_id_rows": duplicates, "orphan_rows": orphans,
                                     "matched_applications": int(frame["case_id"].isin(base_index).sum()),
                                     "projected_columns": columns}
        check(f"{source}_direct_unique", nulls == 0 and duplicates == 0 and orphans == 0,
              source_structures[source], "no null, duplicate, or orphan keys", "Direct source must be at most one row per case_id.")
        if nulls or duplicates or orphans:
            raise RuntimeError(f"Direct source key failure: {source}")
        source_presence[source] = base_index.isin(frame["case_id"]).to_numpy(bool) if hasattr(base_index.isin(frame["case_id"]), "to_numpy") else base_index.isin(frame["case_id"])
        return frame.set_index("case_id").reindex(base_index)

    static0_fields = [raw for _, source, raw, _, _, _ in T_CANDIDATES if source == "static_0"]
    staticcb_t_fields = [raw for _, source, raw, _, _, _ in T_CANDIDATES if source == "static_cb_0"]
    staticcb_ad_fields = [raw for _, _, source, raw, use in AD_CANDIDATES if source == "static_cb_0" and use == "direct"]
    other_fields = [raw for _, _, source, raw, use in AD_CANDIDATES if source == "other_1" and use == "direct"]
    static0 = load_direct("static_0", static0_fields)
    staticcb = load_direct("static_cb_0", list(dict.fromkeys([*staticcb_t_fields, *staticcb_ad_fields])))
    other = load_direct("other_1", other_fields)

    person_fields = [raw for _, source, raw, _, _, _ in T_CANDIDATES if source == "person_1"]
    projections["person_1"] = ["case_id", "num_group1", "personindex_1023L", *person_fields]
    person, person_details = select_applicant_rows(
        [args.train_dir / name for name in group_files["person_1"]], base_index, person_fields
    )
    source_structures["person_1"] = {key: value for key, value in person_details.items() if key != "selected_count"}
    person_ok = person_details["selected_zero"] == 0 and person_details["selected_multiple"] == 0 and person_details["selected_one"] == population
    check("person_applicant_exactly_one", person_ok, source_structures["person_1"],
          {"selected_one": population, "selected_zero": 0, "selected_multiple": 0},
          "Applicant selection uses person_1 num_group1=0; personindex is diagnostic only.")
    if not person_ok:
        raise RuntimeError("Applicant selection is not exactly one row per base application")

    t_feature_names = []
    raw_decisions: list[dict[str, Any]] = []
    source_frames = {"static_0": static0, "static_cb_0": staticcb, "person_1": person}
    for candidate_id, source, raw, name, raw_type, intended in T_CANDIDATES:
        series = source_frames[source][raw]
        if raw_type == "Categorical":
            features[name] = series.astype(object).to_numpy()
            dtype = "category_string"
            invalid_counts[name] = 0
        else:
            cleaned, counts, _ = numeric_clean(series.reset_index(drop=True))
            features[name] = cleaned
            dtype = "float64"
            invalid_counts[name] = counts["unparseable"] + counts["nonfinite"]
        t_feature_names.append(name)
        add_registry(name, "T", "DIRECT_VALUE", [f"{source}.{raw}"], dtype, "Direct approved candidate",
                     "Applicant num_group1=0" if source == "person_1" else "At-most-one source row per case_id",
                     "Preserve missing; invalid numeric source observations become missing with audit count.",
                     "Preserve observed zero.", "Finite source numeric value" if raw_type != "Categorical" else "Nominal code or true missing",
                     intended, "Encoded categories remain nominal; exact source scope and timing limitations are retained.", True, True)
        raw_decisions.append({
            "raw_candidate_id": candidate_id, "feature_family": "T", "source": source, "raw_field": raw,
            "dictionary_description": descriptions.get(raw, ""), "raw_data_type": raw_type,
            "default_use": "Direct application value", "decision_status": "USED_DIRECT",
            "generated_feature_names": json.dumps([name], separators=(",", ":")),
            "zero_missing_rule": "Observed zero preserved; missing remains missing.",
            "timing_or_scope_limitations": intended, "redundancy_note": "Approved raw candidate retained without outcome-driven selection.",
            "decision_reason": "Explicit Task 08 T whitelist.",
        })

    ad_substantive: dict[str, list[str]] = {family: [] for family in ("debit", "deposit", "tax")}
    ad_history_results: dict[tuple[str, str], dict[str, Any]] = {}
    history_sources = {
        source: [raw for _, _, candidate_source, raw, use in AD_CANDIDATES if candidate_source == source and use == "aggregation"]
        for source in ("debitcard_1", "deposit_1", "tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1")
    }
    for source, fields in history_sources.items():
        projections[source] = ["case_id", "num_group1", *fields]
        reducer, structure = aggregate_history([args.train_dir / name for name in group_files[source]], fields, base_index)
        source_structures[source] = structure
        source_presence[source] = reducer.source_count > 0
        structural_ok = not any(structure[key] for key in (
            "null_case_id_rows", "orphan_rows", "null_num_group1_rows", "invalid_num_group1_rows",
            "duplicate_composite_key_distinct_count", "duplicate_composite_key_excess_rows",
        ))
        check(f"{source}_history_structure", structural_ok, structure,
              "no null/orphan/index/composite-key defects", "Historical values are aggregated independently by application.")
        if not structural_ok:
            raise RuntimeError(f"History structural failure: {source}")
        for field in fields:
            ad_history_results[(source, field)] = reducer.result(field)

    direct_ad_frames = {"static_cb_0": staticcb, "other_1": other}
    for candidate_id, family, source, raw, use in AD_CANDIDATES:
        generated = []
        if use == "aggregation":
            result = ad_history_results[(source, raw)]
            for stat in ("mean", "max"):
                name = f"ad__{family}__{raw}__finite_{stat}"
                features[name] = result[stat]
                invalid_counts[name] = int(result["invalid_count"].sum())
                ad_substantive[family].append(name)
                generated.append(name)
                add_registry(name, family, "AGGREGATED_VALUE", [f"{source}.{raw}"], "float64",
                             f"Per-application {stat} of finite observations", "All distinct source rows; num_group1 is local index only",
                             "No source or no finite observations remains missing; partial groups summarize finite observations.",
                             "Finite zeros are retained.", "Native finite numeric observations only",
                             f"Observed-record {stat} for {descriptions.get(raw, raw)}",
                             "Not an account total, current balance, wage, or verified real-world availability measure.", False, True)
            raw_masks[family]["content"] |= result["finite_count"] > 0
            raw_masks[family]["nonzero"] |= result["nonzero_count"] > 0
            status = "USED_AGGREGATION"
            default = "Per-application finite mean and maximum"
        else:
            name = f"ad__{family}__{raw}"
            cleaned, counts, classified = numeric_clean(direct_ad_frames[source][raw].reset_index(drop=True))
            features[name] = cleaned
            invalid_counts[name] = counts["unparseable"] + counts["nonfinite"]
            ad_substantive[family].append(name)
            generated.append(name)
            raw_masks[family]["content"] |= np.isfinite(cleaned)
            raw_masks[family]["nonzero"] |= np.isfinite(cleaned) & (cleaned != 0)
            add_registry(name, family, "DIRECT_VALUE", [f"{source}.{raw}"], "float64", "Direct approved application value",
                         "At-most-one source row per case_id", "No source or invalid/nonfinite remains missing.",
                         "Observed zero is retained.", "Native finite numeric observation only",
                         descriptions.get(raw, raw), "Recorded value with source window/unit limitations; not certified wealth or activity.", False, True)
            status = "USED_DIRECT"
            default = "Direct application value"
        raw_decisions.append({
            "raw_candidate_id": candidate_id, "feature_family": family, "source": source, "raw_field": raw,
            "dictionary_description": descriptions.get(raw, ""), "raw_data_type": "Numeric",
            "default_use": default, "decision_status": status,
            "generated_feature_names": json.dumps(generated, separators=(",", ":")),
            "zero_missing_rule": "Observed finite zero retained; no source/no finite observation remains missing.",
            "timing_or_scope_limitations": "Competition decision-time construct accepted; exact real-world units/window remain limited.",
            "redundancy_note": "C record count and C amount sum are not generated; C mean/max retained.",
            "decision_reason": "Explicit Task 08 AD whitelist and construction rule.",
        })

    b1_expected = {
        "debit": {"content": 60_697, "nonzero": 34_062},
        "deposit": {"content": 134_960, "nonzero": 88_811},
        "tax": {"content": 1_402_486, "nonzero": 1_305_517},
    }
    evidence_masks = {
        "debit": source_presence["debitcard_1"] | source_presence["other_1"],
        "deposit": source_presence["deposit_1"] | source_presence["other_1"],
        "tax": flags["module__tax__field_evidence"].to_numpy(bool)
        | source_presence["tax_registry_a_1"]
        | source_presence["tax_registry_b_1"]
        | source_presence["tax_registry_c_1"],
    }
    indicator_names: list[str] = []
    family_state_masks: dict[str, dict[str, np.ndarray]] = {}
    optional_indicator_omissions = []
    for family in ("debit", "deposit", "tax"):
        final_content = features[ad_substantive[family]].notna().any(axis=1).to_numpy(bool)
        content = raw_masks[family]["content"]
        nonzero = raw_masks[family]["nonzero"]
        check(f"{family}_final_content_matches_raw", np.array_equal(final_content, content), int(final_content.sum()), int(content.sum()),
              "Retained substantive output finite coverage equals approved raw finite coverage.")
        check(f"{family}_batch1_content", int(content.sum()) == b1_expected[family]["content"], int(content.sum()), b1_expected[family]["content"],
              "Reconciliation to accepted Batch 1 raw economic finite coverage.")
        check(f"{family}_batch1_nonzero", int(nonzero.sum()) == b1_expected[family]["nonzero"], int(nonzero.sum()), b1_expected[family]["nonzero"],
              "Reconciliation to accepted Batch 1 raw economic nonzero coverage.")
        family_state_masks[family] = family_states(evidence_masks[family], content, nonzero)
        check(f"{family}_state_partition", sum(int(mask.sum()) for mask in family_state_masks[family].values()) == population,
              {name: int(mask.sum()) for name, mask in family_state_masks[family].items()}, population,
              "Four mutually exclusive AD review states partition the base population.")
        for suffix, mask, description in (
            ("has_numeric_content", content, "Any finite retained substantive family value"),
            ("has_nonzero_content", nonzero, "Any finite nonzero approved raw family observation"),
        ):
            name = f"ad__{family}__{suffix}"
            features[name] = mask.astype(np.int8)
            indicator_names.append(name)
            invalid_counts[name] = 0
            add_registry(name, family, "STATE_INDICATOR", [f"approved_{family}_raw_fields"], "int8", description,
                         "Full base population", "Deterministic zero means criterion not observed; not real-world absence.",
                         "Zero is a defined audit state.", "Deterministic boolean from approved raw observations",
                         description, "Does not prove absence of accounts, activity, or tax records.", False, True)
        evidence_suffix = "has_source_or_field_evidence" if family == "tax" else "has_source_evidence"
        if np.array_equal(evidence_masks[family], content):
            optional_indicator_omissions.append({"family": family, "indicator": f"ad__{family}__{evidence_suffix}",
                                                 "reason": "Exact duplicate of has_numeric_content; omitted."})
        else:
            name = f"ad__{family}__{evidence_suffix}"
            features[name] = evidence_masks[family].astype(np.int8)
            indicator_names.append(name)
            invalid_counts[name] = 0
            evidence_sources = (
                ["Batch1 tax field-evidence flag", "tax_registry_a_1/tax_registry_b_1/tax_registry_c_1 source presence"]
                if family == "tax" else [f"{family} source rows"]
            )
            add_registry(name, family, "STATE_INDICATOR", evidence_sources,
                         "int8", "Any scoped source row or tax-specific field evidence",
                         "Full base population", "Zero means no scoped source/evidence observed, not real-world absence.",
                         "Zero is a defined audit state.", "Deterministic source/evidence boolean",
                         "Scoped source or field evidence", "Tax static_cb row alone is not tax evidence.", False, True)

    ratio_reviews = [
        ratio_review(features["t__requested_credit_amount"].to_numpy(np.float64), features["t__applicant_main_income"].to_numpy(np.float64),
                     "recorded_credit_amount_to_applicant_income", "credamount_770A / mainoccupationinc_384A"),
        ratio_review(features["t__current_debt"].to_numpy(np.float64), features["t__applicant_main_income"].to_numpy(np.float64),
                     "recorded_current_debt_to_applicant_income", "currdebt_22A / mainoccupationinc_384A"),
        ratio_review(features["t__current_application_annuity"].to_numpy(np.float64), features["t__applicant_main_income"].to_numpy(np.float64),
                     "recorded_annuity_to_applicant_income", "annuity_780A / mainoccupationinc_384A"),
    ]

    substantive_names = [name for family in ("debit", "deposit", "tax") for name in ad_substantive[family]]
    t_plus_ad = [*t_feature_names, *substantive_names, *indicator_names]
    feature_sets = {
        "construction_version": VERSION,
        "registry_reference": "feature_registry.csv",
        "metadata": ["case_id", "date_decision", "WEEK_NUM", "MONTH"],
        "T": t_feature_names,
        "AD_substantive_values": substantive_names,
        "AD_substantive_by_family": ad_substantive,
        "AD_indicators": indicator_names,
        "T_plus_AD": t_plus_ad,
    }
    check("raw_candidate_count", len(raw_decisions) == 41, len(raw_decisions), 41, "Exactly 21 T and 20 AD raw candidates.")
    raw_family_counts = Counter(row["feature_family"] for row in raw_decisions)
    check("raw_candidate_family_counts", raw_family_counts == Counter({"T": 21, "debit": 5, "deposit": 4, "tax": 11}),
          dict(raw_family_counts), {"T": 21, "debit": 5, "deposit": 4, "tax": 11}, "Approved raw whitelist counts.")
    check("feature_set_order", t_plus_ad == [*t_feature_names, *substantive_names, *indicator_names] and len(t_plus_ad) == len(set(t_plus_ad)),
          len(t_plus_ad), len(set(t_plus_ad)), "T+AD is ordered T then substantive AD then AD indicators without duplicates.")
    check("no_ad_in_T", all(name.startswith("t__") for name in t_feature_names), t_feature_names, "all t__", "T whitelist contains no AD provenance.")
    check("no_target_projection", all("target" not in columns for columns in projections.values()), projections, "no target", "Projected source columns exclude target.")

    duplicate_groups = exact_duplicate_groups(features, t_plus_ad)
    check("materialized_exact_duplicate_diagnostic", True, duplicate_groups, "diagnostic completed", "Target-free exact value-and-missing-mask diagnostic; no columns removed.")

    feature_audit = feature_audit_rows(features, registry, invalid_counts)
    nonfinite_materialized = sum(int(row["nonfinite_materialized_count"] or 0) for row in feature_audit)
    check("no_nonfinite_materialized", nonfinite_materialized == 0, nonfinite_materialized, 0, "No infinity is delivered as a model value.")

    coverage_rows: list[dict[str, Any]] = []
    for family in ("debit", "deposit", "tax"):
        masks = {
            "RAW_APPROVED_FINITE_CONTENT": raw_masks[family]["content"],
            "RAW_APPROVED_NONZERO_CONTENT": raw_masks[family]["nonzero"],
            "FINAL_RETAINED_SUBSTANTIVE_FINITE_CONTENT": features[ad_substantive[family]].notna().any(axis=1).to_numpy(bool),
            "SOURCE_OR_FIELD_EVIDENCE": evidence_masks[family],
            **family_state_masks[family],
        }
        evidence_n = int(evidence_masks[family].sum())
        for metric, mask in masks.items():
            count = int(mask.sum())
            coverage_rows.append({"family": family, "metric": metric, "application_count": count,
                                  "N": population, "population_rate": count / population,
                                  "conditional_source_rate": count / evidence_n if evidence_n else None,
                                  "conditional_source_denominator": evidence_n,
                                  "definition": "Observed scoped source/economic evidence; zero coverage is not proof of real-world absence.",
                                  "batch1_expected": b1_expected[family].get("content" if "FINITE_CONTENT" in metric else "nonzero")
                                  if metric in {"RAW_APPROVED_FINITE_CONTENT", "RAW_APPROVED_NONZERO_CONTENT"} else None,
                                  "reconciliation_status": "MATCH" if metric == "RAW_APPROVED_FINITE_CONTENT" and count == b1_expected[family]["content"]
                                  or metric == "RAW_APPROVED_NONZERO_CONTENT" and count == b1_expected[family]["nonzero"] else "NOT_APPLICABLE"})
    union_content = np.logical_or.reduce([raw_masks[f]["content"] for f in ("debit", "deposit", "tax")])
    union_nonzero = np.logical_or.reduce([raw_masks[f]["nonzero"] for f in ("debit", "deposit", "tax")])
    union_evidence = np.logical_or.reduce([evidence_masks[f] for f in ("debit", "deposit", "tax")])
    union_states = family_states(union_evidence, union_content, union_nonzero)
    for metric, mask in {"RAW_APPROVED_FINITE_CONTENT": union_content, "RAW_APPROVED_NONZERO_CONTENT": union_nonzero,
                         "SOURCE_OR_FIELD_EVIDENCE": union_evidence, **union_states}.items():
        coverage_rows.append({"family": "any_AD_union", "metric": metric, "application_count": int(mask.sum()),
                              "N": population, "population_rate": float(mask.mean()),
                              "conditional_source_rate": float(mask.sum() / union_evidence.sum()) if union_evidence.any() else None,
                              "conditional_source_denominator": int(union_evidence.sum()),
                              "definition": "Union across debit, deposit, and tax; indicators and metadata excluded.",
                              "batch1_expected": 1_411_439 if metric == "RAW_APPROVED_FINITE_CONTENT" else None,
                              "reconciliation_status": "MATCH" if metric == "RAW_APPROVED_FINITE_CONTENT" and int(mask.sum()) == 1_411_439 else "NOT_APPLICABLE"})
    check("any_ad_batch1_content", int(union_content.sum()) == 1_411_439, int(union_content.sum()), 1_411_439, "Any-family finite content reconciliation.")
    check("no_ad_batch1_content", int((~union_content).sum()) == 115_220, int((~union_content).sum()), 115_220, "No-family finite content reconciliation.")

    overlap_rows = []
    presence = {family: features[ad_substantive[family]].notna().any(axis=1).to_numpy(bool) for family in ("debit", "deposit", "tax")}
    for bits in ("000", "001", "010", "011", "100", "101", "110", "111"):
        mask = np.ones(population, dtype=bool)
        for bit, family in zip(bits, ("debit", "deposit", "tax")):
            mask &= presence[family] if bit == "1" else ~presence[family]
        overlap_rows.append({"combination": bits, "debit": bits[0] == "1", "deposit": bits[1] == "1", "tax": bits[2] == "1",
                             "application_count": int(mask.sum()), "N": population, "population_rate": float(mask.mean()),
                             "coverage_definition": "Any finite retained substantive output within each family."})
    check("ad_overlap_partition", sum(row["application_count"] for row in overlap_rows) == population,
          sum(row["application_count"] for row in overlap_rows), population, "Eight finite-output presence combinations partition N.")
    check("all_three_batch1_content", next(row["application_count"] for row in overlap_rows if row["combination"] == "111") == 51_616,
          next(row["application_count"] for row in overlap_rows if row["combination"] == "111"), 51_616, "All-three coverage reconciliation.")

    # Deterministic review examples selected without target.
    reason_masks = [
        ("no_AD_finite_content", ~union_content),
        ("tax_only_finite_content", presence["tax"] & ~presence["debit"] & ~presence["deposit"]),
        ("debit_finite_content", presence["debit"]),
        ("deposit_finite_content", presence["deposit"]),
        ("all_three_finite_content", presence["debit"] & presence["deposit"] & presence["tax"]),
        ("debit_zero_only", family_state_masks["debit"]["FINITE_CONTENT_ZERO_ONLY"]),
        ("deposit_zero_only", family_state_masks["deposit"]["FINITE_CONTENT_ZERO_ONLY"]),
        ("tax_zero_only", family_state_masks["tax"]["FINITE_CONTENT_ZERO_ONLY"]),
        ("debit_nonzero", family_state_masks["debit"]["FINITE_CONTENT_WITH_NONZERO"]),
        ("deposit_nonzero", family_state_masks["deposit"]["FINITE_CONTENT_WITH_NONZERO"]),
        ("tax_nonzero", family_state_masks["tax"]["FINITE_CONTENT_WITH_NONZERO"]),
        ("missing_recorded_primary_income", features["t__recorded_primary_income"].isna().to_numpy()),
        ("zero_applicant_income_ratio_undefined", features["t__applicant_main_income"].to_numpy(np.float64) == 0),
    ]
    selected_positions: list[int] = []
    selected_reasons: dict[int, list[str]] = defaultdict(list)
    for reason, mask in reason_masks:
        available = np.flatnonzero(mask)
        if available.size:
            position = int(available[0])
            selected_reasons[position].append(reason)
            if position not in selected_positions:
                selected_positions.append(position)
    for source in history_sources:
        structure_mask = source_presence[source]
        available = np.flatnonzero(structure_mask)
        if available.size:
            position = int(available[0])
            selected_reasons[position].append(f"aggregation_check_{source}")
            if position not in selected_positions:
                selected_positions.append(position)
    if len(selected_positions) < 12:
        for position in range(population):
            if position not in selected_positions:
                selected_positions.append(position)
                selected_reasons[position].append("deterministic_base_order_fill")
            if len(selected_positions) >= 12:
                break
    selected_positions = selected_positions[:16]
    example_columns = ["case_id", *t_feature_names, *substantive_names, *indicator_names]
    example_frame = features.iloc[selected_positions][example_columns].copy()
    example_frame.insert(1, "example_reason", [";".join(selected_reasons[pos]) for pos in selected_positions])
    example_rows = example_frame.to_dict(orient="records")

    # Independent raw-row recalculation for one low-multiplicity application per history source.
    aggregation_rows: list[dict[str, Any]] = []
    for source, source_fields in history_sources.items():
        candidates = np.flatnonzero(source_presence[source])
        if not candidates.size:
            continue
        selected_pos = next((pos for pos in selected_positions if source_presence[source][pos]), int(candidates[0]))
        case_id = base_index[selected_pos]
        columns = ["case_id", "num_group1", *source_fields]
        raw_frames = []
        for file_name in group_files[source]:
            table = pq.read_table(args.train_dir / file_name, columns=columns, filters=[("case_id", "=", int(case_id))])
            raw_frames.append(table.to_pandas())
        raw = pd.concat(raw_frames, ignore_index=True)
        for field in source_fields:
            classified = classify_numeric(raw[field])
            finite_values = classified["values"][classified["finite"]]
            expected_mean = float(finite_values.mean()) if finite_values.size else None
            expected_max = float(finite_values.max()) if finite_values.size else None
            built = ad_history_results[(source, field)]
            built_mean = built["mean"][selected_pos]
            built_max = built["max"][selected_pos]
            mean_match = np.isnan(built_mean) if expected_mean is None else np.isclose(expected_mean, built_mean, rtol=0, atol=1e-12)
            max_match = np.isnan(built_max) if expected_max is None else np.isclose(expected_max, built_max, rtol=0, atol=1e-12)
            comparison = bool(mean_match and max_match)
            for row_index, row in raw.iterrows():
                state = next(name for name in ("missing", "unparseable", "nonfinite", "zero", "positive", "negative") if classified[name][row_index])
                aggregation_rows.append({
                    "case_id": case_id, "source": source, "raw_field": field,
                    "record_index": row["num_group1"], "raw_value": row[field], "classified_state": state,
                    "finite_observation_count": int(finite_values.size), "expected_mean": expected_mean,
                    "expected_max": expected_max, "built_mean": built_mean, "built_max": built_max,
                    "comparison_status": "PASS" if comparison else "FAIL",
                })
    check("aggregation_examples_independent", all(row["comparison_status"] == "PASS" for row in aggregation_rows),
          Counter(row["comparison_status"] for row in aggregation_rows), {"PASS": len(aggregation_rows)},
          "Selected raw rows independently recompute stored mean/max features.")

    # Registry and output integrity before writing.
    registered_columns = [row["feature_name"] for row in registry]
    check("registry_matches_feature_columns", registered_columns == list(features.columns), registered_columns, list(features.columns),
          "Every output column is registered in output order.")
    check("output_row_order", np.array_equal(features["case_id"].to_numpy(), base["case_id"].to_numpy()), True, True,
          "Feature table preserves exact base row order.")
    check("output_no_target", "target" not in features.columns, list(features.columns), "target absent", "No outcome column is materialized.")

    b2_rows = read_csv(b2_csv_path)
    original_correction_rows, corrected_b2_rows, correction_status = correction_rows(b2_rows)

    args.audit_output_dir.mkdir(parents=True, exist_ok=False)
    args.feature_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_paths["raw_candidate_decisions.csv"], raw_decisions)
    write_csv(output_paths["feature_registry.csv"], registry)
    output_paths["feature_sets.json"].write_text(json.dumps(json_safe(feature_sets), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output_paths["feature_value_audit.csv"], feature_audit)
    write_csv(output_paths["ad_coverage.csv"], coverage_rows)
    write_csv(output_paths["ad_overlap.csv"], overlap_rows)
    write_csv(output_paths["feature_examples.csv"], example_rows)
    write_csv(output_paths["aggregation_examples.csv"], aggregation_rows)
    write_csv(output_paths["batch2_correction_original_rows.csv"], original_correction_rows)
    features.to_parquet(args.feature_output, engine="pyarrow", compression="zstd", index=False)

    # Guarded correction of exactly six Batch 2 CSV entries plus its manifest size.
    if correction_status == "CORRECTED_SIX_ROWS":
        write_csv(b2_csv_path, corrected_b2_rows, headers=list(b2_rows[0]))
        b2_summary = json.loads(b2_summary_path.read_text(encoding="utf-8"))
        manifest_entry = next(item for item in b2_summary["output_manifest"] if item["name"] == "ad_followup_results.csv")
        manifest_entry["size_bytes"] = b2_csv_path.stat().st_size
        b2_summary_path.write_text(json.dumps(json_safe(b2_summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    corrected_now = read_csv(b2_csv_path)
    corrected_selected = [row for row in corrected_now if row["section"] == "TAX_DETAIL_TIMING" and row["metric"] in CORRECTION_METRICS]
    correction_ok = len(corrected_selected) == 6 and all(row["status"] == "NOT_CHECKED" and row["value"] == "" for row in corrected_selected)
    check("batch2_disabled_latest_correction", correction_ok, corrected_selected, "six blank NOT_CHECKED rows",
          "Only the known disabled-tracking rows are corrected; original entries are preserved under Task 08.")

    expected_changed = {str(b2_csv_path.resolve()), str(b2_summary_path.resolve())} if correction_status == "CORRECTED_SIX_ROWS" else set()
    after_state = file_state([*raw_paths, *prior_paths])
    state_comparison = compare_states(before_state, after_state, expected_changed)
    check("scoped_input_state", state_comparison["all_changes_as_expected"],
          {"expected_changed": sorted(expected_changed)}, "only authorized Batch 2 correction files change",
          "Size/mtime comparison is lightweight and not cryptographic proof.")

    reopened = pq.read_table(args.feature_output)
    reopen_ok = reopened.num_rows == population and reopened.column_names == list(features.columns)
    check("feature_parquet_reopen", reopen_ok, {"rows": reopened.num_rows, "columns": reopened.num_columns},
          {"rows": population, "columns": len(features.columns)}, "Saved feature Parquet reopens with expected schema.")
    del reopened
    check("feature_output_unique_keys", features["case_id"].is_unique and not features["case_id"].isna().any(),
          {"duplicates": int(features["case_id"].duplicated().sum()), "nulls": int(features["case_id"].isna().sum())},
          {"duplicates": 0, "nulls": 0}, "Output application keys remain unique and non-null.")

    write_csv(output_paths["validation_results.csv"], validation)
    validation_counts = dict(Counter(row["status"] for row in validation))
    failures = [row for row in validation if row["status"] == "FAIL"]
    status = "COMPLETED" if not failures else "COMPLETED_WITH_CHECK_FAILURES"

    report = f"""# Task 08 — Application-level T and AD feature table

## Build result

- Status: **{status}**
- Applications: **{population:,}**, preserving exact base order and Batch 1 key agreement.
- T features: **{len(t_feature_names)}** direct approved values.
- AD substantive values: **{len(substantive_names)}** ({len(ad_substantive['debit'])} debit, {len(ad_substantive['deposit'])} deposit, {len(ad_substantive['tax'])} tax).
- AD indicators: **{len(indicator_names)}**. The deposit source-evidence indicator was omitted because it exactly duplicated deposit numeric-content presence.
- Optional ratios built: **0**. All three were deferred for undocumented income period/cross-field transformation comparability.

The 41 approved raw candidates generate more columns because seven history amounts produce finite mean and maximum. They generate fewer than a naive expansion because no latest/minimum/median/standard-deviation/sum features or broad missing indicators were added.

## Coverage

| Family | Finite content | Nonzero content | Source/field evidence |
|---|---:|---:|---:|
| Debit | {int(raw_masks['debit']['content'].sum()):,} | {int(raw_masks['debit']['nonzero'].sum()):,} | {int(evidence_masks['debit'].sum()):,} |
| Deposit | {int(raw_masks['deposit']['content'].sum()):,} | {int(raw_masks['deposit']['nonzero'].sum()):,} | {int(evidence_masks['deposit'].sum()):,} |
| Tax | {int(raw_masks['tax']['content'].sum()):,} | {int(raw_masks['tax']['nonzero'].sum()):,} | {int(evidence_masks['tax'].sum()):,} |
| Any AD | {int(union_content.sum()):,} | {int(union_nonzero.sum()):,} | {int(union_evidence.sum()):,} |

No-source and source-without-finite-content remain distinct from observed zero. Missing monetary histories stay missing.

## Fixed decisions

- Static tax count and sum are retained. No C record-count or C sum predictor was added; C finite mean and maximum remain descriptive candidates.
- Debit, deposit, and tax histories were aggregated independently before joining to base. `num_group1` was never used across sources or as chronology.
- Three ratio candidates remain `DEFERRED_UNIT_UNCERTAINTY`; no annual/monthly conversion, epsilon, clipping, or zero replacement was applied.
- Metadata, diagnostic counts, example rows, and review states are absent from model whitelists.

## Batch 2 correction

Correction status: **{correction_status}**. Six tax latest-tie rows produced while latest tracking was disabled now have blank values and `NOT_CHECKED`. Their original entries are preserved in `batch2_correction_original_rows.csv`. Batch 2 summaries/reports contained no tie values; only the CSV and its manifest byte size were affected.

## Verification

Validation: **{validation_counts.get('PASS', 0)} PASS**, **{validation_counts.get('FAIL', 0)} FAIL**, **{validation_counts.get('NOT_CHECKED', 0)} NOT_CHECKED**. Full details are in `validation_results.csv` and `build_summary.json`.

No target, TEST data, depth-2 values, deferred Bureau/history T values, partition, fitted preprocessing, or model was used.
"""
    output_paths["build_report.md"].write_text(report, encoding="utf-8")

    produced = [*output_paths.values(), args.feature_output]
    manifest = [{"name": path.name, "path": str(path.resolve()), "status": "PRODUCED",
                 "size_bytes": None if path == output_paths["build_summary.json"] else path.stat().st_size}
                for path in produced]
    summary = {
        "task": "TASK_08", "construction_version": VERSION, "execution_status": status,
        "utc_execution_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime_versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__},
        "exact_execution_command": shlex.join([sys.executable, *sys.argv]),
        "paths": {"train_dir": str(args.train_dir.resolve()), "dictionary": str(args.dictionary.resolve()),
                  "task06_dir": str(args.task06_dir.resolve()), "batch1_dir": str(args.batch1_dir.resolve()),
                  "batch2_dir": str(args.batch2_dir.resolve()), "audit_output_dir": str(args.audit_output_dir.resolve()),
                  "feature_output": str(args.feature_output.resolve())},
        "verified_population": {"N": population, "base_key_nonnull": True, "base_key_unique": True,
                                "batch1_exact_key_order": order_match, "output_exact_key_order": True},
        "source_files": group_files, "source_projections": projections, "source_structures": source_structures,
        "raw_candidate_counts": dict(raw_family_counts),
        "feature_counts": {"metadata": 4, "T": len(t_feature_names), "AD_substantive": len(substantive_names),
                           "AD_substantive_by_family": {key: len(value) for key, value in ad_substantive.items()},
                           "AD_indicators": len(indicator_names), "T_plus_AD": len(t_plus_ad), "table_columns": len(features.columns)},
        "ratio_reviews": ratio_reviews, "optional_indicator_omissions": optional_indicator_omissions,
        "exact_duplicate_groups": duplicate_groups,
        "module_coverage": {family: {"finite_content": int(raw_masks[family]["content"].sum()),
                                      "nonzero_content": int(raw_masks[family]["nonzero"].sum()),
                                      "source_or_field_evidence": int(evidence_masks[family].sum()),
                                      "states": {name: int(mask.sum()) for name, mask in family_state_masks[family].items()}}
                            for family in ("debit", "deposit", "tax")},
        "union_coverage": {"finite_content": int(union_content.sum()), "nonzero_content": int(union_nonzero.sum()),
                           "source_or_field_evidence": int(union_evidence.sum()), "no_finite_content": int((~union_content).sum())},
        "batch2_correction": {"status": correction_status, "affected_csv_rows": 6,
                              "original_rows_file": str(output_paths["batch2_correction_original_rows.csv"].resolve()),
                              "corrected_file": str(b2_csv_path.resolve()), "summary_manifest_size_updated": correction_status == "CORRECTED_SIX_ROWS",
                              "summary_or_report_metric_values_affected": False},
        "validation_counts": validation_counts, "validation_results": validation,
        "input_state_comparison": state_comparison,
        "output_manifest": manifest,
        "deferred": ["Three optional T ratios due unit/period uncertainty", "Partitions", "Training-only preprocessing", "Models"],
        "caveats": ["Application rows are not verified distinct natural persons or originated loans.",
                    "Competition decision-time availability does not prove real-world deployment availability.",
                    "Observed deposit and turnover values are not verified current wealth or account balances.",
                    "Module indicator zero does not prove absence of real-world activity."],
    }
    output_paths["build_summary.json"].write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"execution_status": status, "N": population, "T": len(t_feature_names),
                      "AD_substantive": len(substantive_names), "AD_indicators": len(indicator_names),
                      "validation_counts": validation_counts, "feature_output": str(args.feature_output)}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
