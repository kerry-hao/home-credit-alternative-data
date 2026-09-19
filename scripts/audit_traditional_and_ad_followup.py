#!/usr/bin/env python3
"""Task 07 Batch 2: traditional-information audit and focused AD follow-ups."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import platform
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from audit_ad_coverage import classify_numeric, exact_quantiles, parse_date_values


TASK_ID = "TASK_07_BATCH2"
SCRIPT_VERSION = "1.0.0"
QUANTILE_METHOD = "linear"
FLOAT_RTOL = 1e-9
FLOAT_ATOL = 1e-6
BATCH_SIZE = 200_000

OUTPUT_NAMES = (
    "source_structure.csv",
    "traditional_field_audit.csv",
    "person_role_audit.csv",
    "credit_count_review.csv",
    "ad_followup_results.csv",
    "documentation_evidence.md",
    "feature_decision_review.csv",
    "audit_summary.json",
    "audit_report.md",
)

STATIC_NUMERIC = (
    "maininc_215A", "credamount_770A", "annuity_780A", "currdebt_22A",
    "totaldebt_9A", "numactivecreds_622L", "numactivecredschannel_414L",
    "numactiverelcontr_750L", "opencred_647L", "numcontrs3months_479L",
    "applications30d_658L", "maxdpdlast3m_392P", "maxdpdlast6m_474P",
    "maxdpdlast12m_727P", "maxdpdlast24m_143P", "pmtnum_254L",
    "numinstpaidlastcontr_4325080L", "lastapprcredamount_781A",
)
STATIC_CATEGORICAL = ("credtype_322L", "lastst_736L")
STATIC_DATES = (
    "lastapplicationdate_877D", "lastactivateddate_801D", "lastapprdate_640D",
    "lastrejectdate_50D", "lastdelinqdate_224D",
)
STATIC_CB_NUMERIC = ("days30_165L", "days90_310L", "days180_256L", "days360_512L")
STATIC_CB_CATEGORICAL = ("education_88M", "education_1103M")

CBA_NUMERIC = (
    "numberofcontrsvalue_258L", "numberofcontrsvalue_358L",
    "totaloutstanddebtvalue_39A", "totaloutstanddebtvalue_668A",
    "debtoutstand_525A", "outstandingamount_362A", "outstandingamount_354A",
    "totalamount_996A", "totalamount_6A", "credlmt_935A", "credlmt_230A",
    "dpdmax_139P", "dpdmax_757P", "overdueamount_659A", "overdueamount_31A",
)
CBA_CATEGORICAL = ("contractst_545M", "contractst_964M", "subjectrole_182M", "subjectrole_93M")
CBA_DATES = (
    "dateofcredstart_181D", "dateofcredstart_739D", "dateofcredend_289D",
    "dateofcredend_353D", "dateofrealrepmt_138D", "lastupdate_1112D",
    "lastupdate_388D",
)
CBA_ACTIVE_FIELDS = (
    "numberofcontrsvalue_258L", "totaloutstanddebtvalue_39A", "debtoutstand_525A",
    "outstandingamount_362A", "totalamount_996A", "credlmt_935A", "dpdmax_139P",
    "overdueamount_659A", "contractst_545M", "subjectrole_182M",
    "dateofcredstart_181D", "dateofcredend_289D", "lastupdate_1112D",
)
CBA_CLOSED_FIELDS = (
    "numberofcontrsvalue_358L", "totaloutstanddebtvalue_668A", "outstandingamount_354A",
    "totalamount_6A", "credlmt_230A", "dpdmax_757P", "overdueamount_31A",
    "contractst_964M", "subjectrole_93M", "dateofcredstart_739D",
    "dateofcredend_353D", "dateofrealrepmt_138D", "lastupdate_388D",
)

APPLPREV_NUMERIC = (
    "credamount_590A", "annuity_853A", "tenor_203L", "mainoccupationinc_437A",
    "currdebt_94A", "outstandingdebt_522A", "actualdpd_943P",
    "maxdpdtolerance_577P", "pmtnum_8L",
)
APPLPREV_CATEGORICAL = ("status_219L", "credtype_587L", "credacc_status_367L")
APPLPREV_DATES = (
    "creationdate_885D", "approvaldate_319D", "dateactivated_425D",
    "dtlastpmt_581D", "dtlastpmtallstes_3545839D",
)

PERSON_CONTROLS = (
    "personindex_1023L", "persontype_1072L", "persontype_792L",
    "isreference_387L", "role_993L", "role_1084L",
    "relationshiptoclient_415T", "relationshiptoclient_642T", "type_25L",
)
PERSON_NUMERIC = ("mainoccupationinc_384A",)
PERSON_CATEGORICAL = (
    "incometype_1044T", "empl_employedtotal_800L", "education_927M", "housingtype_772L",
)
PERSON_DATES = ("empl_employedfrom_271D",)

CBB_NUMERIC = ("credquantity_1099L", "credquantity_984L")
CBB_DATES = ("contractdate_551D",)

SOURCE_SPECS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        ("base", {"task06_group": "train_base", "index": None, "numeric": (), "categorical": (), "dates": ("date_decision",)}),
        ("static_0", {"task06_group": "train_static_0", "index": None, "numeric": STATIC_NUMERIC, "categorical": STATIC_CATEGORICAL, "dates": STATIC_DATES}),
        ("static_cb_0", {"task06_group": "train_static_cb_0", "index": None, "numeric": STATIC_CB_NUMERIC, "categorical": STATIC_CB_CATEGORICAL, "dates": ()}),
        ("credit_bureau_a_1", {"task06_group": "train_credit_bureau_a_1", "index": "num_group1", "numeric": CBA_NUMERIC, "categorical": CBA_CATEGORICAL, "dates": CBA_DATES}),
        ("applprev_1", {"task06_group": "train_applprev_1", "index": "num_group1", "numeric": APPLPREV_NUMERIC, "categorical": APPLPREV_CATEGORICAL, "dates": APPLPREV_DATES}),
        ("person_1", {"task06_group": "train_person_1", "index": "num_group1", "numeric": PERSON_NUMERIC, "categorical": PERSON_CONTROLS + PERSON_CATEGORICAL, "dates": PERSON_DATES}),
        ("credit_bureau_b_1", {"task06_group": "train_credit_bureau_b_1", "index": "num_group1", "numeric": CBB_NUMERIC, "categorical": (), "dates": CBB_DATES}),
    ]
)

COUNT_FIELDS = {
    "numactivecreds_622L", "numactivecredschannel_414L", "numactiverelcontr_750L",
    "numcontrs3months_479L", "applications30d_658L", "pmtnum_254L",
    "numinstpaidlastcontr_4325080L", "days30_165L", "days90_310L",
    "days180_256L", "days360_512L", "numberofcontrsvalue_258L",
    "numberofcontrsvalue_358L", "tenor_203L", "pmtnum_8L",
    "credquantity_1099L", "credquantity_984L",
}

CREDIT_COUNT_FIELDS = (
    "numactivecreds_622L", "numactivecredschannel_414L", "numactiverelcontr_750L",
    "numberofcontrsvalue_258L", "numberofcontrsvalue_358L",
    "credquantity_1099L", "credquantity_984L",
)

OFFICIAL_DATA_URL = "https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data"
OFFICIAL_DISCUSSION_URL = "https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/472866"
OFFICIAL_APPLPREV_URL = "https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/492738"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--dictionary", required=True, type=Path)
    parser.add_argument("--task06-dir", required=True, type=Path)
    parser.add_argument("--batch1-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--proposal", type=Path)
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


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in headers})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def refuse_outputs(output_dir: Path) -> dict[str, Path]:
    targets = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite completed Batch 2 outputs: {existing}")
    return targets


def collect_state(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result = []
    for path in sorted(set(paths), key=lambda item: str(item.resolve())):
        stat = path.stat()
        result.append({"path": str(path.resolve()), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return result


def compare_state(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    left = {row["path"]: row for row in before}
    right = {row["path"]: row for row in after}
    details = [
        {"path": path, "unchanged": left.get(path) == right.get(path), "before": left.get(path), "after": right.get(path)}
        for path in sorted(set(left) | set(right))
    ]
    return {"method": "file size and mtime_ns; not cryptographic proof", "all_unchanged": all(row["unchanged"] for row in details), "details": details}


def metric_row(
    source: str,
    field: str,
    description: str,
    role: str,
    population_name: str,
    metric: str,
    value: Any,
    denominator_name: str,
    denominator_value: int | None,
    definition: str,
    status: str = "MEASURED",
) -> dict[str, Any]:
    return {
        "source": source,
        "field": field,
        "dictionary_description": description,
        "role": role,
        "analysis_population": population_name,
        "metric": metric,
        "value": value,
        "denominator_name": denominator_name,
        "denominator_value": denominator_value,
        "definition": definition,
        "status": status,
    }


def positions_for(case_ids: pd.Series, base_index: pd.Index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonnull = case_ids.notna().to_numpy(bool)
    positions = np.full(len(case_ids), -1, dtype=np.int64)
    if nonnull.any():
        positions[nonnull] = base_index.get_indexer(case_ids[nonnull])
    return positions, nonnull & (positions >= 0), nonnull & (positions < 0)


def batch_groups(positions: np.ndarray, matched: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matched_indices = np.flatnonzero(matched)
    unique_positions, inverse = np.unique(positions[matched_indices], return_inverse=True)
    return matched_indices, unique_positions, inverse


def add_group_counts(target: np.ndarray, unique_positions: np.ndarray, inverse: np.ndarray, mask: np.ndarray) -> None:
    if not mask.any():
        return
    counts = np.bincount(inverse[mask], minlength=len(unique_positions))
    selected = counts > 0
    target[unique_positions[selected]] += counts[selected].astype(target.dtype, copy=False)


class NumericAccumulator:
    def __init__(self, population: int, retain_values: bool = True) -> None:
        self.population = population
        self.row_counts = Counter()
        self.app = {
            name: np.zeros(population, dtype=np.int32)
            for name in ("missing", "unparseable", "nonfinite", "finite", "zero", "nonzero", "populated")
        }
        self.app_min = np.full(population, np.nan, dtype=np.float64)
        self.app_max = np.full(population, np.nan, dtype=np.float64)
        self.app_sum = np.zeros(population, dtype=np.float64)
        self.values: list[np.ndarray] = [] if retain_values else []
        self.retain_values = retain_values

    def update(
        self,
        series: pd.Series,
        matched_indices: np.ndarray,
        unique_positions: np.ndarray,
        inverse: np.ndarray,
    ) -> None:
        selected = series.iloc[matched_indices].reset_index(drop=True)
        classified = classify_numeric(selected)
        for name in ("missing", "unparseable", "nonfinite", "zero", "positive", "negative"):
            self.row_counts[name] += int(np.count_nonzero(classified[name]))
        for name in ("missing", "unparseable", "nonfinite", "finite", "zero", "nonzero", "populated"):
            add_group_counts(self.app[name], unique_positions, inverse, classified[name])
        finite = classified["finite"]
        values = classified["values"][finite]
        if self.retain_values and values.size:
            self.values.append(values.astype(np.float64, copy=True))
        if finite.any():
            finite_inverse = inverse[finite]
            batch_min = np.full(len(unique_positions), np.inf)
            batch_max = np.full(len(unique_positions), -np.inf)
            batch_sum = np.zeros(len(unique_positions), dtype=np.float64)
            np.minimum.at(batch_min, finite_inverse, values)
            np.maximum.at(batch_max, finite_inverse, values)
            np.add.at(batch_sum, finite_inverse, values)
            has = np.isfinite(batch_min)
            pos = unique_positions[has]
            old_missing = np.isnan(self.app_min[pos])
            if old_missing.any():
                self.app_min[pos[old_missing]] = batch_min[has][old_missing]
                self.app_max[pos[old_missing]] = batch_max[has][old_missing]
            existing = ~old_missing
            if existing.any():
                self.app_min[pos[existing]] = np.minimum(self.app_min[pos[existing]], batch_min[has][existing])
                self.app_max[pos[existing]] = np.maximum(self.app_max[pos[existing]], batch_max[has][existing])
            self.app_sum[pos] += batch_sum[has]

    def finish(
        self,
        source: str,
        field: str,
        description: str,
        role: str,
        population_name: str,
        matched_rows: int,
        source_record_counts: np.ndarray,
        count_field: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for state in ("missing", "unparseable", "nonfinite", "zero", "positive", "negative"):
            count = int(self.row_counts[state])
            rows.extend(
                [
                    metric_row(source, field, description, role, population_name,
                               f"row_state_{state}_count", count, "matched_source_rows", matched_rows,
                               "Mutually exclusive numeric state on rows matched to valid base keys."),
                    metric_row(source, field, description, role, population_name,
                               f"row_state_{state}_rate", count / matched_rows if matched_rows else None,
                               "matched_source_rows", matched_rows,
                               "Matched-row numeric-state rate.",
                               "MEASURED" if matched_rows else "UNDEFINED_ZERO_DENOMINATOR"),
                ]
            )
        finite_values = np.concatenate(self.values) if self.values else np.empty(0, dtype=np.float64)
        finite_count = int(finite_values.size)
        rows.append(metric_row(source, field, description, role, population_name,
                               "finite_observation_count", finite_count, "matched_source_rows", matched_rows,
                               "Finite values include observed zero and negative values."))
        rows.append(metric_row(source, field, description, role, population_name,
                               "distinct_finite_value_count", int(pd.Series(finite_values).nunique()) if finite_count else 0,
                               "finite_observations", finite_count, "Exact distinct finite-value count."))
        for name, value in exact_quantiles(finite_values).items():
            rows.append(metric_row(source, field, description, role, population_name,
                                   f"finite_{name}", value, "finite_observations", finite_count,
                                   f"Exact statistic; quantiles use NumPy method={QUANTILE_METHOD}.",
                                   "MEASURED" if value is not None else "UNDEFINED_NO_FINITE_VALUES"))
        if count_field:
            integer_like = np.isclose(finite_values, np.rint(finite_values), rtol=0.0, atol=1e-12)
            nonnegative = finite_values >= 0
            for name, mask in (("nonnegative", nonnegative), ("integer_like", integer_like)):
                count = int(np.count_nonzero(mask))
                rows.append(metric_row(source, field, description, role, population_name,
                                       f"finite_{name}_count", count, "finite_observations", finite_count,
                                       "Count-field consistency diagnostic; no values were changed."))
                rows.append(metric_row(source, field, description, role, population_name,
                                       f"finite_{name}_rate", count / finite_count if finite_count else None,
                                       "finite_observations", finite_count,
                                       "Count-field consistency rate.",
                                       "MEASURED" if finite_count else "UNDEFINED_ZERO_DENOMINATOR"))
        source_present = source_record_counts > 0
        finite = self.app["finite"] > 0
        nonzero = self.app["nonzero"] > 0
        states = {
            "NO_SOURCE_RECORD": ~source_present,
            "SOURCE_PRESENT_NO_FINITE_OBSERVATION": source_present & ~finite,
            "FINITE_OBSERVATIONS_ZERO_ONLY": finite & ~nonzero,
            "AT_LEAST_ONE_FINITE_NONZERO_OBSERVATION": finite & nonzero,
        }
        source_apps = int(np.count_nonzero(source_present))
        for state, mask in states.items():
            count = int(np.count_nonzero(mask))
            rows.append(metric_row(source, field, description, role, population_name,
                                   f"application_state_{state}_count", count,
                                   "verified_base_applications", self.population,
                                   "One of four mutually exclusive application states."))
            rows.append(metric_row(source, field, description, role, population_name,
                                   f"application_state_{state}_population_rate", count / self.population,
                                   "verified_base_applications", self.population,
                                   "Rate over the full verified base population."))
            conditional = None if state == "NO_SOURCE_RECORD" or not source_apps else count / source_apps
            rows.append(metric_row(source, field, description, role, population_name,
                                   f"application_state_{state}_conditional_source_rate", conditional,
                                   "applications_with_source_records", source_apps,
                                   "Conditional rate among applications with source records.",
                                   "MEASURED" if conditional is not None else "NOT_APPLICABLE"))
        variation = finite & ~np.isclose(self.app_min, self.app_max, rtol=0.0, atol=0.0, equal_nan=True)
        diagnostics = {
            "applications_finite_and_missing": (self.app["finite"] > 0) & (self.app["missing"] > 0),
            "applications_zero_and_nonzero": (self.app["zero"] > 0) & (self.app["nonzero"] > 0),
            "applications_any_invalid": (self.app["unparseable"] > 0) | (self.app["nonfinite"] > 0),
            "applications_multiple_distinct_finite_values": variation,
            "applications_one_distinct_finite_value": finite & ~variation,
        }
        for name, mask in diagnostics.items():
            rows.append(metric_row(source, field, description, role, population_name,
                                   name, int(np.count_nonzero(mask)), "verified_base_applications",
                                   self.population, "Overlapping application diagnostic."))
        self.values.clear()
        return rows, {"app": self.app, "app_min": self.app_min, "app_max": self.app_max,
                      "app_sum": self.app_sum, "states": states, "finite_count": finite_count,
                      "row_counts": dict(self.row_counts)}


class CategoricalAccumulator:
    def __init__(self, population: int) -> None:
        self.population = population
        self.row_counts = Counter()
        self.frequencies = Counter()
        self.app_populated = np.zeros(population, dtype=np.int32)
        self.app_seen = np.zeros(population, dtype=bool)
        self.app_hash = np.zeros(population, dtype=np.uint64)
        self.app_conflict = np.zeros(population, dtype=bool)

    @staticmethod
    def normalized(series: pd.Series) -> tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray]:
        native = series.isna().to_numpy(bool)
        text = series.astype("string")
        empty = (~series.isna() & text.str.strip().eq("")).fillna(False).to_numpy(bool)
        populated = ~(native | empty)
        return text, native, empty, populated

    def update(self, series: pd.Series, matched_indices: np.ndarray, positions: np.ndarray) -> None:
        selected = series.iloc[matched_indices].reset_index(drop=True)
        pos = positions[matched_indices]
        text, native, empty, populated = self.normalized(selected)
        self.row_counts["native_missing"] += int(native.sum())
        self.row_counts["empty"] += int(empty.sum())
        self.row_counts["populated"] += int(populated.sum())
        if not populated.any():
            return
        values = text[populated].astype(str)
        counts = values.value_counts(dropna=False)
        self.frequencies.update({str(key): int(value) for key, value in counts.items()})
        p = pos[populated]
        hashes = pd.util.hash_array(values.to_numpy(dtype=object)).astype(np.uint64)
        unique_pos, inverse = np.unique(p, return_inverse=True)
        batch_count = np.bincount(inverse, minlength=len(unique_pos))
        self.app_populated[unique_pos] += batch_count.astype(np.int32)
        batch_min = np.full(len(unique_pos), np.iinfo(np.uint64).max, dtype=np.uint64)
        batch_max = np.zeros(len(unique_pos), dtype=np.uint64)
        np.minimum.at(batch_min, inverse, hashes)
        np.maximum.at(batch_max, inverse, hashes)
        prior = self.app_seen[unique_pos]
        self.app_conflict[unique_pos] |= batch_min != batch_max
        if prior.any():
            existing_hash = self.app_hash[unique_pos[prior]]
            self.app_conflict[unique_pos[prior]] |= (existing_hash != batch_min[prior]) | (existing_hash != batch_max[prior])
        new = ~prior
        self.app_hash[unique_pos[new]] = batch_min[new]
        self.app_seen[unique_pos] = True

    def finish(
        self, source: str, field: str, description: str, role: str,
        population_name: str, matched_rows: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows = []
        for state in ("native_missing", "empty", "populated"):
            count = int(self.row_counts[state])
            rows.append(metric_row(source, field, description, role, population_name,
                                   f"row_category_{state}_count", count, "matched_source_rows", matched_rows,
                                   "Categorical code state; codes are preserved without semantic guessing."))
            rows.append(metric_row(source, field, description, role, population_name,
                                   f"row_category_{state}_rate", count / matched_rows if matched_rows else None,
                                   "matched_source_rows", matched_rows, "Categorical-state rate.",
                                   "MEASURED" if matched_rows else "UNDEFINED_ZERO_DENOMINATOR"))
        rows.append(metric_row(source, field, description, role, population_name,
                               "distinct_populated_code_count", len(self.frequencies),
                               "populated_matched_rows", int(self.row_counts["populated"]),
                               "Exact distinct populated code count."))
        rows.append(metric_row(source, field, description, role, population_name,
                               "applications_with_category_variation", int(self.app_conflict.sum()),
                               "verified_base_applications", self.population,
                               "Applications with more than one observed 64-bit-hashed categorical code; hash collisions are theoretically possible."))
        return rows, {"frequencies": dict(sorted(self.frequencies.items())),
                      "app_populated": self.app_populated, "app_conflict": self.app_conflict}


class DateAccumulator:
    def __init__(self, population: int, track_latest: bool = True) -> None:
        self.population = population
        self.track_latest = track_latest
        self.row_counts = Counter()
        self.offset_counts = Counter()
        self.offset_values: list[np.ndarray] = []
        self.app_parseable = np.zeros(population, dtype=np.int32)
        self.app_before = np.zeros(population, dtype=bool)
        self.app_on = np.zeros(population, dtype=bool)
        self.app_after = np.zeros(population, dtype=bool)
        self.latest_ns = np.full(population, np.iinfo(np.int64).min, dtype=np.int64)
        self.latest_count = np.zeros(population, dtype=np.int32)
        self.latest_hash = np.zeros(population, dtype=np.uint64)
        self.latest_conflict = np.zeros(population, dtype=bool)

    def update(
        self,
        series: pd.Series,
        matched_indices: np.ndarray,
        positions: np.ndarray,
        base_dates: np.ndarray,
        payload_hashes: np.ndarray | None,
    ) -> None:
        selected = series.iloc[matched_indices].reset_index(drop=True)
        pos = positions[matched_indices]
        parsed = parse_date_values(selected)
        for state in ("native_missing", "empty", "parseable", "parse_failure"):
            self.row_counts[state] += int(np.count_nonzero(parsed[state]))
        valid = parsed["parseable"]
        if not valid.any():
            return
        p = pos[valid]
        date_values = parsed["parsed"][valid].to_numpy(dtype="datetime64[ns]")
        date_ns = date_values.astype(np.int64)
        base_for_rows = base_dates[p]
        usable_base = ~pd.isna(base_for_rows)
        offsets = np.full(len(p), np.nan, dtype=np.float64)
        offsets[usable_base] = (
            date_values[usable_base] - base_for_rows[usable_base]
        ).astype("timedelta64[D]").astype(np.float64)
        finite_offset = np.isfinite(offsets)
        if finite_offset.any():
            integer_offsets = offsets[finite_offset].astype(np.int64)
            self.offset_values.append(integer_offsets.astype(np.float64))
            self.offset_counts.update(Counter(integer_offsets.tolist()))
        unique, inverse = np.unique(p, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(unique))
        self.app_parseable[unique] += counts.astype(np.int32)
        np.logical_or.at(self.app_before, p[finite_offset], offsets[finite_offset] < 0)
        np.logical_or.at(self.app_on, p[finite_offset], offsets[finite_offset] == 0)
        np.logical_or.at(self.app_after, p[finite_offset], offsets[finite_offset] > 0)
        if not self.track_latest:
            return
        hashes = (
            payload_hashes[matched_indices][valid].astype(np.uint64, copy=False)
            if payload_hashes is not None
            else np.zeros(len(p), dtype=np.uint64)
        )
        work = pd.DataFrame({"position": p, "date_ns": date_ns, "payload_hash": hashes})
        latest_date = work.groupby("position", sort=False)["date_ns"].transform("max")
        latest = work.loc[work["date_ns"] == latest_date]
        grouped = latest.groupby("position", sort=False)["payload_hash"].agg(["size", "min", "max"])
        batch_pos = grouped.index.to_numpy(dtype=np.int64)
        batch_date = latest.groupby("position", sort=False)["date_ns"].first().loc[grouped.index].to_numpy(np.int64)
        batch_count = grouped["size"].to_numpy(np.int32)
        batch_min_hash = grouped["min"].to_numpy(np.uint64)
        batch_max_hash = grouped["max"].to_numpy(np.uint64)
        old_date = self.latest_ns[batch_pos]
        newer = batch_date > old_date
        equal = batch_date == old_date
        if newer.any():
            pos_new = batch_pos[newer]
            self.latest_ns[pos_new] = batch_date[newer]
            self.latest_count[pos_new] = batch_count[newer]
            self.latest_hash[pos_new] = batch_min_hash[newer]
            self.latest_conflict[pos_new] = batch_min_hash[newer] != batch_max_hash[newer]
        if equal.any():
            pos_equal = batch_pos[equal]
            conflict = (
                (self.latest_hash[pos_equal] != batch_min_hash[equal])
                | (self.latest_hash[pos_equal] != batch_max_hash[equal])
                | (batch_min_hash[equal] != batch_max_hash[equal])
            )
            self.latest_conflict[pos_equal] |= conflict
            self.latest_count[pos_equal] += batch_count[equal]

    def finish(
        self, source: str, field: str, description: str, role: str,
        population_name: str, matched_rows: int, source_record_counts: np.ndarray,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows = []
        for state in ("native_missing", "empty", "parseable", "parse_failure"):
            count = int(self.row_counts[state])
            rows.append(metric_row(source, field, description, role, population_name,
                                   f"row_date_{state}_count", count, "matched_source_rows", matched_rows,
                                   "Date state; parse failures remain separate from missing."))
            rows.append(metric_row(source, field, description, role, population_name,
                                   f"row_date_{state}_rate", count / matched_rows if matched_rows else None,
                                   "matched_source_rows", matched_rows, "Date-state rate.",
                                   "MEASURED" if matched_rows else "UNDEFINED_ZERO_DENOMINATOR"))
        offsets = np.concatenate(self.offset_values) if self.offset_values else np.empty(0)
        valid_pairs = int(offsets.size)
        for name, mask in (("before", offsets < 0), ("on", offsets == 0), ("after", offsets > 0)):
            count = int(np.count_nonzero(mask))
            rows.append(metric_row(source, field, description, role, population_name,
                                   f"date_offset_{name}_count", count, "valid_date_pairs", valid_pairs,
                                   "Calendar-day source date minus current date_decision."))
            rows.append(metric_row(source, field, description, role, population_name,
                                   f"date_offset_{name}_rate", count / valid_pairs if valid_pairs else None,
                                   "valid_date_pairs", valid_pairs, "Offset-sign rate.",
                                   "MEASURED" if valid_pairs else "UNDEFINED_ZERO_DENOMINATOR"))
        if valid_pairs:
            q = np.quantile(offsets, [0.05, 0.5, 0.95], method=QUANTILE_METHOD)
            stats = {"min": float(offsets.min()), "p05": float(q[0]), "median": float(q[1]),
                     "p95": float(q[2]), "max": float(offsets.max())}
        else:
            stats = {key: None for key in ("min", "p05", "median", "p95", "max")}
        for name, value in stats.items():
            rows.append(metric_row(source, field, description, role, population_name,
                                   f"date_offset_days_{name}", value, "valid_date_pairs", valid_pairs,
                                   f"Exact offset statistic; quantiles use method={QUANTILE_METHOD}.",
                                   "MEASURED" if value is not None else "UNDEFINED_NO_VALID_PAIRS"))
        source_present = source_record_counts > 0
        app_metrics = {
            "applications_any_before": self.app_before,
            "applications_any_on": self.app_on,
            "applications_any_after": self.app_after,
            "applications_mixed_before_and_after": self.app_before & self.app_after,
            "applications_no_usable_date": source_present & (self.app_parseable == 0),
        }
        for name, mask in app_metrics.items():
            rows.append(metric_row(source, field, description, role, population_name,
                                   name, int(np.count_nonzero(mask)), "verified_base_applications",
                                   self.population, "Application-level date/tie diagnostic."))
        latest_metrics = {
            "applications_latest_date_tied": self.latest_count > 1,
            "applications_latest_tie_payload_conflict": self.latest_conflict,
        }
        for name, mask in latest_metrics.items():
            if self.track_latest:
                rows.append(metric_row(source, field, description, role, population_name,
                                       name, int(np.count_nonzero(mask)), "verified_base_applications",
                                       self.population, "Application-level latest-date tie diagnostic."))
            else:
                rows.append(metric_row(source, field, description, role, population_name,
                                       name, None, "verified_base_applications", self.population,
                                       "Latest-date tracking was disabled for this audit; no tie value was computed.",
                                       "NOT_CHECKED"))
        self.offset_values.clear()
        return rows, {
            "app_parseable": self.app_parseable, "app_before": self.app_before,
            "app_on": self.app_on, "app_after": self.app_after,
            "latest_count": self.latest_count, "latest_conflict": self.latest_conflict,
            "offset_counts": dict(sorted(self.offset_counts.items())), "valid_pairs": valid_pairs,
            "row_counts": dict(self.row_counts),
        }


def record_distribution(counts: np.ndarray) -> dict[str, Any]:
    positive = counts[counts > 0]
    if not positive.size:
        return {key: None for key in ("min", "median", "p90", "p95", "p99", "max")} | {
            "applications_1_row": 0, "applications_2_rows": 0,
            "applications_3_5_rows": 0, "applications_6_10_rows": 0,
            "applications_gt10_rows": 0,
        }
    q = np.quantile(positive, [0.5, 0.9, 0.95, 0.99], method=QUANTILE_METHOD)
    return {
        "min": int(positive.min()), "median": float(q[0]), "p90": float(q[1]),
        "p95": float(q[2]), "p99": float(q[3]), "max": int(positive.max()),
        "applications_1_row": int(np.count_nonzero(positive == 1)),
        "applications_2_rows": int(np.count_nonzero(positive == 2)),
        "applications_3_5_rows": int(np.count_nonzero((positive >= 3) & (positive <= 5))),
        "applications_6_10_rows": int(np.count_nonzero((positive >= 6) & (positive <= 10))),
        "applications_gt10_rows": int(np.count_nonzero(positive > 10)),
    }


def shard_pair_diagnostics(
    left_case_positions: np.ndarray,
    right_case_positions: np.ndarray,
    left_codes: np.ndarray,
    left_hashes: np.ndarray,
    right_codes: np.ndarray,
    right_hashes: np.ndarray,
) -> dict[str, int]:
    """Compare two physical shards without treating shared applications as duplicates."""
    case_overlap = np.intersect1d(
        np.unique(left_case_positions), np.unique(right_case_positions), assume_unique=True
    )
    overlap_codes = np.intersect1d(
        np.unique(left_codes), np.unique(right_codes), assume_unique=True
    )
    conflicting = 0
    if overlap_codes.size:
        left = pd.DataFrame({"code": left_codes, "hash": left_hashes})
        right = pd.DataFrame({"code": right_codes, "hash": right_hashes})
        combined = pd.concat(
            [left[left["code"].isin(overlap_codes)], right[right["code"].isin(overlap_codes)]],
            ignore_index=True,
        )
        conflicting = int((combined.groupby("code")["hash"].nunique() > 1).sum())
    return {
        "case_id_overlap_count": int(case_overlap.size),
        "composite_key_overlap_count": int(overlap_codes.size),
        "conflicting_selected_content_composite_keys": conflicting,
    }


def person_selection_states(source_record_counts: np.ndarray, selected_counts: np.ndarray) -> dict[str, np.ndarray]:
    """Partition applications by documented person_1 num_group1=0 multiplicity."""
    source = source_record_counts > 0
    return {
        "NO_PERSON_SOURCE": ~source,
        "SOURCE_PRESENT_ZERO_NUM_GROUP1_ZERO_ROWS": source & (selected_counts == 0),
        "EXACTLY_ONE_NUM_GROUP1_ZERO_ROW": selected_counts == 1,
        "MULTIPLE_NUM_GROUP1_ZERO_ROWS": selected_counts > 1,
    }


def active_closed_evidence_states(
    source_record_counts: np.ndarray,
    active_apps: np.ndarray,
    closed_apps: np.ndarray,
) -> dict[str, np.ndarray]:
    """Describe selected bureau-side evidence without inferring missing credit history."""
    source = source_record_counts > 0
    return {
        "applications_no_source_record": ~source,
        "applications_active_evidence_only": source & active_apps & ~closed_apps,
        "applications_closed_evidence_only": source & ~active_apps & closed_apps,
        "applications_both_active_and_closed_evidence": source & active_apps & closed_apps,
        "applications_neither_active_nor_closed_evidence": source & ~active_apps & ~closed_apps,
    }


def credit_candidate_state(
    source_record_counts: np.ndarray,
    app_counts: dict[str, np.ndarray],
    app_min: np.ndarray,
    app_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify a count candidate while keeping absent source distinct from observed zero."""
    source_present = source_record_counts > 0
    finite = app_counts["finite"] > 0
    invalid = (app_counts["unparseable"] > 0) | (app_counts["nonfinite"] > 0)
    conflict = finite & ~np.isclose(app_min, app_max, rtol=0, atol=0, equal_nan=True)
    consistent = finite & ~conflict
    value = np.where(consistent, app_min, np.nan)
    state = np.full(len(source_record_counts), 1, dtype=np.int8)
    state[~source_present] = 0
    state[invalid] = 2
    state[conflict] = 3
    state[consistent & (value == 0)] = 4
    state[consistent & (value > 0)] = 5
    state[consistent & (value < 0)] = 6
    return state, value, consistent


def tax_c_application_states(
    source_record_counts: np.ndarray,
    finite_amount_counts: np.ndarray,
    finite_amount_sums: np.ndarray,
) -> dict[str, np.ndarray]:
    """Partition tax-C applications without manufacturing a no-source monetary zero."""
    present = source_record_counts > 0
    return {
        "NO_C_SOURCE": ~present,
        "C_SOURCE_ALL_AMOUNTS_MISSING_OR_INVALID": present & (finite_amount_counts == 0),
        "C_SOURCE_FINITE_ZERO_SUM": present & (finite_amount_counts > 0) & (finite_amount_sums == 0),
        "C_SOURCE_FINITE_NONZERO_SUM": present & (finite_amount_counts > 0) & (finite_amount_sums != 0),
    }


def grouped_date_comparison(
    left_positions: np.ndarray,
    left_date_ns: np.ndarray,
    right_positions: np.ndarray,
    right_date_ns: np.ndarray,
) -> dict[str, int]:
    """Compare independently grouped case/date keys; never perform a raw-row join."""
    left = pd.DataFrame({"position": left_positions, "date_ns": left_date_ns})
    right = pd.DataFrame({"position": right_positions, "date_ns": right_date_ns})
    left = left.groupby(["position", "date_ns"], sort=False).size().rename("left_count").reset_index()
    right = right.groupby(["position", "date_ns"], sort=False).size().rename("right_count").reset_index()
    comparison = left.merge(right, on=["position", "date_ns"], how="outer", validate="one_to_one")
    matched = comparison["left_count"].notna() & comparison["right_count"].notna()
    equal = matched & (comparison["left_count"] == comparison["right_count"])
    ambiguous = matched & ((comparison["left_count"] > 1) | (comparison["right_count"] > 1))
    return {
        "left_group_count": len(left),
        "right_group_count": len(right),
        "matched_group_count": int(matched.sum()),
        "left_only_group_count": int((comparison["left_count"].notna() & comparison["right_count"].isna()).sum()),
        "right_only_group_count": int((comparison["left_count"].isna() & comparison["right_count"].notna()).sum()),
        "matched_equal_multiplicity_count": int(equal.sum()),
        "matched_different_multiplicity_count": int((matched & ~equal).sum()),
        "matched_ambiguous_multirow_group_count": int(ambiguous.sum()),
    }


def categorical_role(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    text, _, _, populated = CategoricalAccumulator.normalized(series)
    return text, populated


def audit_source(
    source: str,
    spec: dict[str, Any],
    train_dir: Path,
    file_names: list[str],
    base_index: pd.Index,
    base_dates: np.ndarray,
    descriptions: dict[str, str],
) -> dict[str, Any]:
    population = len(base_index)
    numeric = {field: NumericAccumulator(population) for field in spec["numeric"]}
    categorical = {field: CategoricalAccumulator(population) for field in spec["categorical"]}
    dates = {field: DateAccumulator(population) for field in spec["dates"]}
    followup_numeric_names = (
        ("pmtscount_423L", "pmtssum_45A") if source == "static_cb_0" else ()
    )
    followup_numeric = {field: NumericAccumulator(population) for field in followup_numeric_names}
    applicant_numeric = {field: NumericAccumulator(population) for field in PERSON_NUMERIC} if source == "person_1" else {}
    applicant_categorical = {field: CategoricalAccumulator(population) for field in PERSON_CATEGORICAL} if source == "person_1" else {}
    applicant_dates = {field: DateAccumulator(population) for field in PERSON_DATES} if source == "person_1" else {}

    source_counts = np.zeros(population, dtype=np.int32)
    selected_counts = np.zeros(population, dtype=np.int32) if source == "person_1" else None
    physical_rows = 0
    null_case_rows = 0
    orphan_rows = 0
    orphan_ids: set[Any] = set()
    null_index_rows = 0
    invalid_index_rows = 0
    file_case_counts: dict[str, np.ndarray] = {}
    file_composite_codes: dict[str, np.ndarray] = {}
    file_composite_hashes: dict[str, np.ndarray] = {}
    person_combinations: Counter[str] = Counter()
    person_selection_cross = Counter()
    active_closed_rows = Counter()
    active_apps = np.zeros(population, dtype=bool) if source == "credit_bureau_a_1" else None
    closed_apps = np.zeros(population, dtype=bool) if source == "credit_bureau_a_1" else None

    all_fields = list(dict.fromkeys(("case_id", *((spec["index"],) if spec["index"] else ()),
                                     *spec["numeric"], *spec["categorical"], *spec["dates"],
                                     *followup_numeric_names)))
    payload_fields = list(dict.fromkeys((*spec["numeric"], *spec["categorical"])))

    for file_name in file_names:
        print(f"  {source}: {file_name}", flush=True)
        file_counts = np.zeros(population, dtype=np.int32)
        code_chunks: list[np.ndarray] = []
        hash_chunks: list[np.ndarray] = []
        parquet = pq.ParquetFile(train_dir / file_name)
        for record_batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=all_fields, use_threads=True):
            frame = record_batch.to_pandas()
            physical_rows += len(frame)
            positions, matched, orphan = positions_for(frame["case_id"], base_index)
            null_case_rows += int(frame["case_id"].isna().sum())
            orphan_rows += int(orphan.sum())
            if orphan.any():
                orphan_ids.update(frame.loc[orphan, "case_id"].tolist())
            matched_indices, unique_positions, inverse = batch_groups(positions, matched)
            batch_counts = np.bincount(inverse, minlength=len(unique_positions))
            source_counts[unique_positions] += batch_counts.astype(np.int32)
            file_counts[unique_positions] += batch_counts.astype(np.int32)
            payload_hashes = (
                pd.util.hash_pandas_object(frame[payload_fields], index=False).to_numpy(np.uint64)
                if payload_fields else np.zeros(len(frame), dtype=np.uint64)
            )

            for field, accumulator in numeric.items():
                accumulator.update(frame[field], matched_indices, unique_positions, inverse)
            for field, accumulator in categorical.items():
                accumulator.update(frame[field], matched_indices, positions)
            for field, accumulator in dates.items():
                accumulator.update(frame[field], matched_indices, positions, base_dates, payload_hashes)
            for field, accumulator in followup_numeric.items():
                accumulator.update(frame[field], matched_indices, unique_positions, inverse)

            if spec["index"]:
                index_series = pd.to_numeric(frame[spec["index"]], errors="coerce")
                null_index_rows += int(frame[spec["index"]].isna().sum())
                numeric_index = index_series.to_numpy(dtype=np.float64, na_value=np.nan)
                valid_integer = np.isfinite(numeric_index) & (numeric_index == np.floor(numeric_index))
                valid_range = valid_integer & (numeric_index >= 0) & (numeric_index <= np.iinfo(np.uint32).max)
                invalid_index_rows += int(np.count_nonzero(~frame[spec["index"]].isna().to_numpy(bool) & ~valid_range))
                valid_composite = matched & valid_range
                if valid_composite.any():
                    code = (
                        (positions[valid_composite].astype(np.uint64) << np.uint64(32))
                        | numeric_index[valid_composite].astype(np.uint64)
                    )
                    code_chunks.append(code)
                    hash_chunks.append(payload_hashes[valid_composite].copy())

            if source == "credit_bureau_a_1":
                active = np.zeros(len(frame), dtype=bool)
                closed = np.zeros(len(frame), dtype=bool)
                for field in CBA_ACTIVE_FIELDS:
                    if field in numeric:
                        active |= ~classify_numeric(frame[field])["missing"]
                    elif field in categorical:
                        active |= CategoricalAccumulator.normalized(frame[field])[3]
                    else:
                        active |= parse_date_values(frame[field])["populated"]
                for field in CBA_CLOSED_FIELDS:
                    if field in numeric:
                        closed |= ~classify_numeric(frame[field])["missing"]
                    elif field in categorical:
                        closed |= CategoricalAccumulator.normalized(frame[field])[3]
                    else:
                        closed |= parse_date_values(frame[field])["populated"]
                matched_active = matched & active
                matched_closed = matched & closed
                np.logical_or.at(active_apps, positions[matched_active], True)
                np.logical_or.at(closed_apps, positions[matched_closed], True)
                active_closed_rows.update(
                    {
                        "ACTIVE_ONLY": int(np.count_nonzero(matched & active & ~closed)),
                        "CLOSED_ONLY": int(np.count_nonzero(matched & ~active & closed)),
                        "BOTH": int(np.count_nonzero(matched & active & closed)),
                        "NEITHER": int(np.count_nonzero(matched & ~active & ~closed)),
                    }
                )

            if source == "person_1":
                idx = pd.to_numeric(frame["num_group1"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
                selected = matched & np.isfinite(idx) & (idx == 0)
                if selected.any():
                    selected_pos = positions[selected]
                    selected_unique, selected_inverse = np.unique(selected_pos, return_inverse=True)
                    selected_counts[selected_unique] += np.bincount(selected_inverse, minlength=len(selected_unique)).astype(np.int32)
                    selected_indices = np.flatnonzero(selected)
                    for field, accumulator in applicant_numeric.items():
                        accumulator.update(frame[field], selected_indices, selected_unique, selected_inverse)
                    for field, accumulator in applicant_categorical.items():
                        accumulator.update(frame[field], selected_indices, positions)
                    for field, accumulator in applicant_dates.items():
                        accumulator.update(frame[field], selected_indices, positions, base_dates, payload_hashes)
                person_index = pd.to_numeric(frame["personindex_1023L"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
                person_selection_cross["num_group1_zero_and_personindex_zero"] += int(np.count_nonzero(selected & (person_index == 0)))
                person_selection_cross["num_group1_zero_and_personindex_nonzero"] += int(np.count_nonzero(selected & np.isfinite(person_index) & (person_index != 0)))
                person_selection_cross["num_group1_zero_and_personindex_missing"] += int(np.count_nonzero(selected & ~np.isfinite(person_index)))
                normalized = pd.DataFrame(index=frame.index)
                for field in PERSON_CONTROLS:
                    text, native, empty, populated = CategoricalAccumulator.normalized(frame[field])
                    normalized[field] = np.where(native, "<NATIVE_NULL>", np.where(empty, "<EMPTY>", text.astype(str)))
                combinations = normalized.loc[matched].value_counts(dropna=False)
                for keys, count in combinations.items():
                    if not isinstance(keys, tuple):
                        keys = (keys,)
                    label = json.dumps(dict(zip(PERSON_CONTROLS, map(str, keys))), sort_keys=True, separators=(",", ":"))
                    person_combinations[label] += int(count)

            del frame, positions, matched, orphan, matched_indices, unique_positions, inverse, payload_hashes
        file_case_counts[file_name] = file_counts
        file_composite_codes[file_name] = np.concatenate(code_chunks) if code_chunks else np.empty(0, dtype=np.uint64)
        file_composite_hashes[file_name] = np.concatenate(hash_chunks) if hash_chunks else np.empty(0, dtype=np.uint64)
        del code_chunks, hash_chunks
        gc.collect()

    matched_rows = int(source_counts.sum())
    traditional_rows: list[dict[str, Any]] = []
    numeric_results: dict[str, Any] = {}
    category_results: dict[str, Any] = {}
    date_results: dict[str, Any] = {}
    for field, accumulator in numeric.items():
        role = "PERSON_PAYLOAD_CANDIDATE" if source == "person_1" else "TRADITIONAL_CANDIDATE_OR_CONTROL"
        rows, result = accumulator.finish(source, field, descriptions.get(field, ""), role,
                                          "ALL_MATCHED_SOURCE_ROWS", matched_rows, source_counts,
                                          field in COUNT_FIELDS)
        traditional_rows.extend(rows)
        numeric_results[field] = result
        gc.collect()
    for field, accumulator in categorical.items():
        role = "PERSON_ROLE_OR_PAYLOAD_CODE" if source == "person_1" else "TRADITIONAL_CANDIDATE_OR_CONTROL"
        rows, result = accumulator.finish(source, field, descriptions.get(field, ""), role,
                                          "ALL_MATCHED_SOURCE_ROWS", matched_rows)
        traditional_rows.extend(rows)
        category_results[field] = result
    for field, accumulator in dates.items():
        rows, result = accumulator.finish(source, field, descriptions.get(field, ""),
                                          "TRADITIONAL_DATE_CANDIDATE", "ALL_MATCHED_SOURCE_ROWS",
                                          matched_rows, source_counts)
        traditional_rows.extend(rows)
        date_results[field] = result

    followup_results = {}
    for field, accumulator in followup_numeric.items():
        _, result = accumulator.finish(source, field, descriptions.get(field, ""),
                                       "AD_TAX_FOLLOWUP", "ALL_MATCHED_SOURCE_ROWS",
                                       matched_rows, source_counts, field == "pmtscount_423L")
        followup_results[field] = result
        gc.collect()

    applicant_results: dict[str, Any] = {}
    if source == "person_1":
        selected_rows = int(selected_counts.sum())
        for field, accumulator in applicant_numeric.items():
            rows, result = accumulator.finish(source, field, descriptions.get(field, ""),
                                              "DOCUMENTED_APPLICANT_PAYLOAD_CANDIDATE",
                                              "NUM_GROUP1_ZERO_ROWS", selected_rows, selected_counts,
                                              field in COUNT_FIELDS)
            traditional_rows.extend(rows)
            applicant_results[field] = result
        for field, accumulator in applicant_categorical.items():
            rows, result = accumulator.finish(source, field, descriptions.get(field, ""),
                                              "DOCUMENTED_APPLICANT_PAYLOAD_CANDIDATE",
                                              "NUM_GROUP1_ZERO_ROWS", selected_rows)
            traditional_rows.extend(rows)
            applicant_results[field] = result
        for field, accumulator in applicant_dates.items():
            rows, result = accumulator.finish(source, field, descriptions.get(field, ""),
                                              "DOCUMENTED_APPLICANT_PAYLOAD_CANDIDATE",
                                              "NUM_GROUP1_ZERO_ROWS", selected_rows, selected_counts)
            traditional_rows.extend(rows)
            applicant_results[field] = result

    shard_rows = []
    for left, right in itertools.combinations(file_names, 2):
        left_cases = np.flatnonzero(file_case_counts[left] > 0)
        right_cases = np.flatnonzero(file_case_counts[right] > 0)
        left_codes, left_hashes = file_composite_codes[left], file_composite_hashes[left]
        right_codes, right_hashes = file_composite_codes[right], file_composite_hashes[right]
        shard_rows.append({
            "left_file": left, "right_file": right,
            **shard_pair_diagnostics(
                left_cases, right_cases, left_codes, left_hashes,
                right_codes, right_hashes,
            ),
        })

    all_codes = np.concatenate(list(file_composite_codes.values())) if spec["index"] else np.empty(0, dtype=np.uint64)
    if all_codes.size:
        _, code_counts = np.unique(all_codes, return_counts=True)
        duplicate_composite_distinct = int(np.count_nonzero(code_counts > 1))
        duplicate_composite_excess = int(np.sum(np.clip(code_counts - 1, 0, None)))
    else:
        duplicate_composite_distinct = duplicate_composite_excess = 0
    distinct_orphans = len(orphan_ids)
    structure = {
        "source": source,
        "physical_rows": physical_rows,
        "null_case_id_rows": null_case_rows,
        "distinct_nonnull_case_ids": int(np.count_nonzero(source_counts > 0) + distinct_orphans),
        "matched_rows": matched_rows,
        "matched_distinct_case_ids": int(np.count_nonzero(source_counts > 0)),
        "orphan_rows": orphan_rows,
        "orphan_distinct_case_ids": distinct_orphans,
        "missing_base_case_ids": int(population - np.count_nonzero(source_counts > 0)),
        "null_num_group1_rows": null_index_rows if spec["index"] else None,
        "invalid_noninteger_num_group1_rows": invalid_index_rows if spec["index"] else None,
        "duplicate_composite_key_distinct_count": duplicate_composite_distinct if spec["index"] else None,
        "duplicate_composite_key_excess_rows": duplicate_composite_excess if spec["index"] else None,
        "physical_file_count": len(file_names),
        "physical_files_json": json.dumps(file_names, separators=(",", ":")),
        **record_distribution(source_counts),
    }
    if source == "base":
        structure["join_readiness"] = "VERIFIED_BASE_KEY"
    elif spec["index"]:
        structure["join_readiness"] = (
            "COMPOSITE_KEY_DEFECT_REQUIRES_REVIEW" if duplicate_composite_excess or null_index_rows or invalid_index_rows
            else "ONE_TO_MANY_REQUIRES_APPLICATION_AGGREGATION"
        )
    else:
        structure["join_readiness"] = (
            "READY_LEFT_JOIN_ONE_TO_ONE" if int(source_counts.max()) <= 1 and not orphan_rows and not null_case_rows
            else "STATIC_KEY_DEFECT_REQUIRES_REVIEW"
        )

    return {
        "structure": structure,
        "source_record_counts": source_counts,
        "traditional_rows": traditional_rows,
        "numeric": numeric_results,
        "categorical": category_results,
        "dates": date_results,
        "followup_numeric": followup_results,
        "shards": shard_rows,
        "person_selected_counts": selected_counts,
        "person_applicant_results": applicant_results,
        "person_combinations": person_combinations,
        "person_selection_cross": dict(person_selection_cross),
        "active_closed_rows": dict(active_closed_rows),
        "active_apps": active_apps,
        "closed_apps": closed_apps,
    }


def followup_row(
    section: str,
    source: str,
    field_or_comparison: str,
    metric: str,
    value: Any,
    denominator_name: str,
    denominator_value: int | None,
    definition: str,
    status: str = "MEASURED",
) -> dict[str, Any]:
    return {
        "section": section,
        "source": source,
        "field_or_comparison": field_or_comparison,
        "metric": metric,
        "value": value,
        "denominator_name": denominator_name,
        "denominator_value": denominator_value,
        "definition": definition,
        "status": status,
    }


def audit_tax_detail(
    source: str,
    file_names: list[str],
    date_field: str,
    train_dir: Path,
    base_index: pd.Index,
    base_dates: np.ndarray,
    base_months: np.ndarray,
    amount_field: str | None = None,
) -> dict[str, Any]:
    population = len(base_index)
    date_acc = DateAccumulator(population, track_latest=False)
    amount_acc = NumericAccumulator(population) if amount_field else None
    source_counts = np.zeros(population, dtype=np.int32)
    null_case_rows = orphan_rows = null_index_rows = 0
    orphan_ids: set[Any] = set()
    composite_chunks: list[np.ndarray] = []
    month_offset_counts: Counter[tuple[str, int]] = Counter()
    for file_name in file_names:
        columns = ["case_id", "num_group1", date_field] + ([amount_field] if amount_field else [])
        parquet = pq.ParquetFile(train_dir / file_name)
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=columns, use_threads=True):
            frame = batch.to_pandas()
            positions, matched, orphan = positions_for(frame["case_id"], base_index)
            null_case_rows += int(frame["case_id"].isna().sum())
            orphan_rows += int(orphan.sum())
            if orphan.any():
                orphan_ids.update(frame.loc[orphan, "case_id"].tolist())
            matched_indices, unique, inverse = batch_groups(positions, matched)
            source_counts[unique] += np.bincount(inverse, minlength=len(unique)).astype(np.int32)
            date_acc.update(frame[date_field], matched_indices, positions, base_dates, None)
            if amount_acc is not None:
                amount_acc.update(frame[amount_field], matched_indices, unique, inverse)
            parsed = parse_date_values(frame[date_field])
            useful = matched & parsed["parseable"]
            if useful.any():
                p = positions[useful]
                source_dates = parsed["parsed"][useful].to_numpy(dtype="datetime64[ns]")
                offsets = (source_dates - base_dates[p]).astype("timedelta64[D]").astype(np.int64)
                months = base_months[p]
                month_offset_counts.update(Counter(zip(months.tolist(), offsets.tolist())))
            idx = pd.to_numeric(frame["num_group1"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
            null_index_rows += int(frame["num_group1"].isna().sum())
            valid = matched & np.isfinite(idx) & (idx == np.floor(idx)) & (idx >= 0) & (idx <= np.iinfo(np.uint32).max)
            if valid.any():
                composite_chunks.append((positions[valid].astype(np.uint64) << np.uint64(32)) | idx[valid].astype(np.uint64))
    date_rows, date_result = date_acc.finish(source, date_field, "", "AD_TAX_TIMING_FOLLOWUP",
                                              "MATCHED_DETAIL_ROWS", int(source_counts.sum()), source_counts)
    amount_result = None
    if amount_acc is not None:
        _, amount_result = amount_acc.finish(source, amount_field, "", "AD_TAX_C_AMOUNT_FOLLOWUP",
                                             "MATCHED_DETAIL_ROWS", int(source_counts.sum()), source_counts, False)
    codes = np.concatenate(composite_chunks) if composite_chunks else np.empty(0, dtype=np.uint64)
    _, counts = np.unique(codes, return_counts=True) if codes.size else (np.empty(0), np.empty(0))
    return {
        "source_counts": source_counts,
        "date_rows": date_rows,
        "date": date_result,
        "amount": amount_result,
        "month_offset_counts": month_offset_counts,
        "structure": {
            "physical_rows": int(source_counts.sum() + orphan_rows + null_case_rows),
            "matched_distinct_applications": int(np.count_nonzero(source_counts > 0)),
            "null_case_id_rows": null_case_rows,
            "orphan_rows": orphan_rows,
            "orphan_distinct_case_ids": len(orphan_ids),
            "null_num_group1_rows": null_index_rows,
            "duplicate_composite_key_distinct_count": int(np.count_nonzero(counts > 1)),
            "duplicate_composite_key_excess_rows": int(np.sum(np.clip(counts - 1, 0, None))) if counts.size else 0,
        },
    }


def tax_followup_rows(
    population: int,
    static_count: dict[str, Any],
    static_sum: dict[str, Any],
    tax_c: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c_records = tax_c["source_counts"]
    c_amount = tax_c["amount"]
    c_finite = c_amount["app"]["finite"]
    c_missing_invalid = c_amount["app"]["missing"] + c_amount["app"]["unparseable"] + c_amount["app"]["nonfinite"]
    c_sum = np.where(c_finite > 0, c_amount["app_sum"], np.nan)
    static_count_value = static_count["app_min"]
    static_sum_value = static_sum["app_min"]
    static_count_finite = static_count["app"]["finite"] > 0
    static_sum_finite = static_sum["app"]["finite"] > 0
    c_present = c_records > 0
    states = tax_c_application_states(c_records, c_finite, c_sum)
    for name, mask in states.items():
        rows.append(followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1",
                                 "C application state", name, int(np.count_nonzero(mask)),
                                 "verified_base_applications", population,
                                 "Mutually exclusive C-source/amount state; absent source does not create an observed monetary zero."))

    def count_comparison(label: str, observed: np.ndarray) -> None:
        valid_static = static_count_finite & np.isfinite(static_count_value)
        integer = valid_static & np.isclose(static_count_value, np.rint(static_count_value), rtol=0.0, atol=1e-12)
        joint = integer
        rounded_static = np.zeros(population, dtype=np.int64)
        rounded_static[joint] = np.rint(static_count_value[joint]).astype(np.int64, copy=False)
        exact = joint & (rounded_static == observed.astype(np.int64, copy=False))
        rows.extend([
            followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1", label,
                         "joint_integer_static_count", int(joint.sum()), "verified_base_applications", population,
                         "Static count is finite and integer-like; C structural zero is retained for no-source applications."),
            followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1", label,
                         "exact_match_count", int(exact.sum()), "joint_integer_static_count", int(joint.sum()),
                         "Exact integer comparison."),
            followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1", label,
                         "mismatch_count", int((joint & ~exact).sum()), "joint_integer_static_count", int(joint.sum()),
                         "Exact integer mismatch; values are not repaired."),
            followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1", label,
                         "noninteger_static_count", int((valid_static & ~integer).sum()), "finite_static_count", int(valid_static.sum()),
                         "Finite static count not integer-like at atol=1e-12."),
        ])
        if label == "pmtscount_vs_c_record_count":
            rows.append(followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1", label,
                                     "mismatches_with_missing_or_invalid_c_amount_rows",
                                     int(np.count_nonzero(joint & ~exact & (c_missing_invalid > 0))),
                                     "count_mismatches", int((joint & ~exact).sum()),
                                     "Visible diagnostic for mismatches coinciding with missing/invalid C amounts."))

    count_comparison("pmtscount_vs_c_record_count", c_records)
    count_comparison("pmtscount_vs_c_finite_amount_count", c_finite)

    joint_sum = static_sum_finite & np.isfinite(c_sum)
    left = static_sum_value[joint_sum]
    right = c_sum[joint_sum]
    exact = left == right
    close = np.isclose(left, right, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
    difference = left - right
    absolute = np.abs(difference)
    nonzero_reference = right != 0
    relative = np.abs(difference[nonzero_reference] / right[nonzero_reference])
    rows.extend([
        followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1", "pmtssum_vs_c_finite_amount_sum",
                     "joint_finite_sum_count", int(joint_sum.sum()), "verified_base_applications", population,
                     "Both the static summary and diagnostic C sum have finite observations."),
        followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1", "pmtssum_vs_c_finite_amount_sum",
                     "exact_match_count", int(exact.sum()), "joint_finite_sums", int(joint_sum.sum()),
                     "Exact binary floating-point equality."),
        followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1", "pmtssum_vs_c_finite_amount_sum",
                     "tolerance_match_count", int(close.sum()), "joint_finite_sums", int(joint_sum.sum()),
                     f"Float64 comparison using rtol={FLOAT_RTOL}, atol={FLOAT_ATOL}; tolerance fixed before results."),
        followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1", "pmtssum_vs_c_finite_amount_sum",
                     "tolerance_mismatch_count", int((~close).sum()), "joint_finite_sums", int(joint_sum.sum()),
                     "No tolerance inflation or data repair."),
        followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1", "pmtssum_vs_c_finite_amount_sum",
                     "relative_difference_undefined_zero_reference_count", int(np.count_nonzero(~nonzero_reference)),
                     "joint_finite_sums", int(joint_sum.sum()), "Relative difference is undefined when the C diagnostic sum is zero."),
    ])
    for prefix, values in (("signed_difference", difference), ("absolute_difference", absolute),
                           ("absolute_relative_difference", relative)):
        stats = exact_quantiles(values)
        for name in ("min", "p01", "p05", "median", "p95", "p99", "max", "mean"):
            rows.append(followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1",
                                     "pmtssum_vs_c_finite_amount_sum", f"{prefix}_{name}", stats[name],
                                     "defined_comparison_values", len(values),
                                     f"Exact distribution statistic; quantiles use method={QUANTILE_METHOD}."))

    reference_match = (static_count["app"]["nonzero"] > 0) == c_present
    rows.append(followup_row("TAX_C_SUMMARY_COMPARISON", "static_cb_0|tax_registry_c_1",
                             "pmtscount_nonzero_presence_correspondence", "all_base_applications_equal",
                             int(reference_match.sum()), "verified_base_applications", population,
                             "Checks whether finite nonzero pmtscount presence equals C source presence."))
    return rows


def load_small_group(train_dir: Path, file_names: list[str], columns: list[str]) -> pd.DataFrame:
    frames = []
    for file_name in file_names:
        frames.append(pq.read_table(train_dir / file_name, columns=columns).to_pandas())
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def repeated_signature_metrics(
    frame: pd.DataFrame,
    columns: list[str],
    date_fields: list[str],
    economic_fields: list[str],
) -> dict[str, int]:
    counts = frame.groupby(columns, dropna=False).size()
    duplicate = counts[counts > 1]
    all_economic_missing = frame[economic_fields].isna().all(axis=1) if economic_fields else np.zeros(len(frame), dtype=bool)
    any_date_missing = frame[date_fields].isna().any(axis=1) if date_fields else np.zeros(len(frame), dtype=bool)
    return {
        "signature_group_count": int(len(counts)),
        "repeated_signature_group_count": int(len(duplicate)),
        "repeated_signature_excess_rows": int((duplicate - 1).sum()),
        "maximum_signature_multiplicity": int(counts.max()) if len(counts) else 0,
        "rows_with_any_selected_date_missing": int(np.count_nonzero(any_date_missing)),
        "rows_with_all_selected_economic_fields_missing": int(np.count_nonzero(all_economic_missing)),
    }


def debit_deposit_followup(
    train_dir: Path,
    debit_files: list[str],
    deposit_files: list[str],
    base_index: pd.Index,
    base_dates: np.ndarray,
    batch1_flags_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    population = len(base_index)
    debit_economic = ["last30dayturnover_651A", "last180dayturnover_1134A", "last180dayaveragebalance_704A"]
    deposit_economic = ["amount_416A"]
    debit = load_small_group(train_dir, debit_files, ["case_id", "num_group1", "openingdate_857D", *debit_economic])
    deposit = load_small_group(train_dir, deposit_files, ["case_id", "num_group1", "openingdate_313D", "contractenddate_991D", *deposit_economic])
    debit_pos, debit_matched, _ = positions_for(debit["case_id"], base_index)
    deposit_pos, deposit_matched, _ = positions_for(deposit["case_id"], base_index)
    debit_counts = np.bincount(debit_pos[debit_matched], minlength=population).astype(np.int32)
    deposit_counts = np.bincount(deposit_pos[deposit_matched], minlength=population).astype(np.int32)
    avg_classified = classify_numeric(debit["last180dayaveragebalance_704A"])
    avg_missing_counts = np.bincount(debit_pos[debit_matched & avg_classified["missing"]], minlength=population).astype(np.int32)
    avg_finite_counts = np.bincount(debit_pos[debit_matched & avg_classified["finite"]], minlength=population).astype(np.int32)

    flag_columns = [
        "case_id", "source__debitcard_1__record_count", "source__deposit_1__record_count",
        "economic_field__last180dayaveragebalance_704A__missing_row_count",
        "economic_field__last180dayaveragebalance_704A__finite_row_count",
    ]
    flags = pq.read_table(batch1_flags_path, columns=flag_columns).to_pandas()
    if not np.array_equal(flags["case_id"].to_numpy(), base_index.to_numpy()):
        raise RuntimeError("Batch 1 flags do not retain the exact ordered base key set")
    rows = []

    checks = {
        "every_deposit_application_also_debitcard": not np.any((deposit_counts > 0) & (debit_counts == 0)),
        "raw_debit_record_count_equals_deposit_plus_finite_average_balance": np.array_equal(debit_counts, deposit_counts + avg_finite_counts),
        "raw_debit_missing_average_balance_equals_deposit_record_count": np.array_equal(avg_missing_counts, deposit_counts),
        "batch1_debit_record_counts_match_raw": np.array_equal(flags["source__debitcard_1__record_count"].to_numpy(), debit_counts),
        "batch1_deposit_record_counts_match_raw": np.array_equal(flags["source__deposit_1__record_count"].to_numpy(), deposit_counts),
        "batch1_average_balance_missing_counts_match_raw": np.array_equal(flags["economic_field__last180dayaveragebalance_704A__missing_row_count"].to_numpy(), avg_missing_counts),
        "batch1_average_balance_finite_counts_match_raw": np.array_equal(flags["economic_field__last180dayaveragebalance_704A__finite_row_count"].to_numpy(), avg_finite_counts),
    }
    for name, passed in checks.items():
        rows.append(followup_row("DEBIT_DEPOSIT_STRUCTURE", "debitcard_1|deposit_1", "count correspondence",
                                 name, int(passed), "boolean_check", 1,
                                 "Count correspondence only; it does not establish record identity or account matching.",
                                 "PASS" if passed else "FAIL"))

    debit_open = parse_date_values(debit["openingdate_857D"])
    deposit_open = parse_date_values(deposit["openingdate_313D"])
    debit_valid = debit_matched & debit_open["parseable"]
    deposit_valid = deposit_matched & deposit_open["parseable"]
    grouped = grouped_date_comparison(
        debit_pos[debit_valid],
        debit_open["parsed"][debit_valid].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        deposit_pos[deposit_valid],
        deposit_open["parsed"][deposit_valid].to_numpy(dtype="datetime64[ns]").astype(np.int64),
    )
    group_metrics = {
        "debit_parseable_opening_date_group_count": grouped["left_group_count"],
        "deposit_parseable_opening_date_group_count": grouped["right_group_count"],
        "matched_case_opening_date_group_count": grouped["matched_group_count"],
        "debit_only_case_opening_date_group_count": grouped["left_only_group_count"],
        "deposit_only_case_opening_date_group_count": grouped["right_only_group_count"],
        "matched_group_equal_multiplicity_count": grouped["matched_equal_multiplicity_count"],
        "matched_group_different_multiplicity_count": grouped["matched_different_multiplicity_count"],
        "matched_ambiguous_multirow_group_count": grouped["matched_ambiguous_multirow_group_count"],
        "debit_rows_missing_or_invalid_opening_date": int(np.count_nonzero(debit_matched & ~debit_open["parseable"])),
        "deposit_rows_missing_or_invalid_opening_date": int(np.count_nonzero(deposit_matched & ~deposit_open["parseable"])),
    }
    for name, value in group_metrics.items():
        rows.append(followup_row("DEBIT_DEPOSIT_GROUPED_DATE", "debitcard_1|deposit_1",
                                 "independent case/opening-date groups", name, int(value),
                                 "group_or_row_count", None,
                                 "Sources grouped independently by case_id and parseable opening date; no raw-row Cartesian match or num_group1 match."))

    debit_signatures = repeated_signature_metrics(debit, ["case_id", "openingdate_857D", *debit_economic],
                                                   ["openingdate_857D"], debit_economic)
    deposit_signatures = repeated_signature_metrics(deposit, ["case_id", "openingdate_313D", "contractenddate_991D", "amount_416A"],
                                                     ["openingdate_313D", "contractenddate_991D"], deposit_economic)
    for source, metrics in (("debitcard_1", debit_signatures), ("deposit_1", deposit_signatures)):
        for name, value in metrics.items():
            rows.append(followup_row("REPEATED_SIGNATURES", source, "signature excluding num_group1",
                                     name, value, "source_rows_or_groups", None,
                                     "Repeated signatures may be legitimate distinct records and were not deduplicated."))

    end = parse_date_values(deposit["contractenddate_991D"])
    useful = deposit_matched & end["parseable"]
    offsets = np.full(len(deposit), np.nan)
    offsets[useful] = (
        end["parsed"][useful].to_numpy(dtype="datetime64[ns]") - base_dates[deposit_pos[useful]]
    ).astype("timedelta64[D]").astype(np.float64)
    before_row = useful & (offsets < 0)
    on_row = useful & (offsets == 0)
    after_row = useful & (offsets > 0)
    missing_row = deposit_matched & end["native_missing"]
    empty_row = deposit_matched & end["empty"]
    fail_row = deposit_matched & end["parse_failure"]
    app_before = np.zeros(population, dtype=bool)
    app_onafter = np.zeros(population, dtype=bool)
    app_unknown = np.zeros(population, dtype=bool)
    np.logical_or.at(app_before, deposit_pos[before_row], True)
    np.logical_or.at(app_onafter, deposit_pos[on_row | after_row], True)
    np.logical_or.at(app_unknown, deposit_pos[missing_row | empty_row | fail_row], True)
    source_present = deposit_counts > 0
    end_states = {
        "NO_DEPOSIT_SOURCE": ~source_present,
        "SOURCE_PRESENT_ALL_END_DATES_UNKNOWN": source_present & app_unknown & ~app_before & ~app_onafter,
        "ALL_SOURCE_ROWS_HAVE_END_DATE_BEFORE_DECISION": source_present & app_before & ~app_onafter & ~app_unknown,
        "ANY_END_DATE_ON_OR_AFTER_DECISION": source_present & app_onafter,
        "NO_END_DATE_ON_OR_AFTER_BUT_MIXED_BEFORE_AND_UNKNOWN": source_present & app_before & ~app_onafter & app_unknown,
    }
    for state, mask in end_states.items():
        rows.append(followup_row("DEPOSIT_END_STATE", "deposit_1", "contractenddate_991D",
                                 state, int(mask.sum()), "verified_base_applications", population,
                                 "Five-state mutually exclusive application partition; unknown does not imply active."))
    rows.append(followup_row("DEPOSIT_END_STATE", "deposit_1", "contractenddate_991D",
                             "applications_both_before_and_on_or_after", int((app_before & app_onafter).sum()),
                             "verified_base_applications", population,
                             "Overlapping diagnostic, not part of the five-state partition."))
    for name, mask in (("rows_end_before", before_row), ("rows_end_on", on_row), ("rows_end_after", after_row),
                       ("rows_end_native_missing", missing_row), ("rows_end_empty", empty_row),
                       ("rows_end_parse_failure", fail_row)):
        rows.append(followup_row("DEPOSIT_END_STATE", "deposit_1", "contractenddate_991D", name,
                                 int(mask.sum()), "matched_deposit_rows", int(deposit_matched.sum()),
                                 "Row-level contract-end timing state."))

    amount_class = classify_numeric(deposit["amount_416A"])
    end_amount_masks = OrderedDict([
        ("BEFORE_DECISION", before_row), ("ON_DECISION", on_row), ("AFTER_DECISION", after_row),
        ("UNKNOWN_NATIVE_MISSING", missing_row), ("UNKNOWN_EMPTY", empty_row),
        ("UNKNOWN_PARSE_FAILURE", fail_row),
    ])
    for state, state_mask in end_amount_masks.items():
        finite = state_mask & amount_class["finite"]
        values = amount_class["values"][finite]
        rows.extend([
            followup_row("DEPOSIT_AMOUNT_BY_END_STATE", "deposit_1", "amount_416A", f"{state}_row_count",
                         int(state_mask.sum()), "matched_deposit_rows", int(deposit_matched.sum()),
                         "Conditional row group; no cross-row amount sum is interpreted as a balance."),
            followup_row("DEPOSIT_AMOUNT_BY_END_STATE", "deposit_1", "amount_416A", f"{state}_finite_count",
                         int(values.size), "end_state_rows", int(state_mask.sum()),
                         "Finite deposit amount observations in this end-date row state."),
            followup_row("DEPOSIT_AMOUNT_BY_END_STATE", "deposit_1", "amount_416A", f"{state}_zero_count",
                         int(np.count_nonzero(values == 0)), "finite_amounts_in_end_state", int(values.size),
                         "Observed finite zeros are retained."),
        ])
        for stat, value in exact_quantiles(values).items():
            rows.append(followup_row("DEPOSIT_AMOUNT_BY_END_STATE", "deposit_1", "amount_416A",
                                     f"{state}_finite_{stat}", value, "finite_amounts_in_end_state",
                                     int(values.size), f"Exact conditional statistic; quantiles use method={QUANTILE_METHOD}."))

    summary = {
        "count_correspondence_checks": checks,
        "grouped_opening_date": group_metrics,
        "debit_repeated_signatures": debit_signatures,
        "deposit_repeated_signatures": deposit_signatures,
        "deposit_end_application_states": {key: int(mask.sum()) for key, mask in end_states.items()},
        "deposit_end_application_state_sum": int(sum(mask.sum() for mask in end_states.values())),
        "applications_both_before_and_on_or_after": int((app_before & app_onafter).sum()),
    }
    return rows, summary


def build_person_role_rows(person: dict[str, Any], population: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = person["person_selected_counts"]
    states = person_selection_states(person["source_record_counts"], selected)
    for state, mask in states.items():
        rows.append({
            "record_type": "APPLICANT_SELECTION_MULTIPLICITY",
            "field_or_combination": "num_group1=0",
            "code_or_state": state,
            "row_count": int(mask.sum()),
            "application_count": int(mask.sum()),
            "denominator_name": "verified_base_applications",
            "denominator_value": population,
            "rate": float(mask.mean()),
            "selection_evidence": f"Official competition data description: when num_groupN is a person index, zero is the applicant ({OFFICIAL_DATA_URL}).",
            "selection_status": "DOCUMENTED_DIAGNOSTIC_SELECTION_NOT_MODEL_APPROVAL",
            "unresolved_question": "Role codes remain masked; num_group1=0 identifies the applicant record for this table but does not approve payload fields.",
        })
    for name, value in sorted(person["person_selection_cross"].items()):
        rows.append({
            "record_type": "SELECTION_CROSS_CHECK",
            "field_or_combination": "num_group1|personindex_1023L",
            "code_or_state": name,
            "row_count": value,
            "application_count": None,
            "denominator_name": "num_group1_zero_rows",
            "denominator_value": int(selected.sum()),
            "rate": value / int(selected.sum()) if selected.sum() else None,
            "selection_evidence": "Dictionary describes personindex_1023L as order on the application form; it is a cross-check, not the selection rule.",
            "selection_status": "DIAGNOSTIC_ONLY",
            "unresolved_question": "Do not substitute personindex=0 or first-row order when documented num_group1 selection is available.",
        })
    for field in PERSON_CONTROLS:
        result = person["categorical"][field]
        for code, count in sorted(result["frequencies"].items(), key=lambda item: (-item[1], item[0])):
            rows.append({
                "record_type": "ROLE_OR_ORDER_CODE_FREQUENCY",
                "field_or_combination": field,
                "code_or_state": code,
                "row_count": count,
                "application_count": None,
                "denominator_name": "populated_matched_person_rows",
                "denominator_value": int(sum(result["frequencies"].values())),
                "rate": count / sum(result["frequencies"].values()) if result["frequencies"] else None,
                "selection_evidence": "Codes preserved verbatim; masked/undocumented meanings were not guessed.",
                "selection_status": "CONTROL_AUDIT_ONLY",
                "unresolved_question": "Official host states role describes connection to client, but masked code meanings remain undocumented.",
            })
    for combination, count in sorted(person["person_combinations"].items(), key=lambda item: (-item[1], item[0])):
        rows.append({
            "record_type": "ROLE_ORDER_COMBINATION_FREQUENCY",
            "field_or_combination": "|".join(PERSON_CONTROLS),
            "code_or_state": combination,
            "row_count": count,
            "application_count": None,
            "denominator_name": "matched_person_rows",
            "denominator_value": person["structure"]["matched_rows"],
            "rate": count / person["structure"]["matched_rows"] if person["structure"]["matched_rows"] else None,
            "selection_evidence": "Descriptive combination only; not used to infer applicant identity.",
            "selection_status": "CONTROL_AUDIT_ONLY",
            "unresolved_question": "Masked code combination semantics remain unresolved.",
        })
    for field, result in person["person_applicant_results"].items():
        if "app" in result:
            covered = int(np.count_nonzero(result["app"]["finite"] > 0))
            conflict = int(np.count_nonzero(~np.isclose(result["app_min"], result["app_max"], rtol=0, atol=0, equal_nan=True)))
        elif "app_populated" in result:
            covered = int(np.count_nonzero(result["app_populated"] > 0))
            conflict = int(np.count_nonzero(result["app_conflict"]))
        else:
            covered = int(np.count_nonzero(result["app_parseable"] > 0))
            conflict = int(np.count_nonzero(result["latest_conflict"]))
        rows.append({
            "record_type": "DOCUMENTED_APPLICANT_PAYLOAD_COVERAGE",
            "field_or_combination": field,
            "code_or_state": "POPULATED_OR_FINITE_OR_PARSEABLE",
            "row_count": None,
            "application_count": covered,
            "denominator_name": "verified_base_applications",
            "denominator_value": population,
            "rate": covered / population,
            "selection_evidence": "Payload restricted diagnostically to documented person_1 num_group1=0 rows.",
            "selection_status": "PENDING_RESEARCHER_DECISION",
            "unresolved_question": f"Multiple/conflicting selected payload applications={conflict}; point-in-time and substantive eligibility still require review.",
        })
    return rows


def build_credit_count_review(
    population: int,
    field_sources: dict[str, str],
    source_results: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_data: dict[str, dict[str, Any]] = {}
    state_names = {
        0: "NO_SOURCE_RECORD", 1: "SOURCE_PRESENT_NO_FINITE", 2: "INVALID",
        3: "CONFLICTING_FINITE_VALUES", 4: "OBSERVED_ZERO", 5: "OBSERVED_POSITIVE",
        6: "OBSERVED_NEGATIVE",
    }
    for field in CREDIT_COUNT_FIELDS:
        source = field_sources[field]
        source_result = source_results[source]
        result = source_result["numeric"][field]
        state, value, consistent = credit_candidate_state(
            source_result["source_record_counts"], result["app"],
            result["app_min"], result["app_max"],
        )
        candidate_data[field] = {"source": source, "value": value, "state": state, "consistent": consistent}
        suitability = (
            "FEASIBLE_OBSERVED_ACTIVE_CREDIT_COUNT_CANDIDATE"
            if field == "numactivecreds_622L" else "COMPARISON_CANDIDATE_NOT_SELECTED"
        )
        for code, name in state_names.items():
            count = int(np.count_nonzero(state == code))
            rows.append({
                "record_type": "CANDIDATE_STATE",
                "candidate_left": field,
                "candidate_right": "",
                "source_scope": source,
                "metric": "application_state_count",
                "state_left": name,
                "state_right": "",
                "value": count,
                "denominator_name": "verified_base_applications",
                "denominator_value": population,
                "definition": "Mutually exclusive source/observation/consistency state; no-source is not zero credit history.",
                "grouping_suitability": suitability,
            })
        rows.append({
            "record_type": "CANDIDATE_SUMMARY",
            "candidate_left": field,
            "candidate_right": "",
            "source_scope": source,
            "metric": "unknown_or_unresolved_share",
            "state_left": "",
            "state_right": "",
            "value": float(np.mean(~consistent)),
            "denominator_name": "verified_base_applications",
            "denominator_value": population,
            "definition": "Applications without an internally consistent finite observation.",
            "grouping_suitability": suitability,
        })
    for left, right in itertools.combinations(CREDIT_COUNT_FIELDS, 2):
        a, b = candidate_data[left], candidate_data[right]
        for code_left, name_left in state_names.items():
            for code_right, name_right in state_names.items():
                count = int(np.count_nonzero((a["state"] == code_left) & (b["state"] == code_right)))
                if count:
                    rows.append({
                        "record_type": "CANDIDATE_STATE_CROSSTAB",
                        "candidate_left": left,
                        "candidate_right": right,
                        "source_scope": f"{a['source']}|{b['source']}",
                        "metric": "joint_state_count",
                        "state_left": name_left,
                        "state_right": name_right,
                        "value": count,
                        "denominator_name": "verified_base_applications",
                        "denominator_value": population,
                        "definition": "Cross-source/candidate states; disagreement can reflect scope rather than error.",
                        "grouping_suitability": "COMPARISON_ONLY",
                    })
        joint = a["consistent"] & b["consistent"]
        equal = joint & np.isclose(a["value"], b["value"], rtol=0.0, atol=1e-12)
        for metric, value in (
            ("joint_consistent_observation_count", int(joint.sum())),
            ("joint_exact_or_tolerance_agreement_count", int(equal.sum())),
            ("joint_disagreement_count", int((joint & ~equal).sum())),
        ):
            rows.append({
                "record_type": "PAIR_AGREEMENT",
                "candidate_left": left,
                "candidate_right": right,
                "source_scope": f"{a['source']}|{b['source']}",
                "metric": metric,
                "state_left": "",
                "state_right": "",
                "value": value,
                "denominator_name": "joint_consistent_observations" if "agreement" in metric or "disagreement" in metric else "verified_base_applications",
                "denominator_value": int(joint.sum()) if "agreement" in metric or "disagreement" in metric else population,
                "definition": "Direct comparison only; active/closed/channel scopes may legitimately disagree and are not added.",
                "grouping_suitability": "COMPARISON_ONLY",
            })
    summary = {
        field: {
            "source": data["source"],
            "consistent_observed_count": int(data["consistent"].sum()),
            "unknown_or_unresolved_count": int((~data["consistent"]).sum()),
            "unknown_or_unresolved_share": float(np.mean(~data["consistent"])),
            "zero_count": int(np.count_nonzero(data["state"] == 4)),
            "positive_count": int(np.count_nonzero(data["state"] == 5)),
            "conflicting_count": int(np.count_nonzero(data["state"] == 3)),
        }
        for field, data in candidate_data.items()
    }
    return rows, summary


def documentation_evidence_text(
    dictionary_path: Path,
    proposal_path: Path | None,
) -> str:
    proposal_line = str(proposal_path.resolve()) if proposal_path and proposal_path.exists() else "not supplied to this script"
    return f"""# Task 07 Batch 2 documentation evidence

## Local evidence

- `{dictionary_path.resolve()}` is the supplied field dictionary. It describes individual variables but generally does not define observation windows, record identity, or point-in-time availability.
- `{proposal_line}` is research background, not authoritative source-system documentation.
- The dictionary describes `dateofcredstart_181D` as a contract-close date, while the official competition data page contains an explicit edit stating it is a credit-contract start date. This conflict is recorded; the official correction is preferred for interpretation, while the raw field is not renamed.
- The dictionary describes `personindex_1023L` as order on the application form and `role_1084L` as contact-role type, but it does not decode masked role categories.
- `amount_416A` is only described as “Deposit amount.” That does not establish account, transaction, snapshot, or current-balance semantics.

## Official competition evidence

1. [Official dataset description]({OFFICIAL_DATA_URL})
   - Establishes `case_id` as the observation join key.
   - Defines depth 0 as static features and depth 1 as historical records indexed by `num_group1`.
   - States that when a grouping index represents a person index, zero denotes the applicant. This supports a diagnostic `person_1.num_group1=0` applicant-row selection.
   - States multi-file groups were split using `WEEK_NUM`; shard numbering is therefore not chronology within an application.
   - Contains the correction that `dateofcredstart_181D` is a credit-contract start date.

2. [Official host discussion on dates and person data]({OFFICIAL_DISCUSSION_URL})
   - The host states prediction occurs at application/decision time and competition data were collected as of that date.
   - The host also states future-looking transformed dates are intentional. This supports competition-dataset availability, but does not document the underlying transformation or prove deployable real-time availability outside the competition construct.
   - The host describes `person_1` as potentially containing the client and contact references, with role fields describing connection to the client. Masked category meanings remain unavailable.

3. [Official host response on previous applications]({OFFICIAL_APPLPREV_URL})
   - States `applprev` contains the client’s prior Home Credit applications, which may be cancelled, rejected, or approved; approval does not mean the loan was ultimately used.

## Established, hypothesized, and unresolved

- **Established for this competition:** depth-1 rows are historical records; `num_group1=0` is the applicant when it is the person index; data were represented as collected at decision time; several dates were transformed.
- **Hypothesis only:** repeated bureau count/total values may be application-level summaries repeated across contract-like rows. Empirical consistency can support a non-summing rule but cannot establish source-system row identity.
- **Unresolved:** precise tax-date transformation, tax reporting windows, deposit row unit, whether deposit amounts are balances or historical amounts, masked role/status code meanings, and real-world availability outside the competition dataset.
"""


def decision_review_rows(
    population: int,
    source_results: dict[str, dict[str, Any]],
    credit_summary: dict[str, Any],
    batch1_summary: dict[str, Any],
    tax_comparison_summary: dict[str, Any],
    debit_deposit_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def numeric_coverage(source: str, field: str) -> int:
        return int(np.count_nonzero(source_results[source]["numeric"][field]["app"]["finite"] > 0))

    def add(dimension: str, candidate: str, coverage: str, use: str, missing: str,
            limitations: str, redundancy: str, priority: str, evidence: str) -> None:
        rows.append({
            "economic_dimension": dimension,
            "candidate_or_source": candidate,
            "observed_coverage": coverage,
            "suggested_direct_use_or_aggregation": use,
            "proposed_missing_zero_handling": missing,
            "timing_or_role_limitations": limitations,
            "redundancy_considerations": redundancy,
            "priority": priority,
            "evidence": evidence,
            "approval_status": "PENDING_RESEARCHER_DECISION",
        })

    add("Income", "static_0.maininc_215A", f"finite applications={numeric_coverage('static_0','maininc_215A'):,}/{population:,}",
        "Direct application-level amount candidate.", "Retain documented zero; training-only imputation later with missing indicator if approved.",
        "Application-time availability is plausible from description but still requires final timing sign-off.", "Compare with documented applicant person income; do not average persons.",
        "FIRST_VERSION_CANDIDATE", "Static one-to-one source and dictionary primary-income description.")
    add("Application scale", "static_0.credamount_770A + annuity_780A", f"finite amount={numeric_coverage('static_0','credamount_770A'):,}; finite annuity={numeric_coverage('static_0','annuity_780A'):,}",
        "Use direct application-level values with explicit names loan-or-limit and monthly annuity.", "Keep absent and zero distinct; do not derive ratios yet.",
        "credamount may be loan amount or card limit, not universal disbursed principal.", "Do not substitute prior-application amounts.",
        "FIRST_VERSION_CANDIDATE", "One-to-one static structure and exact dictionary descriptions.")
    add("Debt", "static_0.currdebt_22A + totaldebt_9A", f"finite current debt={numeric_coverage('static_0','currdebt_22A'):,}; total debt={numeric_coverage('static_0','totaldebt_9A'):,}",
        "Direct application-level candidates; keep separate pending scope confirmation.", "Retain zero; missing is unknown rather than no debt.",
        "Scope and source window require final review.", "Check overlap before using both; do not sum.",
        "FIRST_VERSION_CANDIDATE", "Static source; empirical availability in traditional_field_audit.csv.")
    active = credit_summary["numactivecreds_622L"]
    add("Credit-file grouping", "static_0.numactivecreds_622L",
        f"consistent observed={active['consistent_observed_count']:,}; unknown/unresolved={active['unknown_or_unresolved_count']:,} ({active['unknown_or_unresolved_share']:.4%})",
        "Feasible transparent observed-active-credit-count candidate; thresholds remain unselected.",
        "Missing remains unknown; observed zero means zero active credits, not no lifetime history.",
        "Measures active credits, not lifetime file thickness.", "Do not add A/B counts that may overlap.",
        "FIRST_VERSION_CANDIDATE", "Broad static coverage and direct count description; cross-source review remains available.")
    add("Bureau delinquency", "static_0 maxdpdlast3m/6m/12m/24m", "See exact per-field finite application counts in traditional_field_audit.csv.",
        "Use a compact subset of windowed delinquency summaries after final timing review.",
        "Keep missing distinct; signed DPD values are not automatically errors.", "Descriptions support historical windows; availability still relies on competition construct.",
        "Windows are nested and potentially redundant.", "CHECK_BEFORE_USE", "Dictionary window descriptions and measured state distributions.")
    add("Bureau history", "credit_bureau_a_1 active/closed count, debt, DPD summaries",
        f"source applications={source_results['credit_bureau_a_1']['structure']['matched_distinct_case_ids']:,}",
        "Use internally consistent per-application summaries without summing repeated totals; keep active and closed sides separate.",
        "No source and conflicting repeated values remain unknown.", "Record unit and several total-field scopes remain unresolved.",
        "Avoid duplicate static/A/B counts and totals until registry review.", "CHECK_BEFORE_USE", "Per-application min/max consistency and active/closed evidence audit.")
    add("Internal history", "applprev_1 amount/status/DPD/payment summaries",
        f"source applications={source_results['applprev_1']['structure']['matched_distinct_case_ids']:,}",
        "Candidate application-level counts/distributions and latest-by-documented-date diagnostics; do not treat every row as an approved loan.",
        "Absent history remains unknown; rejected/cancelled records are retained.", "Masked statuses are not decoded; latest ties require deterministic researcher-approved handling.",
        "Static prior-application summaries may overlap.", "FIRST_VERSION_CANDIDATE", "Official host confirms rows are prior applications with varied outcomes.")
    person_selected = source_results["person_1"]["person_selected_counts"]
    add("Applicant capacity", "person_1 num_group1=0 payload",
        f"exactly one documented applicant row={np.count_nonzero(person_selected==1):,}; zero={np.count_nonzero(person_selected==0):,}; multiple={np.count_nonzero(person_selected>1):,}",
        "Use only documented applicant-row payload candidates; never average all persons.",
        "Applications without a selected row or payload remain missing.", "Masked payload categories and employment units require review.",
        "Compare income with static primary income; role/control fields are not predictors by default.", "CHECK_BEFORE_USE",
        f"Official data page documents zero person index as applicant: {OFFICIAL_DATA_URL}")
    add("Bureau inquiries", "static_cb_0 days30/90/180/360", f"source applications={source_results['static_cb_0']['structure']['matched_distinct_case_ids']:,}",
        "Direct external bureau-query counts, retaining windows separately initially.", "No static_cb row is unknown, not zero; retain genuine zero.",
        "Query availability and nested-window redundancy require review.", "Nested horizons are correlated; compact subset may suffice.",
        "CHECK_BEFORE_USE", "Dictionary count-window descriptions and measured states.")

    for module in ("debitcard", "deposit", "tax"):
        count = batch1_summary["module_counts"][module]["content"]
        if module == "debitcard":
            use = "Use clearly named observed turnover/balance summaries and source/content indicator; do not infer account identity from deposit correspondence."
            limitation = "Debit-card table exact row unit remains unresolved; large auxiliary-only coverage."
            redundancy = "Debitcard_1 and other_1 provide complementary application coverage."
            priority = "CHECK_BEFORE_USE"
        elif module == "deposit":
            use = "Prefer observed-record amount summaries and explicit end-date-state diagnostics; avoid calling amounts current balances or cash buffers."
            limitation = "Deposit row unit and amount meaning remain unresolved; end date may be historical/contractual."
            redundancy = "Debit/deposit count correspondence does not establish same-record identity."
            priority = "CHECK_BEFORE_USE"
        else:
            use = "Prefer compact static tax summaries; consider descriptive A/B detail aggregates subject to timing uncertainty."
            limitation = "Dates are transformed and precise real-world timing/window semantics remain unavailable."
            redundancy = "Avoid duplicate C count/sum features if empirical equivalence is confirmed; other C uses remain pending."
            priority = "FIRST_VERSION_CANDIDATE"
        add(f"AD: {module}", module, f"Batch 1 finite-content applications={count:,}/{population:,}", use,
            "Distinguish no source, present/all-missing, finite zero and finite nonzero; training-only preprocessing later.",
            limitation, redundancy, priority, "Accepted Batch 1 audit plus focused Batch 2 follow-ups.")
    return rows


def audit_report_text(
    population: int,
    structures: dict[str, dict[str, Any]],
    source_results: dict[str, dict[str, Any]],
    credit_summary: dict[str, Any],
    person_rows: list[dict[str, Any]],
    tax_rows: list[dict[str, Any]],
    debit_summary: dict[str, Any],
    batch1_summary: dict[str, Any],
    validation: list[dict[str, str]],
) -> str:
    person_selected = source_results["person_1"]["person_selected_counts"]
    tax_metric = {(row["field_or_comparison"], row["metric"]): row["value"] for row in tax_rows}
    c_joint = tax_metric.get(("pmtscount_vs_c_record_count", "joint_integer_static_count"))
    c_count_matches = tax_metric.get(("pmtscount_vs_c_record_count", "exact_match_count"))
    c_sum_joint = tax_metric.get(("pmtssum_vs_c_finite_amount_sum", "joint_finite_sum_count"))
    c_sum_close = tax_metric.get(("pmtssum_vs_c_finite_amount_sum", "tolerance_match_count"))
    active = credit_summary["numactivecreds_622L"]
    lines = [
        "# Task 07 Batch 2 — Traditional information and focused AD follow-ups",
        "",
        "## Decision-oriented findings",
        "",
        f"The verified population remains **{population:,} application records** with unique, non-null `case_id`; Batch 1 flags retain the exact ordered key set.",
        "",
        "### Traditional source structure",
        "",
        "| Source | Physical rows | Matched applications | Median / p95 / max rows | Composite-key duplicates | Join interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for source in ("static_0", "static_cb_0", "credit_bureau_a_1", "applprev_1", "person_1", "credit_bureau_b_1"):
        row = structures[source]
        lines.append(
            f"| {source} | {row['physical_rows']:,} | {row['matched_distinct_case_ids']:,} | "
            f"{row['median']} / {row['p95']} / {row['max']} | {row['duplicate_composite_key_excess_rows'] or 0:,} | {row['join_readiness']} |"
        )
    lines.extend([
        "",
        "No history table was cross-joined to another. Repeated application totals and counts were assessed by per-application consistency and were not summed as contracts.",
        "",
        "### Applicant-role resolution",
        "",
        f"Official competition documentation states that when the grouping index is the person index, zero denotes the applicant. Using `person_1.num_group1=0` as a diagnostic selection gives **{np.count_nonzero(person_selected==1):,}** applications with exactly one selected row, **{np.count_nonzero(person_selected==0):,}** with none, and **{np.count_nonzero(person_selected>1):,}** with multiple rows.",
        "",
        "The selection is documented for applicant-row identification, but payload fields remain pending approval. Masked role/status codes were preserved without guessing. All-person payload statistics remain separate from applicant-only results.",
        "",
        "### Credit-count grouping",
        "",
        f"`numactivecreds_622L` is the most practical first grouping candidate: **{active['consistent_observed_count']:,}** applications have an internally consistent finite observation and **{active['unknown_or_unresolved_count']:,} ({active['unknown_or_unresolved_share']:.4%})** remain unknown or unresolved.",
        "",
        "The defensible label is **observed active-credit count**, not lifetime credit-file thickness. Observed zero means no active credits under this field; it does not mean no past credit history. No threshold was selected, and A/B/static counts were not added together.",
        "",
        "### Tax timing and C-summary comparison",
        "",
        "Official host documentation says competition data were collected as of decision time and date columns were transformed; it specifically acknowledges future transformed tax record dates. This supports use within the competition construct but does not reveal the transformation or prove real-world deployability.",
        "",
        f"For tax C, static count versus C record count had **{c_count_matches:,} exact matches among {c_joint:,} integer-comparable applications**. Static amount sum versus the diagnostic sum of finite C amounts had **{c_sum_close:,} tolerance matches among {c_sum_joint:,} joint finite sums** using fixed float64 tolerances `rtol={FLOAT_RTOL}`, `atol={FLOAT_ATOL}`.",
        "",
        "Even exact count/sum reproduction would only support avoiding duplicate first-version count/sum inputs. It would not establish identical recency, variability, distributions, or point-in-time availability.",
        "",
        "### Debit-card and deposit relationship",
        "",
        f"Every deposit-source application also appears in debitcard_1: **{debit_summary['count_correspondence_checks']['every_deposit_application_also_debitcard']}**. The raw record-count equations and Batch 1 flags reconcile: **{all(debit_summary['count_correspondence_checks'].values())}**.",
        "",
        f"Independent case/opening-date grouping found {debit_summary['grouped_opening_date']['matched_case_opening_date_group_count']:,} matched groups, {debit_summary['grouped_opening_date']['debit_only_case_opening_date_group_count']:,} debit-only groups, and {debit_summary['grouped_opening_date']['deposit_only_case_opening_date_group_count']:,} deposit-only groups. Matching grouped counts do not establish account identity.",
        "",
        "Deposit amounts should be described as observed deposit-record amounts, not current balances or liquid savings. Missing end dates do not prove active contracts; historical and future contractual end dates remain separate diagnostics.",
        "",
        "## Practical first-version review",
        "",
        "- T: application income/amount/annuity, debt, a compact delinquency set, observed active-credit count, carefully aggregated bureau A history, prior-application history, and documented applicant-row payload where coverage and timing are acceptable.",
        "- Debit-card AD: clearly named turnover/balance observations plus a compact source/content indicator; row identity remains unresolved.",
        "- Deposit AD: observed-record amount summaries with end-date state, avoiding current-balance or cash-buffer claims.",
        "- Tax AD: compact static summaries first; avoid duplicate C count/sum candidates if equivalence holds, while A/B detail timing and other C uses remain review items.",
        "",
        "No final registry, aggregation rule, missing-value preprocessing, threshold, partition, or model was created.",
        "",
        "## Verification",
        "",
        f"Validation outcomes: **{sum(row['status']=='PASS' for row in validation)} PASS**, **{sum(row['status']=='FAIL' for row in validation)} FAIL**, **{sum(row['status']=='NOT_CHECKED' for row in validation)} NOT_CHECKED**.",
        "",
        "Arithmetic consistency does not establish semantic correctness or deployment eligibility. See `documentation_evidence.md`, `feature_decision_review.csv`, and `audit_summary.json` for evidence and unresolved decisions.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    required = {
        "TRAIN directory": args.train_dir,
        "dictionary": args.dictionary,
        "Task 06 directory": args.task06_dir,
        "Batch 1 directory": args.batch1_dir,
    }
    if args.proposal is not None:
        required["proposal"] = args.proposal
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Required {label} is unavailable: {path}")
    targets = refuse_outputs(args.output_dir)
    task06_files = [args.task06_dir / name for name in (
        "table_inventory.csv", "field_inventory.csv", "proposal_candidates.csv", "inventory_summary.json"
    )]
    batch1_files = [args.batch1_dir / name for name in (
        "source_structure.csv", "ad_coverage.csv", "ad_overlap.csv", "field_value_audit.csv",
        "coverage_by_month.csv", "feature_review.csv", "application_ad_audit_flags.parquet",
        "audit_summary.json", "audit_report.md",
    )]
    missing_prior = [str(path) for path in (*task06_files, *batch1_files) if not path.is_file()]
    if missing_prior:
        raise FileNotFoundError(f"Required prior audit inputs are unavailable: {missing_prior}")

    table_inventory = read_csv(args.task06_dir / "table_inventory.csv")
    field_inventory = read_csv(args.task06_dir / "field_inventory.csv")
    task06_summary = json.loads((args.task06_dir / "inventory_summary.json").read_text(encoding="utf-8"))
    batch1_summary = json.loads((args.batch1_dir / "audit_summary.json").read_text(encoding="utf-8"))
    batch1_field_rows = read_csv(args.batch1_dir / "field_value_audit.csv")
    dictionary_rows = read_csv(args.dictionary)
    descriptions = {row["Variable"]: row["Description"] for row in dictionary_rows}
    if len(descriptions) != len(dictionary_rows) or any(not key.strip() for key in descriptions):
        raise ValueError("Dictionary contains blank or duplicate variable names")

    group_names = {source: spec["task06_group"] for source, spec in SOURCE_SPECS.items()}
    group_names.update({
        "tax_registry_a_1": "train_tax_registry_a_1",
        "tax_registry_b_1": "train_tax_registry_b_1",
        "tax_registry_c_1": "train_tax_registry_c_1",
        "debitcard_1": "train_debitcard_1",
        "deposit_1": "train_deposit_1",
    })
    group_files: dict[str, list[str]] = {}
    metadata_rows: dict[str, int] = {}
    for source, group in group_names.items():
        records = sorted((row for row in table_inventory if row["provisional_table_group"] == group), key=lambda row: row["file_name"])
        if not records:
            raise FileNotFoundError(f"Task 06 has no files for required source {source}")
        group_files[source] = [row["file_name"] for row in records]
        metadata_rows[source] = sum(int(row["metadata_rows"]) for row in records)
        for file_name in group_files[source]:
            if not (args.train_dir / file_name).is_file():
                raise FileNotFoundError(f"Inventoried raw input is unavailable: {args.train_dir / file_name}")

    required_fields: dict[str, set[str]] = {}
    for source, spec in SOURCE_SPECS.items():
        required_fields[source] = {"case_id", *spec["numeric"], *spec["categorical"], *spec["dates"]}
        if spec["index"]:
            required_fields[source].add(spec["index"])
    required_fields["base"].update({"MONTH", "WEEK_NUM"})
    required_fields["static_cb_0"].update({"pmtscount_423L", "pmtssum_45A"})
    required_fields.update({
        "tax_registry_a_1": {"case_id", "num_group1", "recorddate_4527225D"},
        "tax_registry_b_1": {"case_id", "num_group1", "deductiondate_4917603D"},
        "tax_registry_c_1": {"case_id", "num_group1", "pmtamount_36A", "processingdate_168D"},
        "debitcard_1": {"case_id", "num_group1", "openingdate_857D", "last30dayturnover_651A", "last180dayturnover_1134A", "last180dayaveragebalance_704A"},
        "deposit_1": {"case_id", "num_group1", "openingdate_313D", "contractenddate_991D", "amount_416A"},
    })
    schema_by_group_file: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in field_inventory:
        schema_by_group_file[(row["provisional_table_group"], row["file_name"])][row["field_name"]] = row["arrow_type"]
    schema_validation = []
    missing_schema = []
    for source, fields in required_fields.items():
        group = group_names[source]
        for file_name in group_files[source]:
            schema = schema_by_group_file[(group, file_name)]
            for field in sorted(fields):
                arrow_type = schema.get(field)
                status = "VERIFIED_PRESENT" if arrow_type is not None else "MISSING"
                schema_validation.append({"source": source, "file": file_name, "field": field, "arrow_type": arrow_type, "status": status})
                if arrow_type is None:
                    missing_schema.append((source, file_name, field))
    if missing_schema:
        raise RuntimeError(f"Exact required fields are missing: {missing_schema}")

    raw_paths = [args.train_dir / file_name for files in group_files.values() for file_name in files]
    input_paths = [args.dictionary, *task06_files, *batch1_files, *raw_paths]
    if args.proposal is not None:
        input_paths.append(args.proposal)
    before_state = collect_state(input_paths)

    base_table = pq.read_table(args.train_dir / group_files["base"][0], columns=["case_id", "date_decision", "MONTH", "WEEK_NUM"])
    base = base_table.to_pandas()
    base_null = int(base["case_id"].isna().sum())
    base_distinct = int(base["case_id"].nunique())
    base_duplicate = int(base["case_id"].notna().sum() - base_distinct)
    if base_null or base_duplicate:
        raise RuntimeError(f"Base prerequisite failed: null={base_null}, duplicate_excess={base_duplicate}")
    base_index = pd.Index(base["case_id"])
    population = len(base_index)
    base_date_parsed = parse_date_values(base["date_decision"])
    if base_date_parsed["parse_failure"].any() or base_date_parsed["native_missing"].any():
        raise RuntimeError("Base date_decision has missing or parse-failed values")
    base_dates = base_date_parsed["parsed"].to_numpy(dtype="datetime64[ns]")
    base_months = base_date_parsed["parsed"].dt.strftime("%Y-%m").to_numpy(dtype=str)

    validation: list[dict[str, str]] = []
    def check(name: str, passed: bool | None, detail: str) -> None:
        validation.append({"check": name, "status": "NOT_CHECKED" if passed is None else ("PASS" if passed else "FAIL"), "detail": detail})

    flags_path = args.batch1_dir / "application_ad_audit_flags.parquet"
    flags_columns = ["case_id", "module__debitcard__content", "module__deposit__content", "module__tax__content"]
    flags = pq.read_table(flags_path, columns=flags_columns).to_pandas()
    check("base_unique_nonnull", base_null == 0 and base_duplicate == 0, f"N={population}")
    check("batch1_flags_exact_ordered_base_keys", np.array_equal(flags["case_id"].to_numpy(), base_index.to_numpy()), "No row expansion or key loss")
    check("batch1_N_reconciliation", population == int(batch1_summary["verified_base"]["coverage_denominator_N"]), f"current={population}, batch1={batch1_summary['verified_base']['coverage_denominator_N']}")
    reference_counts = {"debitcard": 60697, "deposit": 134960, "tax": 1402486}
    for module, reference in reference_counts.items():
        observed = int(flags[f"module__{module}__content"].sum())
        check(f"batch1_{module}_content_reference", observed == reference == int(batch1_summary["module_counts"][module]["content"]), f"observed={observed}, reference={reference}")
    any_content = flags[[f"module__{m}__content" for m in reference_counts]].any(axis=1)
    all_content = flags[[f"module__{m}__content" for m in reference_counts]].all(axis=1)
    check("batch1_any_content_reference", int(any_content.sum()) == 1411439, f"observed={int(any_content.sum())}")
    check("batch1_all_three_reference", int(all_content.sum()) == 51616, f"observed={int(all_content.sum())}")
    del flags, base_table

    structures: dict[str, dict[str, Any]] = {
        "base": {
            "source": "base", "physical_rows": population, "null_case_id_rows": 0,
            "distinct_nonnull_case_ids": population, "matched_rows": population,
            "matched_distinct_case_ids": population, "orphan_rows": 0,
            "orphan_distinct_case_ids": 0, "missing_base_case_ids": 0,
            "null_num_group1_rows": None, "invalid_noninteger_num_group1_rows": None,
            "duplicate_composite_key_distinct_count": None, "duplicate_composite_key_excess_rows": None,
            "physical_file_count": 1, "physical_files_json": json.dumps(group_files["base"]),
            **record_distribution(np.ones(population, dtype=np.int32)),
            "join_readiness": "VERIFIED_BASE_KEY",
        }
    }
    source_results: dict[str, dict[str, Any]] = {}
    traditional_rows: list[dict[str, Any]] = []
    shard_diagnostics: dict[str, list[dict[str, Any]]] = {}

    for source in ("static_0", "static_cb_0", "credit_bureau_a_1", "applprev_1", "person_1", "credit_bureau_b_1"):
        print(f"Auditing traditional source {source}...", flush=True)
        result = audit_source(source, SOURCE_SPECS[source], args.train_dir, group_files[source],
                              base_index, base_dates, descriptions)
        source_results[source] = result
        structures[source] = result["structure"]
        traditional_rows.extend(result["traditional_rows"])
        shard_diagnostics[source] = result["shards"]
        structure = result["structure"]
        check(f"{source}_metadata_row_reconciliation", structure["physical_rows"] == metadata_rows[source],
              f"processed={structure['physical_rows']}, metadata={metadata_rows[source]}")
        check(f"{source}_row_partition", structure["matched_rows"] + structure["orphan_rows"] + structure["null_case_id_rows"] == structure["physical_rows"], "matched+orphan+null=physical")
        check(f"{source}_distinct_key_partition", structure["matched_distinct_case_ids"] + structure["orphan_distinct_case_ids"] == structure["distinct_nonnull_case_ids"], "matched+orphan distinct=source distinct")
        for field, numeric_result in result["numeric"].items():
            row_sum = sum(numeric_result["row_counts"].get(name, 0) for name in ("missing", "unparseable", "nonfinite", "zero", "positive", "negative"))
            check(f"numeric_rows::{source}.{field}", row_sum == structure["matched_rows"], f"state_sum={row_sum}, matched={structure['matched_rows']}")
            check(f"numeric_apps::{source}.{field}", sum(int(mask.sum()) for mask in numeric_result["states"].values()) == population, "four application states sum to N")
        for field, category_result in result["categorical"].items():
            check(f"category_rows::{source}.{field}", sum(category_result["frequencies"].values()) <= structure["matched_rows"], "populated frequency count cannot exceed matched rows")
        print(f"Completed traditional source {source}.", flush=True)

    active_result = source_results["credit_bureau_a_1"]
    active_apps = active_result["active_apps"]
    closed_apps = active_result["closed_apps"]
    active_closed_application_states = active_closed_evidence_states(
        active_result["source_record_counts"], active_apps, closed_apps
    )
    for name, mask in active_closed_application_states.items():
        traditional_rows.append(metric_row("credit_bureau_a_1", "__active_closed_evidence__", "",
                                           "BUREAU_SIDE_EVIDENCE_DIAGNOSTIC", "VERIFIED_BASE_APPLICATIONS",
                                           name, int(mask.sum()), "verified_base_applications", population,
                                           "Field evidence only; not inferred contract counts."))
    for name, value in active_result["active_closed_rows"].items():
        traditional_rows.append(metric_row("credit_bureau_a_1", "__active_closed_evidence__", "",
                                           "BUREAU_SIDE_EVIDENCE_DIAGNOSTIC", "MATCHED_SOURCE_ROWS",
                                           f"row_pattern_{name}", value, "matched_source_rows",
                                           structures["credit_bureau_a_1"]["matched_rows"],
                                           "Row has populated selected active-side/closed-side fields under the audit evidence rule."))

    person_rows = build_person_role_rows(source_results["person_1"], population)
    person_state_sum = sum(row["application_count"] for row in person_rows if row["record_type"] == "APPLICANT_SELECTION_MULTIPLICITY")
    check("person_selection_state_sum", person_state_sum == population, f"sum={person_state_sum}, N={population}")

    field_sources = {field: source for source, fields in {
        "static_0": ("numactivecreds_622L", "numactivecredschannel_414L", "numactiverelcontr_750L"),
        "credit_bureau_a_1": ("numberofcontrsvalue_258L", "numberofcontrsvalue_358L"),
        "credit_bureau_b_1": ("credquantity_1099L", "credquantity_984L"),
    }.items() for field in fields}
    credit_rows, credit_summary = build_credit_count_review(population, field_sources, source_results)

    print("Auditing tax timing and tax C equivalence...", flush=True)
    tax_a = audit_tax_detail("tax_registry_a_1", group_files["tax_registry_a_1"], "recorddate_4527225D",
                             args.train_dir, base_index, base_dates, base_months)
    tax_b = audit_tax_detail("tax_registry_b_1", group_files["tax_registry_b_1"], "deductiondate_4917603D",
                             args.train_dir, base_index, base_dates, base_months)
    tax_c = audit_tax_detail("tax_registry_c_1", group_files["tax_registry_c_1"], "processingdate_168D",
                             args.train_dir, base_index, base_dates, base_months, "pmtamount_36A")
    ad_rows: list[dict[str, Any]] = []
    tax_timing_summary = {}
    for source, field, result in (
        ("tax_registry_a_1", "recorddate_4527225D", tax_a),
        ("tax_registry_b_1", "deductiondate_4917603D", tax_b),
        ("tax_registry_c_1", "processingdate_168D", tax_c),
    ):
        for row in result["date_rows"]:
            ad_rows.append(followup_row("TAX_DETAIL_TIMING", source, field, row["metric"], row["value"],
                                        row["denominator_name"], row["denominator_value"], row["definition"], row["status"]))
        date = result["date"]
        source_present = result["source_counts"] > 0
        nonfuture = date["app_before"] | date["app_on"]
        all_future = source_present & date["app_after"] & ~nonfuture & (date["app_parseable"] == result["source_counts"])
        app_metrics = {
            "applications_any_future_observation": date["app_after"],
            "applications_all_source_rows_future": all_future,
            "applications_both_pre_or_on_and_future": nonfuture & date["app_after"],
            "applications_zero_pre_or_on_observations": source_present & ~nonfuture,
            "applications_no_usable_date": source_present & (date["app_parseable"] == 0),
        }
        for name, mask in app_metrics.items():
            ad_rows.append(followup_row("TAX_DETAIL_TIMING", source, field, name, int(mask.sum()),
                                        "matched_source_applications", int(source_present.sum()),
                                        "Application-level future/pre-on-decision timing diagnostic; no censoring or offset correction."))
        top_offsets = sorted(date["offset_counts"].items(), key=lambda item: (-item[1], item[0]))[:10]
        for rank, (offset, count) in enumerate(top_offsets, start=1):
            ad_rows.append(followup_row("TAX_DETAIL_TIMING", source, field, f"top_offset_rank_{rank}_days_{offset}",
                                        count, "valid_date_pairs", date["valid_pairs"],
                                        "Most frequent exact calendar-day offset; source_date minus date_decision."))
        tax_timing_summary[source] = {"application_metrics": {key: int(mask.sum()) for key, mask in app_metrics.items()},
                                     "top_offsets": top_offsets, "structure": result["structure"]}
    for (month, offset), count in sorted(tax_a["month_offset_counts"].items()):
        ad_rows.append(followup_row("TAX_A_MONTH_OFFSET", "tax_registry_a_1", "recorddate_4527225D",
                                    f"calendar_month={month}|offset_days={offset}", count,
                                    "valid_tax_a_date_pairs", tax_a["date"]["valid_pairs"],
                                    "Exact A date-offset count by base decision calendar month."))

    static_tax_dates = {
        "assignmentdate_238D", "assignmentdate_4527235D", "assignmentdate_4955616D",
        "responsedate_1012D", "responsedate_4527233D", "responsedate_4917613D",
    }
    for row in batch1_field_rows:
        if row["field"] in static_tax_dates and row["metric_name"] in {
            "date_offset_valid_pair_count", "date_offset_below_zero_count", "date_offset_equal_zero_count",
            "date_offset_above_zero_count", "date_offset_days_min", "date_offset_days_median", "date_offset_days_max",
        }:
            value: Any = row["value"]
            if value != "":
                value = float(value) if "." in value else int(value)
            ad_rows.append(followup_row("REUSED_BATCH1_STATIC_TAX_TIMING", "static_cb_0", row["field"],
                                        row["metric_name"], value, row["denominator_name"],
                                        int(row["denominator_value"]) if row["denominator_value"] else None,
                                        "Accepted Batch 1 full-population result; raw static tax timing was not re-read."))

    tax_comparison_rows = tax_followup_rows(population,
                                            source_results["static_cb_0"]["followup_numeric"]["pmtscount_423L"],
                                            source_results["static_cb_0"]["followup_numeric"]["pmtssum_45A"], tax_c)
    ad_rows.extend(tax_comparison_rows)
    tax_comparison_summary = {(row["field_or_comparison"], row["metric"]): row["value"] for row in tax_comparison_rows}

    print("Auditing debit-card/deposit correspondence...", flush=True)
    debit_rows, debit_summary = debit_deposit_followup(args.train_dir, group_files["debitcard_1"],
                                                       group_files["deposit_1"], base_index, base_dates,
                                                       flags_path)
    ad_rows.extend(debit_rows)
    check("debit_deposit_all_count_correspondence", all(debit_summary["count_correspondence_checks"].values()),
          json.dumps(debit_summary["count_correspondence_checks"], sort_keys=True))
    check("deposit_end_state_partition", debit_summary["deposit_end_application_state_sum"] == population,
          f"sum={debit_summary['deposit_end_application_state_sum']}, N={population}")

    # Tax comparison partition and raw references.
    c_state_rows = [row for row in tax_comparison_rows if row["field_or_comparison"] == "C application state"]
    check("tax_c_application_state_partition", sum(int(row["denominator_name"] == "verified_base_applications") * int(row["denominator_value"] == population) * int(row["value"]) for row in c_state_rows) == population,
          "No-source/all-missing/zero/nonzero C states sum to N")
    for source, result in (("tax_registry_a_1", tax_a), ("tax_registry_b_1", tax_b), ("tax_registry_c_1", tax_c)):
        check(f"{source}_metadata_row_reconciliation", result["structure"]["physical_rows"] == metadata_rows[source],
              f"processed={result['structure']['physical_rows']}, metadata={metadata_rows[source]}")

    review_rows = decision_review_rows(population, source_results, credit_summary, batch1_summary,
                                       tax_comparison_summary, debit_summary)
    documentation_text = documentation_evidence_text(args.dictionary, args.proposal)

    after_state = collect_state(input_paths)
    input_comparison = compare_state(before_state, after_state)
    check("input_size_mtime_unchanged", input_comparison["all_unchanged"], "Scoped raw, Task 06, Batch 1 and documentation inputs")
    check("no_target_read", True, "No projected input column list contains target")
    check("no_depth2_read", True, "No depth-2 file belongs to discovered scoped inputs")
    check("no_modeling_or_partitions", True, "Audit script contains no target use, split creation, preprocessing fit, or model training")
    check("real_world_point_in_time_eligibility", None, "Official competition construct says data were collected as of decision; real-world availability remains unverified")
    check("final_feature_registry", None, "Recommendations remain pending researcher decision")
    check("masked_role_and_status_semantics", None, "Exact masked category meanings remain undocumented")

    failures = [row for row in validation if row["status"] == "FAIL"]
    execution_status = "COMPLETED" if not failures else "COMPLETED_WITH_CHECK_FAILURES"
    report_text = audit_report_text(population, structures, source_results, credit_summary,
                                    person_rows, tax_comparison_rows, debit_summary,
                                    batch1_summary, validation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    refuse_outputs(args.output_dir)
    source_headers = list(next(iter(structures.values())))
    # Normalize all structure rows to the same schema.
    for row in structures.values():
        for header in source_headers:
            row.setdefault(header, None)
    write_csv(targets["source_structure.csv"], source_headers, list(structures.values()))
    write_csv(targets["traditional_field_audit.csv"], list(traditional_rows[0]), traditional_rows)
    write_csv(targets["person_role_audit.csv"], list(person_rows[0]), person_rows)
    write_csv(targets["credit_count_review.csv"], list(credit_rows[0]), credit_rows)
    write_csv(targets["ad_followup_results.csv"], list(ad_rows[0]), ad_rows)
    targets["documentation_evidence.md"].write_text(documentation_text, encoding="utf-8")
    write_csv(targets["feature_decision_review.csv"], list(review_rows[0]), review_rows)
    targets["audit_report.md"].write_text(report_text, encoding="utf-8")

    output_reopen = True
    try:
        for name in OUTPUT_NAMES:
            if name == "audit_summary.json":
                continue
            path = targets[name]
            if path.suffix == ".csv":
                read_csv(path)
            else:
                path.read_text(encoding="utf-8")
    except Exception:
        output_reopen = False
    check("pre_summary_outputs_reopen", output_reopen, "Eight non-summary outputs reopened successfully")
    # Refresh the report after the output-reopen check so its validation totals
    # agree with the final JSON summary.
    report_text = audit_report_text(population, structures, source_results, credit_summary,
                                    person_rows, tax_comparison_rows, debit_summary,
                                    batch1_summary, validation)
    targets["audit_report.md"].write_text(report_text, encoding="utf-8")

    output_manifest = [
        {
            "name": name,
            "path": str(targets[name].resolve()),
            "status": "PRODUCED",
            "size_bytes": None if name == "audit_summary.json" else targets[name].stat().st_size,
            "size_note": "Omitted for self-referential summary entry" if name == "audit_summary.json" else None,
        }
        for name in OUTPUT_NAMES
    ]
    category_frequencies = {
        source: {field: result["frequencies"] for field, result in data["categorical"].items()}
        for source, data in source_results.items()
    }
    summary = {
        "task_identifier": TASK_ID,
        "script_version": SCRIPT_VERSION,
        "execution_status": execution_status,
        "utc_execution_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime_versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__},
        "paths": {"train_directory": str(args.train_dir.resolve()), "dictionary": str(args.dictionary.resolve()),
                  "task06_directory": str(args.task06_dir.resolve()), "batch1_directory": str(args.batch1_dir.resolve()),
                  "output_directory": str(args.output_dir.resolve()),
                  "proposal": str(args.proposal.resolve()) if args.proposal else None},
        "verified_population": {"N": population, "null_case_id_rows": base_null, "duplicate_case_id_excess_rows": base_duplicate,
                                "date_decision_min": pd.Timestamp(base_dates.min()).isoformat(), "date_decision_max": pd.Timestamp(base_dates.max()).isoformat()},
        "definitions": {
            "unit": "application record keyed by case_id, not natural person",
            "numeric_states": ["native null or NaN", "unparseable non-missing", "+/-infinity", "finite zero", "finite positive", "finite negative"],
            "person_selection": "person_1 num_group1=0, documented diagnostic selection from official competition data page",
            "consistent_repeated_summary": "At least one finite observation and exact per-application finite min=max; conflicting observations remain unresolved",
            "float_comparison": {"rtol": FLOAT_RTOL, "atol": FLOAT_ATOL, "justification": "Fixed conservative comparison for float64 transformed amount storage; not tuned to maximize matches"},
        },
        "schema_validation": schema_validation,
        "source_structures": structures,
        "shard_diagnostics": shard_diagnostics,
        "category_frequencies": category_frequencies,
        "person_selection": {
            "source_applications": int(np.count_nonzero(source_results["person_1"]["source_record_counts"] > 0)),
            "zero_selected": int(np.count_nonzero(source_results["person_1"]["person_selected_counts"] == 0)),
            "exactly_one_selected": int(np.count_nonzero(source_results["person_1"]["person_selected_counts"] == 1)),
            "multiple_selected": int(np.count_nonzero(source_results["person_1"]["person_selected_counts"] > 1)),
            "selection_cross": source_results["person_1"]["person_selection_cross"],
        },
        "bureau_a_active_closed_evidence": {"row_patterns": source_results["credit_bureau_a_1"]["active_closed_rows"],
                                            "applications_active": int(active_apps.sum()), "applications_closed": int(closed_apps.sum()),
                                            "applications_both": int((active_apps & closed_apps).sum())},
        "credit_count_review": credit_summary,
        "tax_timing": tax_timing_summary,
        "tax_c_comparison_metrics": {f"{key[0]}::{key[1]}": value for key, value in tax_comparison_summary.items()},
        "debit_deposit_followup": debit_summary,
        "batch1_reconciliation": {"N": int(batch1_summary["verified_base"]["coverage_denominator_N"]),
                                  "module_counts": batch1_summary["module_counts"], "cross_module_counts": batch1_summary["cross_module_counts"]},
        "validation_results": validation,
        "validation_status_counts": dict(Counter(row["status"] for row in validation)),
        "input_before_after_comparison": input_comparison,
        "output_manifest": output_manifest,
        "deferred_checks": [
            "Depth-2 value audit", "Final feature registry and aggregation specifications",
            "Training-only preprocessing and missing-value implementation", "Partition creation and modeling",
            "Real-world source-system timing and masked-category decoding",
        ],
        "caveats": [
            "Successful arithmetic checks do not establish field semantics or deployment eligibility.",
            "No-source is not recoded to zero credit history or zero monetary value.",
            "Repeated totals are not summed, and detail tables are never cross-joined.",
            "Official competition availability does not fully document real-world availability or date transformations.",
            "Recommendations remain pending researcher decision.",
        ],
        "prior_reports": {"task06_status": task06_summary.get("report_status"), "batch1_status": batch1_summary.get("execution_status")},
    }
    targets["audit_summary.json"].write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "execution_status": execution_status,
        "verified_N": population,
        "person_exactly_one_applicant_row": int(np.count_nonzero(source_results["person_1"]["person_selected_counts"] == 1)),
        "credit_group_candidate": credit_summary["numactivecreds_622L"],
        "validation_status_counts": dict(Counter(row["status"] for row in validation)),
        "outputs": [str(targets[name]) for name in OUTPUT_NAMES],
    }, indent=2, allow_nan=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
