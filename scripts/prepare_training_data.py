#!/usr/bin/env python3
"""Prepare frozen training inputs from the completed Task 08 feature table.

This module deliberately implements only deterministic data preparation.  It
does not instantiate or fit any predictive, calibration, or policy model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


VERSION = "1.0.0"
EXPECTED_N = 1_526_659
CAP_QUANTILE = 0.999
MISSING_TOKEN = "__MISSING__"
UNSEEN_TOKEN = "__UNSEEN__"
ALIAS_OLD = "t__last_approved_credit_amount"
ALIAS_NEW = "t__last_application_credit_amount"

T_CATEGORICAL = [
    "t__credit_product_type",
    "t__applicant_income_type",
    "t__applicant_education",
]
T_MONETARY = [
    "t__recorded_primary_income",
    "t__applicant_main_income",
    "t__requested_credit_amount",
    "t__current_application_annuity",
    "t__current_debt",
    "t__total_debt",
    ALIAS_NEW,
]
DPD_FIELDS = [
    "t__max_dpd_last_3m",
    "t__max_dpd_last_12m",
    "t__max_dpd_last_24m",
]
T_COUNT = [
    "t__observed_active_credit_count",
    "t__bureau_queries_30d",
    "t__bureau_queries_360d",
    "t__client_loan_payment_count",
    "t__applications_30d",
    "t__contracts_3m",
    "t__active_revolving_credit_count",
    "t__paid_installments_last_contract",
]
AD_TAX_COUNT = [
    "ad__tax__pmtscount_423L",
    "ad__tax__pmtcount_693L",
    "ad__tax__pmtcount_4527229L",
    "ad__tax__pmtcount_4955617L",
]
AD_TAX_MONETARY = [
    "ad__tax__pmtssum_45A",
    "ad__tax__pmtaverage_3A",
    "ad__tax__pmtaverage_4527227A",
    "ad__tax__pmtaverage_4955615A",
    "ad__tax__amount_4527230A__finite_mean",
    "ad__tax__amount_4527230A__finite_max",
    "ad__tax__amount_4917619A__finite_mean",
    "ad__tax__amount_4917619A__finite_max",
    "ad__tax__pmtamount_36A__finite_mean",
    "ad__tax__pmtamount_36A__finite_max",
]
AD_INDICATORS = [
    "ad__debit__has_numeric_content",
    "ad__debit__has_nonzero_content",
    "ad__debit__has_source_evidence",
    "ad__deposit__has_numeric_content",
    "ad__deposit__has_nonzero_content",
    "ad__tax__has_numeric_content",
    "ad__tax__has_nonzero_content",
    "ad__tax__has_source_or_field_evidence",
]
METADATA = ["case_id", "date_decision", "WEEK_NUM", "MONTH"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA or value is None:
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def file_state(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path.resolve()), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def sha256_text(values: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def schema_fingerprint(columns: list[str], dtypes: Iterable[Any]) -> str:
    return sha256_text(f"{c}\t{d}" for c, d in zip(columns, dtypes))


def ensure_new_targets(audit_dir: Path, interim_dir: Path) -> None:
    existing = [p for p in (audit_dir, interim_dir) if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing task output: " + ", ".join(map(str, existing)))


def aliased_feature_sets(source: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(source))
    for key in ("T", "T_plus_AD"):
        result[key] = [ALIAS_NEW if c == ALIAS_OLD else c for c in result[key]]
    return result


def apply_alias(frame: pd.DataFrame) -> pd.DataFrame:
    has_old, has_new = ALIAS_OLD in frame.columns, ALIAS_NEW in frame.columns
    if has_old == has_new:
        raise ValueError(f"Input must contain exactly one of {ALIAS_OLD!r} and {ALIAS_NEW!r}")
    return frame.rename(columns={ALIAS_OLD: ALIAS_NEW}) if has_old else frame


def stratified_holdout(labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return retained and held-out positions using independent class shuffles.

    Per-class held-out counts use round-half-to-even through Python round.  This
    is recorded explicitly because scikit-learn is absent from the environment.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Stratification labels must be a one-dimensional binary array")
    rng = np.random.default_rng(seed)
    keep_parts: list[np.ndarray] = []
    hold_parts: list[np.ndarray] = []
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        perm = rng.permutation(idx)
        n_hold = int(round(len(idx) * fraction))
        hold_parts.append(perm[:n_hold])
        keep_parts.append(perm[n_hold:])
    return np.sort(np.concatenate(keep_parts)), np.sort(np.concatenate(hold_parts))


def make_splits(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(len(labels), dtype=np.int64)
    remain_local, evaluation = stratified_holdout(labels, 0.15, 20260918)
    remaining = positions[remain_local]
    train_local, validation_local = stratified_holdout(labels[remaining], 15 / 85, 20260919)
    train = remaining[train_local]
    validation = remaining[validation_local]
    tuning_local, calibration_local = stratified_holdout(labels[validation], 0.5, 20260920)
    tuning = validation[tuning_local]
    calibration = validation[calibration_local]
    outer = np.full(len(labels), "", dtype=object)
    role = np.full(len(labels), "", dtype=object)
    outer[train] = "train"
    outer[validation] = "validation"
    outer[evaluation] = "evaluation"
    role[tuning] = "validation_tuning"
    role[calibration] = "validation_calibration"
    if np.any(outer == "") or len(np.unique(np.concatenate([train, validation, evaluation]))) != len(labels):
        raise AssertionError("Outer partitions are not exhaustive and disjoint")
    if set(np.flatnonzero(role != "")) != set(validation):
        raise AssertionError("Validation roles do not exactly partition validation")
    return outer, role


def original_source_group(column: str, all_columns: set[str]) -> tuple[str, list[str]]:
    for suffix in ("__finite_mean", "__finite_max"):
        if column.endswith(suffix):
            base = column[: -len(suffix)]
            deps = [base + "__finite_mean", base + "__finite_max"]
            if all(dep in all_columns for dep in deps):
                return base, deps
    return column, [column]


def category_output_name(column: str, category: str) -> str:
    digest = hashlib.sha1(category.encode("utf-8")).hexdigest()[:10]
    return f"ohe__{column}__{digest}"


@dataclass
class TrainingPreprocessor:
    construction_version: str = VERSION
    t_features: list[str] = field(default_factory=list)
    ad_substantive: list[str] = field(default_factory=list)
    ad_indicators: list[str] = field(default_factory=lambda: list(AD_INDICATORS))
    numeric_substantive: list[str] = field(default_factory=list)
    monetary: list[str] = field(default_factory=list)
    dpd: list[str] = field(default_factory=lambda: list(DPD_FIELDS))
    categories: list[str] = field(default_factory=lambda: list(T_CATEGORICAL))
    cap_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    scaler_means: dict[str, float] = field(default_factory=dict)
    scaler_scales: dict[str, float] = field(default_factory=dict)
    constant_columns: list[str] = field(default_factory=list)
    all_missing_columns: list[str] = field(default_factory=list)
    missing_flags: list[dict[str, Any]] = field(default_factory=list)
    category_vocabularies: dict[str, list[str]] = field(default_factory=dict)
    category_outputs: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    output_features: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    split_fingerprint: str = ""
    training_key_fingerprint: str = ""
    fitted: bool = False

    def configure(self, feature_sets: dict[str, Any]) -> None:
        self.t_features = list(feature_sets["T"])
        self.ad_substantive = list(feature_sets["AD_substantive_values"])
        self.ad_indicators = list(feature_sets["AD_indicators"])
        ad_debit_deposit = [c for c in self.ad_substantive if c.startswith("ad__debit__") or c.startswith("ad__deposit__")]
        self.monetary = [*T_MONETARY, *ad_debit_deposit, *AD_TAX_MONETARY]
        t_numeric = [c for c in self.t_features if c not in T_CATEGORICAL]
        self.numeric_substantive = [*t_numeric, *self.ad_substantive]
        expected = {
            "T": 21,
            "T_numeric": 18,
            "AD_substantive": 27,
            "AD_indicators": 8,
            "monetary": 30,
            "DPD": 3,
            "counts": 12,
            "categorical": 3,
            "numeric_substantive": 45,
        }
        observed = {
            "T": len(self.t_features), "T_numeric": len(t_numeric),
            "AD_substantive": len(self.ad_substantive), "AD_indicators": len(self.ad_indicators),
            "monetary": len(self.monetary), "DPD": len(self.dpd),
            "counts": len(T_COUNT) + len(AD_TAX_COUNT), "categorical": len(self.categories),
            "numeric_substantive": len(self.numeric_substantive),
        }
        if observed != expected:
            raise ValueError(f"Explicit type-list contract mismatch: observed={observed}, expected={expected}")
        if set(self.monetary) & set(self.dpd):
            raise ValueError("Monetary and DPD type lists overlap")
        if set(self.numeric_substantive) != set(self.monetary + self.dpd + T_COUNT + AD_TAX_COUNT):
            raise ValueError("Explicit numeric type lists do not cover the 45 substantive numeric inputs")

    def _validate_input(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = apply_alias(frame)
        expected = self.t_features + self.ad_substantive + self.ad_indicators
        missing = [c for c in expected if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing approved feature columns: {missing}")
        values = frame[expected].copy()
        for column in self.numeric_substantive:
            values[column] = pd.to_numeric(values[column], errors="raise").astype(np.float64)
            arr = values[column].to_numpy(np.float64, copy=False)
            if np.isinf(arr).any():
                raise ValueError(f"Infinite numeric input is invalid: {column}")
        for column in self.ad_indicators:
            vals = set(values[column].dropna().astype(int).unique())
            if values[column].isna().any() or not vals.issubset({0, 1}):
                raise ValueError(f"Binary indicator contract failed: {column}, values={vals}")
        return values

    def fit(
        self,
        frame: pd.DataFrame,
        train_mask: np.ndarray,
        feature_sets: dict[str, Any],
        split_fingerprint: str,
        training_key_fingerprint: str,
    ) -> "TrainingPreprocessor":
        self.configure(feature_sets)
        values = self._validate_input(frame)
        train_mask = np.asarray(train_mask, dtype=bool)
        if len(train_mask) != len(values) or not train_mask.any():
            raise ValueError("A nonempty row-aligned outer-TRAIN mask is required")
        self.split_fingerprint = split_fingerprint
        self.training_key_fingerprint = training_key_fingerprint

        # T/count/DPD negatives are audited, never repaired.
        forbidden_negative = [
            c for c in self.numeric_substantive if c not in self.ad_substantive or c in AD_TAX_COUNT
            if bool((values.loc[train_mask, c] < 0).any())
        ]
        # All substantive AD fields that are monetary are the only fields clipped.
        ad_monetary = [c for c in self.monetary if c.startswith("ad__")]
        if forbidden_negative:
            raise ValueError(f"Unexpected negative T/count/DPD values: {forbidden_negative}")

        processed = values[self.numeric_substantive].copy()
        for column in ad_monetary:
            arr = processed[column].to_numpy(np.float64, copy=False)
            processed[column] = np.where(np.isfinite(arr) & (arr < 0), 0.0, arr)

        for column in self.monetary:
            train = processed.loc[train_mask, column].to_numpy(np.float64)
            finite = train[np.isfinite(train)]
            if not len(finite):
                self.cap_parameters[column] = {
                    "status": "TRAIN_ALL_MISSING", "quantile": CAP_QUANTILE,
                    "method": "linear", "train_finite_count": 0, "cap": None,
                }
                self.all_missing_columns.append(column)
                continue
            cap = float(np.quantile(finite, CAP_QUANTILE, method="linear"))
            positive = bool(np.any(finite > 0))
            if cap == 0 and positive:
                status, applied_cap = "CAP_SKIPPED_DEGENERATE_ZERO", None
            else:
                status, applied_cap = "CAP_APPLIED", cap
                processed[column] = np.where(np.isfinite(processed[column]), np.minimum(processed[column], cap), np.nan)
            self.cap_parameters[column] = {
                "status": status, "quantile": CAP_QUANTILE, "method": "linear",
                "train_finite_count": int(len(finite)), "cap": applied_cap,
                "computed_quantile": cap, "train_positive_count": int(np.count_nonzero(finite > 0)),
            }

        transformed = processed.copy()
        for column in self.monetary + self.dpd:
            arr = transformed[column].to_numpy(np.float64, copy=False)
            finite = np.isfinite(arr)
            if np.any(arr[finite] < 0):
                raise ValueError(f"Negative input cannot be passed to log1p: {column}")
            transformed[column] = np.log1p(arr)

        for column in self.numeric_substantive:
            train = transformed.loc[train_mask, column].to_numpy(np.float64)
            finite = train[np.isfinite(train)]
            if len(finite):
                median = float(np.median(finite))
            else:
                median = 0.0
                if column not in self.all_missing_columns:
                    self.all_missing_columns.append(column)
            self.medians[column] = median
            filled_train = np.where(np.isfinite(train), train, median)
            mean = float(np.mean(filled_train, dtype=np.float64))
            scale = float(np.std(filled_train, ddof=0, dtype=np.float64))
            if not math.isfinite(scale) or scale == 0:
                scale = 1.0
                self.constant_columns.append(column)
            self.scaler_means[column] = mean
            self.scaler_scales[column] = scale

        all_numeric = set(self.numeric_substantive)
        handled: set[str] = set()
        for column in self.numeric_substantive:
            if column in handled or not values.loc[train_mask, column].isna().any():
                continue
            base, deps = original_source_group(column, all_numeric)
            if len(deps) == 2:
                same = np.array_equal(
                    values.loc[train_mask, deps[0]].isna().to_numpy(),
                    values.loc[train_mask, deps[1]].isna().to_numpy(),
                )
                if not same:
                    raise ValueError(f"Construction-guaranteed mean/max masks differ: {deps}")
            else:
                deps = [column]
                base = column
            name = f"missing__{base}"
            self.missing_flags.append({"name": name, "dependencies": deps, "family": "T" if column.startswith("t__") else "AD"})
            handled.update(deps)

        for column in self.categories:
            source = values.loc[train_mask, column]
            observed = sorted(source.dropna().astype(str).unique().tolist())
            if MISSING_TOKEN in observed or UNSEEN_TOKEN in observed:
                raise ValueError(f"Reserved category token collision in {column}")
            vocab = [*observed, MISSING_TOKEN, UNSEEN_TOKEN]
            self.category_vocabularies[column] = vocab
            self.category_outputs[column] = [
                {"category": category, "output": category_output_name(column, category)} for category in vocab
            ]

        t_numeric = [c for c in self.t_features if c not in self.categories]
        t_missing = [r["name"] for r in self.missing_flags if r["family"] == "T"]
        ad_missing = [r["name"] for r in self.missing_flags if r["family"] == "AD"]
        ohe = [row["output"] for c in self.categories for row in self.category_outputs[c]]
        t_outputs = [*t_numeric, *t_missing, *ohe]
        ad_outputs = [*self.ad_substantive, *self.ad_indicators, *ad_missing]
        self.output_features = {
            "linear_nn": {"T": t_outputs, "T_plus_AD": [*t_outputs, *ad_outputs]},
            "gbdt": {"T": list(t_outputs), "T_plus_AD": [*t_outputs, *ad_outputs]},
        }
        self.fitted = True
        return self

    def _shared_numeric(self, values: pd.DataFrame) -> pd.DataFrame:
        shared = values[self.numeric_substantive].copy()
        for column in [c for c in self.monetary if c.startswith("ad__")]:
            arr = shared[column].to_numpy(np.float64, copy=False)
            shared[column] = np.where(np.isfinite(arr) & (arr < 0), 0.0, arr)
        for column, params in self.cap_parameters.items():
            cap = params.get("cap")
            if cap is not None:
                arr = shared[column].to_numpy(np.float64, copy=False)
                shared[column] = np.where(np.isfinite(arr), np.minimum(arr, cap), np.nan)
        return shared

    def transform(self, frame: pd.DataFrame, representation: str) -> pd.DataFrame:
        if not self.fitted or representation not in {"linear_nn", "gbdt"}:
            raise ValueError("A fitted preprocessor and valid representation are required")
        values = self._validate_input(frame)
        shared = self._shared_numeric(values)
        if representation == "linear_nn":
            for column in self.monetary + self.dpd:
                arr = shared[column].to_numpy(np.float64, copy=False)
                finite = np.isfinite(arr)
                if np.any(arr[finite] < 0):
                    raise ValueError(f"Negative transform-time input cannot enter log1p: {column}")
                shared[column] = np.log1p(arr)
            numeric_data = {
                c: ((shared[c].fillna(self.medians[c]).to_numpy(np.float64) - self.scaler_means[c]) / self.scaler_scales[c]).astype(np.float32)
                for c in self.numeric_substantive
            }
        else:
            numeric_data = {c: shared[c].to_numpy(np.float32) for c in self.numeric_substantive}

        output: dict[str, np.ndarray] = dict(numeric_data)
        for column in self.ad_indicators:
            output[column] = values[column].to_numpy(np.int8)
        for item in self.missing_flags:
            # Shared mean/max flags use either dependency after verified equality.
            output[item["name"]] = values[item["dependencies"][0]].isna().to_numpy(np.int8)
        for column in self.categories:
            source = values[column]
            mapped = source.astype("string").fillna(MISSING_TOKEN).astype(str)
            known = set(self.category_vocabularies[column]) - {MISSING_TOKEN, UNSEEN_TOKEN}
            mapped = mapped.where(mapped.isin(known | {MISSING_TOKEN}), UNSEEN_TOKEN)
            for row in self.category_outputs[column]:
                output[row["output"]] = mapped.eq(row["category"]).to_numpy(np.int8)
        order = self.output_features[representation]["T_plus_AD"]
        return pd.DataFrame({column: output[column] for column in order}, index=frame.index)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingPreprocessor":
        obj = cls()
        for key, value in data.items():
            if not hasattr(obj, key):
                raise ValueError(f"Unknown serialized preprocessor field: {key}")
            setattr(obj, key, value)
        if not obj.fitted:
            raise ValueError("Serialized preprocessor is not fitted")
        return obj

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "TrainingPreprocessor":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def coverage_masks(features: pd.DataFrame, feature_sets: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for family, columns in feature_sets["AD_substantive_by_family"].items():
        array = features[columns].to_numpy(np.float64)
        finite = np.isfinite(array)
        result[f"{family}_finite"] = finite.any(axis=1)
        result[f"{family}_nonzero"] = (finite & (array != 0)).any(axis=1)
    result["any_ad"] = result["debit_finite"] | result["deposit_finite"] | result["tax_finite"]
    result["no_ad"] = ~result["any_ad"]
    result["tax_only"] = result["tax_finite"] & ~result["debit_finite"] & ~result["deposit_finite"]
    result["banking_any"] = result["debit_finite"] | result["deposit_finite"]
    result["all_three"] = result["debit_finite"] & result["deposit_finite"] & result["tax_finite"]
    return result


def summarize_split(
    name: str, mask: np.ndarray, features: pd.DataFrame, target: np.ndarray,
    coverage: dict[str, np.ndarray], nested: bool,
) -> dict[str, Any]:
    n = int(mask.sum())
    dates = pd.to_datetime(features.loc[mask, "date_decision"], errors="coerce")
    weeks = pd.to_numeric(features.loc[mask, "WEEK_NUM"], errors="coerce")
    row: dict[str, Any] = {
        "split": name, "is_nested_validation_subgroup": nested, "N": n,
        "target_1_count": int(target[mask].sum()), "target_1_rate": float(target[mask].mean()),
        "date_min": dates.min().date().isoformat(), "date_max": dates.max().date().isoformat(),
        "week_min": int(weeks.min()), "week_max": int(weeks.max()),
    }
    for key, values in coverage.items():
        row[f"{key}_count"] = int(np.count_nonzero(values & mask))
        row[f"{key}_rate"] = float(np.count_nonzero(values & mask) / n)
    return row


def banking_examples(
    train_dir: Path, feature_table: pd.DataFrame, feature_sets: dict[str, Any]
) -> list[dict[str, Any]]:
    specs = {
        "debitcard_1": (
            train_dir / "train_debitcard_1.parquet",
            ["last30dayturnover_651A", "last180dayturnover_1134A", "last180dayaveragebalance_704A"],
            "debit",
        ),
        "deposit_1": (train_dir / "train_deposit_1.parquet", ["amount_416A"], "deposit"),
    }
    feature_by_key = feature_table.set_index("case_id", drop=False)
    rows: list[dict[str, Any]] = []
    for source, (path, fields, family) in specs.items():
        raw = pd.read_parquet(path, columns=["case_id", "num_group1", *fields])
        selected: list[int] = []
        for field in fields:
            numeric = pd.to_numeric(raw[field], errors="coerce")
            work = pd.DataFrame({"case_id": raw["case_id"], "value": numeric})
            grouped = work.groupby("case_id", sort=True)["value"].agg(
                source_row_count="size", finite_count="count", distinct_finite="nunique",
                finite_mean="mean", finite_max="max",
                zero_count=lambda x: int((x == 0).sum()),
            )
            grouped["missing_count"] = grouped["source_row_count"] - grouped["finite_count"]
            grouped["preferred_mixed_state"] = (
                ((grouped["zero_count"] > 0) | (grouped["missing_count"] > 0))
                & (grouped["finite_count"] > grouped["zero_count"])
            )
            good = grouped[(grouped["finite_count"] >= 2) & (grouped["distinct_finite"] >= 2)]
            good = good.sort_values(["preferred_mixed_state", "source_row_count"], ascending=[False, False], kind="stable")
            for case_id in good.index[:20]:
                if int(case_id) not in selected:
                    selected.append(int(case_id))
                    break
            if len(selected) >= 2:
                break
        if len(selected) < 2:
            # Closest deterministic fallback: at least two finite values.
            numeric = raw[fields].apply(pd.to_numeric, errors="coerce")
            counts = numeric.notna().groupby(raw["case_id"]).sum().max(axis=1)
            for case_id in counts[counts >= 2].sort_index().index:
                if int(case_id) not in selected:
                    selected.append(int(case_id))
                if len(selected) == 2:
                    break
        for case_id in selected[:2]:
            case_rows = raw.loc[raw["case_id"].eq(case_id)].sort_values("num_group1")
            for field in fields:
                numeric = pd.to_numeric(case_rows[field], errors="coerce").astype(float)
                finite = numeric[np.isfinite(numeric)]
                expected_mean = float(finite.mean()) if len(finite) else None
                expected_max = float(finite.max()) if len(finite) else None
                mean_col = f"ad__{family}__{field}__finite_mean"
                max_col = f"ad__{family}__{field}__finite_max"
                observed_mean = feature_by_key.at[case_id, mean_col]
                observed_max = feature_by_key.at[case_id, max_col]
                mean_diff = None if expected_mean is None else float(observed_mean - expected_mean)
                max_diff = None if expected_max is None else float(observed_max - expected_max)
                status = "PASS" if (
                    (expected_mean is None and pd.isna(observed_mean) and pd.isna(observed_max))
                    or (np.isclose(observed_mean, expected_mean, rtol=1e-10, atol=1e-8)
                        and np.isclose(observed_max, expected_max, rtol=1e-10, atol=1e-8))
                ) else "FAIL"
                for _, source_row in case_rows.iterrows():
                    value = pd.to_numeric(pd.Series([source_row[field]]), errors="coerce").iloc[0]
                    state = "MISSING_OR_INVALID" if pd.isna(value) else ("ZERO" if float(value) == 0 else "FINITE_NONZERO")
                    rows.append({
                        "source": source, "case_id": case_id, "raw_field": field,
                        "num_group1": int(source_row["num_group1"]), "raw_value": value,
                        "classified_state": state, "finite_count": int(len(finite)),
                        "independent_finite_mean": expected_mean, "independent_finite_max": expected_max,
                        "task08_mean": observed_mean, "task08_max": observed_max,
                        "mean_difference": mean_diff, "max_difference": max_diff,
                        "multi_record_nonzero_varied": bool(len(finite) >= 2 and finite.nunique() >= 2 and (finite != 0).any()),
                        "comparison_status": status,
                    })
    return rows


def build_effects(
    values: pd.DataFrame, preprocessor: TrainingPreprocessor, outer: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shared = preprocessor._shared_numeric(preprocessor._validate_input(values))
    for column in preprocessor.numeric_substantive:
        original = pd.to_numeric(apply_alias(values)[column], errors="coerce").to_numpy(np.float64)
        clipped_negative = original.copy()
        if column in preprocessor.monetary and column.startswith("ad__"):
            clipped_negative[np.isfinite(clipped_negative) & (clipped_negative < 0)] = 0
        capped = shared[column].to_numpy(np.float64)
        transformed = capped.copy()
        if column in preprocessor.monetary + preprocessor.dpd:
            transformed = np.log1p(transformed)
        for part in ("train", "validation", "evaluation"):
            mask = outer == part
            raw = original[mask]
            neg_stage = clipped_negative[mask]
            cap_stage = capped[mask]
            trans = transformed[mask]
            finite_raw = raw[np.isfinite(raw)]
            finite_cap = cap_stage[np.isfinite(cap_stage)]
            finite_trans = trans[np.isfinite(trans)]
            params = preprocessor.cap_parameters.get(column, {})
            cap = params.get("cap")
            def q(a: np.ndarray, p: float) -> float | None:
                return float(np.quantile(a, p, method="linear")) if len(a) else None
            rows.append({
                "input_column": column, "outer_partition": part, "N": int(mask.sum()),
                "original_finite_count": int(len(finite_raw)), "original_missing_count": int(np.count_nonzero(np.isnan(raw))),
                "original_zero_count": int(np.count_nonzero(np.isfinite(raw) & (raw == 0))),
                "original_negative_count": int(np.count_nonzero(np.isfinite(raw) & (raw < 0))),
                "negative_clipped_count": int(np.count_nonzero(np.isfinite(raw) & (raw < 0))) if column.startswith("ad__") and column in preprocessor.monetary else 0,
                "upper_clipped_count": int(np.count_nonzero(np.isfinite(neg_stage) & (cap is not None) & (neg_stage > cap))) if cap is not None else 0,
                "linear_missing_filled_count": int(np.count_nonzero(np.isnan(raw))),
                "linear_result_finite_count": int(mask.sum()), "linear_result_missing_count": 0,
                "gbdt_result_finite_count": int(len(finite_cap)), "gbdt_result_missing_count": int(np.count_nonzero(np.isnan(cap_stage))),
                "original_denominator": "original finite observations",
                "original_median": q(finite_raw, .5), "original_p99": q(finite_raw, .99), "original_max": float(finite_raw.max()) if len(finite_raw) else None,
                "capped_median": q(finite_cap, .5), "capped_p99": q(finite_cap, .99), "capped_max": float(finite_cap.max()) if len(finite_cap) else None,
                "transformed_median": q(finite_trans, .5), "transformed_p99": q(finite_trans, .99), "transformed_max": float(finite_trans.max()) if len(finite_trans) else None,
                "model_matrix_denominator": "all partition rows after linear imputation; observed values only for GBDT",
            })
    return rows


def chinese_meaning(column: str, description: str) -> str:
    direct = {
        "t__recorded_primary_income": "记录的主要收入金额",
        "t__applicant_main_income": "申请人主要职业收入金额",
        "t__requested_credit_amount": "本次申请的信贷金额或额度",
        "t__current_application_annuity": "本次申请的月供金额",
        "t__credit_product_type": "信贷产品类别",
        "t__applicant_income_type": "申请人收入来源类别",
        "t__applicant_education": "申请人教育类别（掩码代码）",
        "t__current_debt": "当前记录债务金额",
        "t__total_debt": "记录的总债务金额",
        "t__observed_active_credit_count": "观察到的活跃信贷数量",
        "t__max_dpd_last_3m": "最近3个月最大逾期天数",
        "t__max_dpd_last_12m": "最近12个月最大逾期天数",
        "t__max_dpd_last_24m": "最近24个月最大逾期天数",
        "t__bureau_queries_30d": "最近30天征信查询次数",
        "t__bureau_queries_360d": "最近360天征信查询次数",
        "t__client_loan_payment_count": "客户贷款付款次数",
        "t__applications_30d": "最近30天申请次数",
        "t__contracts_3m": "最近3个月合同数量",
        "t__active_revolving_credit_count": "活跃循环信贷数量",
        ALIAS_NEW: "上一次申请的信贷金额（不等同于已批准金额）",
        "t__paid_installments_last_contract": "上一合同已支付分期数",
    }
    if column in direct:
        return direct[column]
    if column.startswith("ad__debit__"):
        return "借记卡相关记录的金额/周转或其原始状态标志"
    if column.startswith("ad__deposit__"):
        return "存款或账户相关记录的金额/余额或其原始状态标志"
    if column.startswith("ad__tax__"):
        return "税务扣缴相关记录的金额、次数或其原始状态标志"
    return description or "已批准输入的计算表示"


def expanded_registry(
    old_registry: pd.DataFrame, preprocessor: TrainingPreprocessor
) -> list[dict[str, Any]]:
    old = old_registry.set_index("feature_name", drop=False)
    rows: list[dict[str, Any]] = []
    indicator_sources = {
        "ad__debit__has_numeric_content": "debitcard_1.last30dayturnover_651A,last180dayturnover_1134A,last180dayaveragebalance_704A; other_1.amtdebitincoming_4809443A,amtdebitoutgoing_4809440A",
        "ad__debit__has_nonzero_content": "same approved debit raw fields; any original finite nonzero observation",
        "ad__debit__has_source_evidence": "debitcard_1 source-row presence OR other_1 source-row presence",
        "ad__deposit__has_numeric_content": "deposit_1.amount_416A; other_1.amtdepositbalance_4809441A,amtdepositincoming_4809444A,amtdepositoutgoing_4809442A",
        "ad__deposit__has_nonzero_content": "same approved deposit raw fields; any original finite nonzero observation",
        "ad__tax__has_numeric_content": "static_cb_0 approved 8 tax fields; tax_registry_a_1.amount_4527230A; tax_registry_b_1.amount_4917619A; tax_registry_c_1.pmtamount_36A",
        "ad__tax__has_nonzero_content": "same approved tax raw fields; any original finite nonzero observation",
        "ad__tax__has_source_or_field_evidence": "Task07 Batch1 module__tax__field_evidence (populated approved tax fields plus static_cb_0 assignment/response dates and requesttype_4525192L, and registry record/deduction/processing dates) OR tax_registry_a_1/b_1/c_1 source-row presence",
    }
    output_to_category = {
        row["output"]: (column, row["category"])
        for column, mappings in preprocessor.category_outputs.items() for row in mappings
    }
    missing_by_name = {item["name"]: item for item in preprocessor.missing_flags}
    for representation in ("linear_nn", "gbdt"):
        for output in preprocessor.output_features[representation]["T_plus_AD"]:
            if output in output_to_category:
                input_column, category = output_to_category[output]
                source = old.loc[input_column]
                role = "ONE_HOT"
                definition = f"1 when {input_column} maps to category {category!r}, otherwise 0"
                dependencies = source["source_fields"]
                meaning = f"类别指示：{chinese_meaning(input_column, '')} = {category}"
                preprocessing = "TRAIN-only vocabulary; missing/unseen reserved tokens; one-hot; unscaled"
                family = "T"
            elif output in missing_by_name:
                item = missing_by_name[output]
                input_column = item["dependencies"][0]
                source = old.loc[ALIAS_OLD if input_column == ALIAS_NEW else input_column]
                role = "MISSING_FLAG"
                definition = "Original pre-processing missingness shared only for guaranteed mean/max pairs"
                dependencies = json.dumps(item["dependencies"], ensure_ascii=False)
                meaning = "原始字段是否缺失：" + chinese_meaning(input_column, "")
                preprocessing = "Original missing mask captured before clipping/capping/imputation; binary and unscaled"
                family = item["family"]
            else:
                input_column = output
                lookup = ALIAS_OLD if input_column == ALIAS_NEW else input_column
                source = old.loc[lookup]
                role = str(source["feature_role"])
                dependencies = indicator_sources.get(input_column, str(source["source_fields"]))
                definition = str(source["economic_description"])
                meaning = chinese_meaning(input_column, definition)
                family = "T" if input_column.startswith("t__") else str(source["feature_family"])
                if input_column in preprocessor.numeric_substantive:
                    preprocessing = (
                        "AD monetary negatives->0 where authorized; TRAIN 0.999 linear upper cap for monetary; "
                        + ("log1p; TRAIN median fill; TRAIN standardization" if representation == "linear_nn" else "raw scale; NaN preserved")
                    )
                else:
                    preprocessing = "Original binary state retained; no rescaling"
            source_name = ALIAS_OLD if input_column == ALIAS_NEW else input_column
            source_row = old.loc[source_name]
            rows.append({
                "representation": representation, "input_column": input_column,
                "historical_task08_input_column": source_name, "prepared_output_name": output,
                "alias_provenance": f"{ALIAS_OLD}->{ALIAS_NEW}" if input_column == ALIAS_NEW else "",
                "source_tables_fields_or_dependencies": dependencies, "feature_family": family,
                "prepared_role": role, "dictionary_or_feature_definition": definition,
                "chinese_plain_meaning": meaning, "row_selection_or_original_aggregation": source_row["row_selection_rule"],
                "representation_preprocessing": preprocessing, "zero_missing_rule": f"{source_row['zero_rule']} {source_row['missing_rule']}",
                "include_in_T": output in preprocessor.output_features[representation]["T"],
                "include_in_T_plus_AD": True, "units_timing_limitations": source_row["interpretation_limitations"],
                "decision_reason": "Frozen approved input; derived representation only, no added substantive candidate.",
            })
    return rows


def build_rules(preprocessor: TrainingPreprocessor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing_dependency = {dep: item["name"] for item in preprocessor.missing_flags for dep in item["dependencies"]}
    for representation in ("linear_nn", "gbdt"):
        for column in preprocessor.t_features + preprocessor.ad_substantive + preprocessor.ad_indicators:
            if column in preprocessor.categories:
                kind, steps = "categorical", "TRAIN vocabulary; reserved missing/unseen; deterministic one-hot"
                benefit, cost = "Nominal categories become numeric without ordinal assumption", "More columns; unseen levels have no learned category-specific effect"
            elif column in preprocessor.ad_indicators:
                kind, steps = "original_binary_indicator", "retain original 0/1; do not recompute after clipping; unscaled"
                benefit, cost = "Preserves observed collection/content state", "May reflect collection regime rather than economics"
            else:
                kind = "monetary" if column in preprocessor.monetary else ("DPD" if column in preprocessor.dpd else "count_like")
                negative = "finite negative->0" if column.startswith("ad__") and column in preprocessor.monetary else "no automatic action; negative is blocking for log inputs"
                cap = "TRAIN q0.999 linear cap" if column in preprocessor.monetary else "none"
                if representation == "linear_nn":
                    steps = f"{negative}; {cap}; " + ("log1p; " if column in preprocessor.monetary + preprocessor.dpd else "") + "TRAIN median fill; TRAIN mean/scale"
                    benefit, cost = "Finite, controlled-scale input for regularized Logit/MLP", "Tail compression, log functional-form change, and computational imputation"
                else:
                    steps = f"{negative}; {cap}; preserve NaN; no log/imputation/scaling"
                    benefit, cost = "Keeps raw-scale splits and native missingness for a compatible GBDT", "Shared cap reduces extreme-tail distinction; flags may be redundant"
            rows.append({
                "representation": representation, "input_column": column, "explicit_type": kind,
                "authorized_negative_action": "finite negative->0" if column.startswith("ad__") and column in preprocessor.monetary else "none",
                "steps": steps, "missing_flag": missing_dependency.get(column, ""),
                "fit_scope": "outer TRAIN only for caps/medians/scales/vocabularies; fixed application to all partitions",
                "method_benefit": benefit, "information_cost": cost,
                "in_T": column in preprocessor.t_features, "in_T_plus_AD": True,
            })
    return rows


def parameter_rows(preprocessor: TrainingPreprocessor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in preprocessor.numeric_substantive:
        cap = preprocessor.cap_parameters.get(column, {})
        rows.append({
            "parameter_type": "numeric", "input_column": column,
            "cap_quantile": cap.get("quantile"), "cap_method": cap.get("method"),
            "cap_train_observation_count": cap.get("train_finite_count"), "cap_status": cap.get("status"),
            "cap_value": cap.get("cap"), "computed_quantile": cap.get("computed_quantile"),
            "transformed_train_median": preprocessor.medians[column],
            "scaler_mean": preprocessor.scaler_means[column], "scaler_scale": preprocessor.scaler_scales[column],
            "constant_train_status": column in preprocessor.constant_columns,
            "train_all_missing_status": column in preprocessor.all_missing_columns,
            "split_fingerprint": preprocessor.split_fingerprint,
            "training_key_fingerprint": preprocessor.training_key_fingerprint,
            "artifact": "preprocessing/preprocessor.json",
        })
    for column in preprocessor.categories:
        rows.append({
            "parameter_type": "categorical", "input_column": column,
            "category_vocabulary_json": json.dumps(preprocessor.category_vocabularies[column], ensure_ascii=False),
            "reserved_missing_token": MISSING_TOKEN, "reserved_unseen_token": UNSEEN_TOKEN,
            "split_fingerprint": preprocessor.split_fingerprint,
            "training_key_fingerprint": preprocessor.training_key_fingerprint,
            "artifact": "preprocessing/preprocessor.json",
        })
    return rows


def category_audit(values: pd.DataFrame, preprocessor: TrainingPreprocessor, outer: np.ndarray) -> list[dict[str, Any]]:
    values = apply_alias(values)
    rows = []
    for column in preprocessor.categories:
        known = set(preprocessor.category_vocabularies[column]) - {MISSING_TOKEN, UNSEEN_TOKEN}
        for part in ("train", "validation", "evaluation"):
            source = values.loc[outer == part, column]
            missing = source.isna()
            unseen = ~missing & ~source.astype(str).isin(known)
            mapped = source.astype("string").fillna(MISSING_TOKEN).astype(str)
            mapped = mapped.where(mapped.isin(known | {MISSING_TOKEN}), UNSEEN_TOKEN)
            encoded = np.column_stack([
                mapped.eq(item["category"]).to_numpy(np.int8)
                for item in preprocessor.category_outputs[column]
            ])
            binary_ok = bool(np.isin(encoded, [0, 1]).all())
            row_sum_ok = bool(np.all(encoded.sum(axis=1) == 1))
            rows.append({
                "input_column": column, "outer_partition": part, "N": int(len(source)),
                "train_vocabulary_json": json.dumps(preprocessor.category_vocabularies[column], ensure_ascii=False),
                "reserved_missing_token": MISSING_TOKEN, "reserved_unseen_token": UNSEEN_TOKEN,
                "missing_count": int(missing.sum()), "unseen_count": int(unseen.sum()),
                "output_names_json": json.dumps([x["output"] for x in preprocessor.category_outputs[column]]),
                "output_count": len(preprocessor.category_outputs[column]),
                "binary_integrity": "PASS" if binary_ok else "FAIL",
                "one_hot_row_sum_integrity": "PASS" if row_sum_ok else "FAIL",
            })
    return rows


def preprocessing_examples(
    features: pd.DataFrame, preprocessor: TrainingPreprocessor
) -> list[dict[str, Any]]:
    values = apply_alias(features)
    shared = preprocessor._shared_numeric(preprocessor._validate_input(values))
    examples: list[tuple[int, str, str]] = []
    used: set[tuple[int, str]] = set()
    def add(mask: np.ndarray, column: str, reason: str) -> None:
        positions = np.flatnonzero(mask)
        if len(positions):
            item = (int(positions[0]), column)
            if item not in used and len(examples) < 16:
                examples.append((item[0], column, reason)); used.add(item)
    for column in [c for c in preprocessor.monetary if c.startswith("ad__")]:
        raw = pd.to_numeric(values[column], errors="coerce").to_numpy(np.float64)
        add(np.isfinite(raw) & (raw < 0), column, "authorized_AD_negative_clipped")
    for column in preprocessor.monetary:
        raw = pd.to_numeric(values[column], errors="coerce").to_numpy(np.float64)
        cap = preprocessor.cap_parameters[column].get("cap")
        if cap is not None:
            add(np.isfinite(raw) & (raw > cap), column, "high_tail_upper_clipped")
    for column in preprocessor.numeric_substantive:
        raw = pd.to_numeric(values[column], errors="coerce").to_numpy(np.float64)
        add(np.isnan(raw), column, "original_missing_imputed_only_for_linear_nn")
        add(np.isfinite(raw) & (raw == 0), column, "observed_zero_preserved_before_centering")
        if len(examples) >= 16:
            break
    rows: list[dict[str, Any]] = []
    for pos, column, reason in examples:
        raw = float(values.iloc[pos][column]) if pd.notna(values.iloc[pos][column]) else None
        neg = raw
        if raw is not None and column.startswith("ad__") and column in preprocessor.monetary and raw < 0:
            neg = 0.0
        capped = shared.iloc[pos][column]
        log_value = np.log1p(capped) if column in preprocessor.monetary + preprocessor.dpd and pd.notna(capped) else capped
        imputed = preprocessor.medians[column] if pd.isna(log_value) else float(log_value)
        standardized = (imputed - preprocessor.scaler_means[column]) / preprocessor.scaler_scales[column]
        missing_flag = next((item["name"] for item in preprocessor.missing_flags if column in item["dependencies"]), "")
        rows.append({
            "case_id": int(features.iloc[pos]["case_id"]), "selection_reason": reason, "input_column": column,
            "original_value": raw, "after_authorized_negative_clipping": neg,
            "after_shared_upper_cap": capped, "linear_log_or_raw_value": log_value,
            "linear_imputed_value": imputed, "linear_standardized_value": standardized,
            "gbdt_tree_input_value": capped, "missing_flag_name": missing_flag,
            "missing_flag_value": int(pd.isna(raw)) if missing_flag else None,
            "interpretation": "Computational representation; imputed values are not estimated true balances/income.",
        })
    return rows


def validation_row(check: str, status: str, observed: Any, expected: Any, explanation: str) -> dict[str, Any]:
    return {"check_id": check, "status": status, "observed": json.dumps(json_safe(observed), ensure_ascii=False),
            "expected": json.dumps(json_safe(expected), ensure_ascii=False), "explanation": explanation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", required=True, type=Path)
    parser.add_argument("--task08-audit-dir", required=True, type=Path)
    parser.add_argument("--task06-audit-dir", required=True, type=Path)
    parser.add_argument("--task07-batch1-dir", required=True, type=Path)
    parser.add_argument("--dictionary", required=True, type=Path)
    parser.add_argument("--train-base", required=True, type=Path)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--audit-output-dir", required=True, type=Path)
    parser.add_argument("--interim-output-dir", required=True, type=Path)
    parser.add_argument("--test-command", default="")
    parser.add_argument("--test-result", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_new_targets(args.audit_output_dir, args.interim_output_dir)
    required = [
        args.feature_table, args.dictionary, args.train_base,
        args.task08_audit_dir / "feature_registry.csv", args.task08_audit_dir / "feature_sets.json",
        args.task08_audit_dir / "raw_candidate_decisions.csv", args.task08_audit_dir / "feature_value_audit.csv",
        args.task08_audit_dir / "ad_coverage.csv", args.task08_audit_dir / "ad_overlap.csv",
        args.task08_audit_dir / "build_summary.json",
        args.task06_audit_dir / "table_inventory.csv", args.task07_batch1_dir / "application_ad_audit_flags.parquet",
        args.train_dir / "train_debitcard_1.parquet", args.train_dir / "train_deposit_1.parquet",
    ]
    missing_inputs = [str(p) for p in required if not p.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Required inputs missing: {missing_inputs}")
    input_before = {str(p): file_state(p) for p in required}

    source_sets = json.loads((args.task08_audit_dir / "feature_sets.json").read_text(encoding="utf-8"))
    feature_sets = aliased_feature_sets(source_sets)
    old_registry = pd.read_csv(args.task08_audit_dir / "feature_registry.csv")
    feature_pf = pq.ParquetFile(args.feature_table)
    schema_names = feature_pf.schema.names
    expected_schema = source_sets["metadata"] + source_sets["T_plus_AD"]
    if feature_pf.metadata.num_rows != EXPECTED_N or schema_names != expected_schema:
        raise RuntimeError(f"Frozen Task08 table contract failed: rows={feature_pf.metadata.num_rows}, schema_match={schema_names == expected_schema}")
    if len(source_sets["T"]) != 21 or len(source_sets["AD_substantive_values"]) != 27 or len(source_sets["AD_indicators"]) != 8 or len(source_sets["T_plus_AD"]) != 56:
        raise RuntimeError("Frozen Task08 feature-set counts failed")

    features = pd.read_parquet(args.feature_table)
    if features["case_id"].isna().any() or features["case_id"].duplicated().any():
        raise RuntimeError("Task08 feature keys must be non-null and unique")
    features = apply_alias(features)
    base = pd.read_parquet(args.train_base, columns=["case_id", "target"])
    base_audit = {
        "feature_rows": len(features), "feature_unique_keys": int(features["case_id"].nunique()),
        "feature_null_keys": int(features["case_id"].isna().sum()), "feature_duplicate_keys": int(features["case_id"].duplicated().sum()),
        "base_rows": len(base), "base_unique_keys": int(base["case_id"].nunique()),
        "base_null_keys": int(base["case_id"].isna().sum()), "base_duplicate_keys": int(base["case_id"].duplicated().sum()),
    }
    feature_keys = pd.Index(features["case_id"])
    base_keys = pd.Index(base["case_id"])
    base_audit["feature_keys_unmatched_to_base"] = int((~feature_keys.isin(base_keys)).sum())
    base_audit["base_keys_unmatched_to_features"] = int((~base_keys.isin(feature_keys)).sum())
    base_audit["target_missing_count"] = int(base["target"].isna().sum())
    base_audit["target_values_json"] = json.dumps(sorted(base["target"].dropna().unique().tolist()))
    blocking = (
        base_audit["feature_null_keys"] or base_audit["feature_duplicate_keys"] or base_audit["base_null_keys"]
        or base_audit["base_duplicate_keys"] or base_audit["feature_keys_unmatched_to_base"]
        or base_audit["base_keys_unmatched_to_features"] or base_audit["target_missing_count"]
        or set(base["target"].dropna().unique()) != {0, 1}
    )
    if blocking:
        raise RuntimeError(f"Blocking target-join integrity failure: {base_audit}")
    joined = features[["case_id"]].merge(base, on="case_id", how="left", validate="one_to_one", sort=False)
    base_audit["joined_key_order_equal"] = bool(joined["case_id"].equals(features["case_id"].reset_index(drop=True)))
    base_audit["joined_label_alignment_equal"] = bool(
        np.array_equal(joined.set_index("case_id").loc[features["case_id"], "target"].to_numpy(), joined["target"].to_numpy())
    )
    if not base_audit["joined_key_order_equal"] or not base_audit["joined_label_alignment_equal"]:
        raise RuntimeError("Target join did not preserve feature order/alignment")
    target = joined["target"].to_numpy(np.int8)

    outer, role = make_splits(target)
    membership_fp = sha256_text(f"{case_id}\t{o}\t{r}" for case_id, o, r in zip(features["case_id"], outer, role))
    train_mask = outer == "train"
    training_key_fp = sha256_text(features.loc[train_mask, "case_id"])
    outer2, role2 = make_splits(target)
    split_reproduced = bool(np.array_equal(outer, outer2) and np.array_equal(role, role2))

    coverage = coverage_masks(features, feature_sets)
    expected_coverage = {
        "debit_finite": 60_697, "deposit_finite": 134_960, "tax_finite": 1_402_486,
        "any_ad": 1_411_439, "no_ad": 115_220, "all_three": 51_616,
        "banking_any": 140_965, "tax_only": 1_270_474,
        "debit_nonzero": 34_062, "deposit_nonzero": 88_811, "tax_nonzero": 1_305_517,
    }
    observed_coverage = {key: int(mask.sum()) for key, mask in coverage.items()}
    if any(observed_coverage[k] != v for k, v in expected_coverage.items()):
        raise RuntimeError(f"Original coverage reconciliation failed: {observed_coverage}")

    preprocessor = TrainingPreprocessor().fit(
        features, train_mask, feature_sets, membership_fp, training_key_fp
    )
    linear = preprocessor.transform(features, "linear_nn")
    gbdt = preprocessor.transform(features, "gbdt")
    if not np.isfinite(linear.to_numpy(np.float32)).all():
        raise RuntimeError("Linear/NN matrix contains nonfinite values")
    if np.isinf(gbdt.select_dtypes(include=[np.number]).to_numpy(np.float32)).any():
        raise RuntimeError("GBDT matrix contains infinity")

    manifest = features[METADATA].copy()
    manifest.insert(1, "base_order", np.arange(len(features), dtype=np.int64))
    manifest["target"] = target
    manifest["outer_split"] = outer
    manifest["validation_role"] = role
    for key, mask in coverage.items():
        manifest[f"original_{key}"] = mask.astype(np.int8)
    linear.insert(0, "case_id", features["case_id"].to_numpy())
    gbdt.insert(0, "case_id", features["case_id"].to_numpy())

    bank_rows = banking_examples(args.train_dir, features, feature_sets)
    if not bank_rows or any(row["comparison_status"] != "PASS" for row in bank_rows):
        raise RuntimeError("Independent banking aggregation examples failed")

    split_masks = {
        "full_population": np.ones(len(features), dtype=bool),
        "train": outer == "train", "validation": outer == "validation",
        "validation_tuning": role == "validation_tuning",
        "validation_calibration": role == "validation_calibration",
        "evaluation": outer == "evaluation",
    }
    split_rows = [summarize_split(name, mask, features, target, coverage, name.startswith("validation_")) for name, mask in split_masks.items()]
    split_summary = pd.DataFrame(split_rows)[["split", "is_nested_validation_subgroup", "N", "target_1_count", "target_1_rate", "date_min", "date_max", "week_min", "week_max"]]
    split_coverage = pd.DataFrame(split_rows).drop(columns=["target_1_count", "target_1_rate", "date_min", "date_max", "week_min", "week_max"])
    monthly = (
        pd.DataFrame({"outer_split": outer, "decision_month": pd.to_datetime(features["date_decision"]).dt.strftime("%Y-%m"), "target": target})
        .groupby(["outer_split", "decision_month"], sort=True, observed=True)["target"]
        .agg(N="size", target_1_count="sum", target_1_rate="mean").reset_index()
    )

    target_audit_rows = [
        {"check": key, "observed": value, "status": "PASS", "meaning": "Measured target-join integrity result."}
        for key, value in base_audit.items()
    ]
    registry_rows = expanded_registry(old_registry, preprocessor)
    rules_rows = build_rules(preprocessor)
    params_rows = parameter_rows(preprocessor)
    effects_rows = build_effects(features, preprocessor, outer)
    category_rows = category_audit(features, preprocessor, outer)
    example_rows = preprocessing_examples(features, preprocessor)

    # Validation rows are measured before any file writing.
    validations: list[dict[str, Any]] = []
    validations.append(validation_row("task08_schema_count_whitelist", "PASS", [len(features), len(features.columns), len(feature_sets["T_plus_AD"])], [EXPECTED_N, 60, 56], "Frozen schema and whitelist verified after alias-only rename."))
    validations.append(validation_row("target_join_integrity", "PASS", base_audit, "unique complete binary one-to-one order-preserving join", "Blocking checks passed."))
    validations.append(validation_row("outer_split_disjoint_exhaustive", "PASS", pd.Series(outer).value_counts().to_dict(), len(features), "Every application has exactly one outer membership."))
    validations.append(validation_row("validation_roles_exact", "PASS", pd.Series(role[role != ""]).value_counts().to_dict(), int((outer == "validation").sum()), "Roles partition outer validation exactly."))
    validations.append(validation_row("split_reproduction", "PASS" if split_reproduced else "FAIL", split_reproduced, True, "Same seeds and labels reproduce identical membership."))
    validations.append(validation_row("coverage_reconciliation", "PASS", observed_coverage, expected_coverage, "Original pre-processing substantive coverage."))
    validations.append(validation_row("banking_aggregation_examples", "PASS", len(bank_rows), ">0 all PASS", "Independent direct group calculations match Task08 features."))
    validations.append(validation_row("alias_value_preservation", "PASS", ALIAS_NEW in features.columns and ALIAS_OLD not in features.columns, True, "Alias changes name only; historical Task08 input remains untouched."))
    ad_negative_fields = [c for c in preprocessor.monetary if c.startswith("ad__")]
    validations.append(validation_row("negative_clipping_scope", "PASS", len(ad_negative_fields), 23, "Only named AD monetary fields are eligible."))
    validations.append(validation_row("cap_train_observed_scope", "PASS", {c:p["train_finite_count"] for c,p in preprocessor.cap_parameters.items()}, "finite outer-TRAIN observations only", "Caps precede imputation and include observed zeros."))
    validations.append(validation_row("linear_all_finite", "PASS", True, True, "All prepared linear/NN values finite."))
    validations.append(validation_row("gbdt_no_infinity", "PASS", True, True, "NaN retained; infinity absent."))
    original_masks_match = all(np.array_equal(gbdt[c].isna().to_numpy(), features[c].isna().to_numpy()) for c in preprocessor.numeric_substantive)
    validations.append(validation_row("gbdt_original_nan_masks", "PASS" if original_masks_match else "FAIL", original_masks_match, True, "Shared clipping/capping preserve original numeric NaN masks."))
    t_block_equal = preprocessor.output_features["linear_nn"]["T"] == preprocessor.output_features["linear_nn"]["T_plus_AD"][:len(preprocessor.output_features["linear_nn"]["T"])] and preprocessor.output_features["gbdt"]["T"] == preprocessor.output_features["gbdt"]["T_plus_AD"][:len(preprocessor.output_features["gbdt"]["T"])]
    validations.append(validation_row("T_exact_prefix", "PASS" if t_block_equal else "FAIL", t_block_equal, True, "T is the exact leading block in T+AD for both representations."))
    excluded = set(METADATA + ["target", "base_order", "outer_split", "validation_role"])
    predictors = set(preprocessor.output_features["linear_nn"]["T_plus_AD"] + preprocessor.output_features["gbdt"]["T_plus_AD"])
    validations.append(validation_row("metadata_label_excluded", "PASS" if not excluded & predictors else "FAIL", sorted(excluded & predictors), [], "No metadata, target, or membership fields in predictor lists."))
    validations.append(validation_row("no_model_fitted", "PASS", "preprocessing only", "no predictor/calibrator/policy model", "No model API is imported or instantiated."))
    validations.append(validation_row("target_semantics", "NOT_CHECKABLE", "binary competition target only", "formal horizon/real-world meaning", "Existing authorized evidence does not verify formal target horizon or profitability meaning."))
    if any(row["status"] == "FAIL" for row in validations):
        raise RuntimeError("One or more pre-write validation checks failed")

    # First writes occur only after all blocking computation and checks succeed.
    args.audit_output_dir.mkdir(parents=True)
    args.interim_output_dir.mkdir(parents=True)
    prep_dir = args.interim_output_dir / "preprocessing"
    prep_dir.mkdir()
    artifact_path = prep_dir / "preprocessor.json"
    preprocessor.save(artifact_path)
    reload_preprocessor = TrainingPreprocessor.load(artifact_path)
    sample_positions = np.unique(np.linspace(0, len(features) - 1, 101, dtype=int))
    sample = features.iloc[sample_positions].copy()
    reload_linear = reload_preprocessor.transform(sample, "linear_nn")
    reload_gbdt = reload_preprocessor.transform(sample, "gbdt")
    reload_ok = bool(
        np.allclose(reload_linear.to_numpy(np.float64), linear.iloc[sample_positions, 1:].to_numpy(np.float64), rtol=0, atol=1e-6, equal_nan=True)
        and np.allclose(reload_gbdt.to_numpy(np.float64), gbdt.iloc[sample_positions, 1:].to_numpy(np.float64), rtol=0, atol=1e-6, equal_nan=True)
    )
    validations.append(validation_row("preprocessor_reload_sample", "PASS" if reload_ok else "FAIL", reload_ok, True, "JSON artifact transforms approved label-free sample identically; atol=1e-6."))
    if not reload_ok:
        raise RuntimeError("Preprocessor reload verification failed")

    manifest_path = args.interim_output_dir / "application_manifest.parquet"
    linear_path = args.interim_output_dir / "linear_nn_inputs.parquet"
    gbdt_path = args.interim_output_dir / "gbdt_inputs.parquet"
    manifest.to_parquet(manifest_path, index=False, compression="zstd")
    linear.to_parquet(linear_path, index=False, compression="zstd")
    del linear
    gbdt.to_parquet(gbdt_path, index=False, compression="zstd")

    prepared_sets = {
        "construction_version": VERSION,
        "metadata_exclusions": ["case_id", "date_decision", "WEEK_NUM", "MONTH", "base_order", "target", "outer_split", "validation_role", "original coverage labels"],
        "historical_candidates": source_sets,
        "alias_mapping": {ALIAS_OLD: ALIAS_NEW},
        "linear_nn": preprocessor.output_features["linear_nn"],
        "gbdt": preprocessor.output_features["gbdt"],
        "counts": {
            "substantive_numeric": 45, "original_binary": 8,
            "missing_flags": len(preprocessor.missing_flags),
            "one_hot": sum(len(v) for v in preprocessor.category_outputs.values()),
            "linear_nn_T": len(preprocessor.output_features["linear_nn"]["T"]),
            "linear_nn_T_plus_AD": len(preprocessor.output_features["linear_nn"]["T_plus_AD"]),
            "gbdt_T": len(preprocessor.output_features["gbdt"]["T"]),
            "gbdt_T_plus_AD": len(preprocessor.output_features["gbdt"]["T_plus_AD"]),
        },
        "fit_artifact": str(artifact_path.resolve()),
    }
    write_json(prep_dir / "parameters.json", {
        "construction_version": VERSION, "split_fingerprint": membership_fp,
        "training_key_fingerprint": training_key_fp, "parameters": preprocessor.to_dict(),
    })
    write_json(prep_dir / "feature_sets.json", prepared_sets)

    write_csv(args.audit_output_dir / "banking_aggregation_examples.csv", bank_rows)
    write_csv(args.audit_output_dir / "target_join_audit.csv", target_audit_rows)
    write_csv(args.audit_output_dir / "feature_registry.csv", registry_rows)
    write_csv(args.audit_output_dir / "split_summary.csv", split_summary)
    write_csv(args.audit_output_dir / "split_coverage.csv", split_coverage)
    write_csv(args.audit_output_dir / "split_monthly_composition.csv", monthly)
    write_csv(args.audit_output_dir / "preprocessing_rules.csv", rules_rows)
    write_csv(args.audit_output_dir / "preprocessing_parameters.csv", params_rows)
    write_json(args.audit_output_dir / "preprocessing_parameters.json", {
        "construction_version": VERSION, "split_fingerprint": membership_fp,
        "training_key_fingerprint": training_key_fp, "artifact": str(artifact_path.resolve()),
        "parameters": preprocessor.to_dict(),
    })
    write_csv(args.audit_output_dir / "preprocessing_effects.csv", effects_rows)
    write_csv(args.audit_output_dir / "categorical_encoding_audit.csv", category_rows)
    write_json(args.audit_output_dir / "feature_sets.json", prepared_sets)
    write_csv(args.audit_output_dir / "preprocessing_examples.csv", example_rows)

    # Reopen saved Parquets and verify physical contract.
    reopened = {}
    for name, path, expected_columns in (
        ("manifest", manifest_path, manifest.columns.tolist()),
        ("linear_nn", linear_path, ["case_id", *preprocessor.output_features["linear_nn"]["T_plus_AD"]]),
        ("gbdt", gbdt_path, ["case_id", *preprocessor.output_features["gbdt"]["T_plus_AD"]]),
    ):
        pf = pq.ParquetFile(path)
        reopened[name] = {"rows": pf.metadata.num_rows, "columns": pf.schema.names, "size_bytes": path.stat().st_size}
        if pf.metadata.num_rows != len(features) or pf.schema.names != expected_columns:
            raise RuntimeError(f"Saved Parquet contract failed: {name}")
        sampled_keys = pd.read_parquet(path, columns=["case_id"]).iloc[sample_positions]["case_id"].to_numpy()
        if not np.array_equal(sampled_keys, features.iloc[sample_positions]["case_id"].to_numpy()):
            raise RuntimeError(f"Saved Parquet key order failed: {name}")
    validations.append(validation_row("saved_parquet_reopen_contract", "PASS", reopened, "row/schema/key-order match", "All three prepared Parquets reopened and matched registered contracts."))

    input_after = {str(p): file_state(p) for p in required}
    unchanged = input_before == input_after
    validations.append(validation_row("input_size_mtime_unchanged", "PASS" if unchanged else "FAIL", unchanged, True, "Lightweight size/mtime comparison; not cryptographic proof."))
    if not unchanged:
        raise RuntimeError("An input size or mtime changed during preparation")
    write_csv(args.audit_output_dir / "validation_results.csv", validations)

    runtime = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pa.__version__,
        "scikit_learn": "NOT_INSTALLED", "joblib": "NOT_INSTALLED",
    }
    report = f"""# Task 08 follow-up — training-data preparation report

Status: **COMPLETED**. This follow-up is the training-preparation milestone after the completed Task 08 feature build; historical Task 08 files were not rewritten.

## Prepared population and partitions

- Verified applications: {len(features):,}; target join: one-to-one, complete, binary, order preserving.
- Outer split counts: train {int((outer == 'train').sum()):,}; validation {int((outer == 'validation').sum()):,}; final evaluation {int((outer == 'evaluation').sum()):,}.
- Validation roles: tuning {int((role == 'validation_tuning').sum()):,}; calibration {int((role == 'validation_calibration').sum()):,}.
- Membership fingerprint: `{membership_fp}`.
- Original any-AD coverage: {observed_coverage['any_ad']:,}; no finite AD: {observed_coverage['no_ad']:,}. Accepted Task 08 coverage counts reconcile exactly.

## Representation contract

All learned parameters use outer TRAIN only. The linear/NN representation applies authorized AD-negative clipping, shared monetary caps, log1p to monetary/DPD fields, median filling, and standardization. The GBDT representation uses the same clipping/caps but preserves raw scale and NaN. Both share original missing flags and TRAIN-only one-hot mapping.

- Linear/NN: T {len(preprocessor.output_features['linear_nn']['T'])} columns; T+AD {len(preprocessor.output_features['linear_nn']['T_plus_AD'])} columns.
- GBDT: T {len(preprocessor.output_features['gbdt']['T'])} columns; T+AD {len(preprocessor.output_features['gbdt']['T_plus_AD'])} columns.
- Reload verification: PASS on {len(sample_positions)} deterministic rows at absolute tolerance 1e-6.
- `constant_train_columns` covers the 45 substantive numeric inputs only; reserved binary/one-hot outputs can be constant in TRAIN and are audited separately before modeling.
- Test command/result supplied for this run: `{args.test_command or 'not supplied'}` / `{args.test_result or 'not supplied'}`.
- scikit-learn/joblib were not installed; the fixed formulas and deterministic stratified split were implemented directly with NumPy/Pandas and saved as strict JSON.

## Boundary and limitations

No predictive model, calibration model, decision threshold, or policy simulation was fitted. Random-split success later would not establish time stability, prospective deployment validity, or borrower independence. Median-filled values are computational placeholders, and source/coverage flags can reflect collection regimes rather than real-world absence.
"""
    (args.audit_output_dir / "preparation_report.md").write_text(report, encoding="utf-8")
    output_files = sorted([p for p in args.audit_output_dir.rglob("*") if p.is_file()] + [p for p in args.interim_output_dir.rglob("*") if p.is_file()])
    summary = {
        "task": "TASK_08_FOLLOWUP_TRAINING_PREPARATION",
        "repository_task_index_mapping": "Task 08 follow-up; README records this as the training-preparation milestone without rewriting Task 08.",
        "construction_version": VERSION, "execution_status": "COMPLETED", "utc_timestamp": utc_now(),
        "runtime_versions": runtime, "exact_execution_command": " ".join(sys.argv),
        "split_implementation": "NumPy default_rng class-wise shuffle; per-class holdout count=round(class_count*fraction); sorted original integer positions; seeds 20260918/20260919/20260920.",
        "tests_actually_run": {"command": args.test_command or None, "result": args.test_result or None},
        "verified_population": len(features), "target_distribution": {str(k): int(v) for k, v in pd.Series(target).value_counts().sort_index().items()},
        "target_join": base_audit, "split_counts": {k: int(v) for k, v in pd.Series(outer).value_counts().items()},
        "validation_role_counts": {k: int(v) for k, v in pd.Series(role[role != ""]).value_counts().items()},
        "membership_fingerprint": membership_fp, "training_key_fingerprint": training_key_fp,
        "coverage": observed_coverage, "prepared_feature_counts": prepared_sets["counts"],
        "cap_status_counts": pd.Series([x["status"] for x in preprocessor.cap_parameters.values()]).value_counts().to_dict(),
        "all_missing_train_columns": preprocessor.all_missing_columns, "constant_train_columns": preprocessor.constant_columns,
        "constant_train_columns_scope": "45 substantive numeric columns only; binary, missing-indicator, and one-hot outputs are outside this field.",
        "reload_verification": {"status": "PASS", "sample_rows": len(sample_positions), "absolute_tolerance": 1e-6},
        "saved_parquets": reopened,
        "validation_status_counts": pd.Series([r["status"] for r in validations]).value_counts().to_dict(),
        "input_state_before": input_before, "input_state_after": input_after,
        "produced_files": [{**file_state(p), "relative_to_project": os.path.relpath(p, args.audit_output_dir.parent.parent.parent)} for p in output_files],
        "deferred": ["predictive model fitting", "calibration model fitting", "threshold/policy selection", "chronological supplement", "new features or ratios"],
        "limitations": [
            "Random application split does not establish temporal stability or borrower-level independence.",
            "Formal target horizon and real-world profitability meaning remain unverified.",
            "JSON artifact requires compatible trusted local code; it is not a production service.",
            "Input preservation check uses size and mtime, not cryptographic hashes.",
            "scikit-learn and joblib were absent; prescribed formulas were implemented with NumPy/Pandas without installation.",
        ],
    }
    write_json(args.audit_output_dir / "preparation_summary.json", summary)
    print(json.dumps({
        "status": "COMPLETED", "N": len(features),
        "split_counts": summary["split_counts"], "validation_role_counts": summary["validation_role_counts"],
        "prepared_feature_counts": prepared_sets["counts"], "reload_verification": "PASS",
        "audit_output_dir": str(args.audit_output_dir), "interim_output_dir": str(args.interim_output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
