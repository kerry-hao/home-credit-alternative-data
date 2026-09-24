#!/usr/bin/env python3
"""Prepare compact aggregate CSVs for the Task 13 Streamlit showcase."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd


EXPECTED_FINAL = 228_999
FAMILIES = {
    "logit": ("logit_T", "logit_T_plus_AD"),
    "lightgbm": ("lightgbm_T", "lightgbm_T_plus_AD"),
    "mlp": ("mlp_T", "mlp_T_plus_AD"),
}
COPY_FILES = {
    "overall_model_metrics.csv": 6,
    "overall_incremental_gains.csv": 3,
    "subgroup_incremental_gains.csv": 45,
    "fixed_approval_rate_curve.csv": 966,
    "fixed_risk_budget_curve.csv": 756,
    "economic_incremental_value.csv": 18,
}
OUTPUT_FILES = set(COPY_FILES) | {"risk_migration_5_10_20.csv"}
MIGRATION_COLUMNS = [
    "algorithm_family", "t_model_id", "t_plus_ad_model_id", "n_bins",
    "t_risk_bin", "t_plus_ad_risk_bin", "case_count", "share_of_total",
    "share_within_t_bin", "target_default_count", "observed_default_rate",
    "bin_assignment",
]


class ShowcaseDataError(RuntimeError):
    """The aggregate showcase package cannot be produced safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path)
    source.add_argument("--task12-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/showcase",
    )
    return parser.parse_args()


def stable_risk_bins(probability: np.ndarray, base_order: np.ndarray, n_bins: int) -> np.ndarray:
    if len(probability) != len(base_order) or not np.isfinite(probability).all():
        raise ShowcaseDataError("Risk-bin inputs are misaligned or non-finite")
    order = np.lexsort((base_order, probability))
    bins = np.empty(len(order), dtype=np.int16)
    bins[order] = np.minimum(n_bins, np.arange(len(order), dtype=np.int64) * n_bins // len(order) + 1)
    return bins


def build_migration(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"case_id", "base_order", "target"} | {
        f"calibrated_pd__{model}" for pair in FAMILIES.values() for model in pair
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ShowcaseDataError(f"Final predictions are missing columns: {missing}")
    if len(predictions) != EXPECTED_FINAL or not predictions.case_id.is_unique:
        raise ShowcaseDataError("Final predictions must contain 228,999 unique cases")
    base_order = predictions.base_order.to_numpy(np.int64)
    target = predictions.target.to_numpy(np.int8)
    rows: list[dict[str, object]] = []
    for family, (t_model, tad_model) in FAMILIES.items():
        t_probability = predictions[f"calibrated_pd__{t_model}"].to_numpy(np.float64)
        tad_probability = predictions[f"calibrated_pd__{tad_model}"].to_numpy(np.float64)
        for n_bins in (5, 10, 20):
            t_bins = stable_risk_bins(t_probability, base_order, n_bins)
            tad_bins = stable_risk_bins(tad_probability, base_order, n_bins)
            origin_totals = np.bincount(t_bins, minlength=n_bins + 1)
            for t_bin in range(1, n_bins + 1):
                for tad_bin in range(1, n_bins + 1):
                    mask = (t_bins == t_bin) & (tad_bins == tad_bin)
                    count = int(mask.sum())
                    defaults = int(target[mask].sum())
                    rows.append({
                        "algorithm_family": family,
                        "t_model_id": t_model,
                        "t_plus_ad_model_id": tad_model,
                        "n_bins": n_bins,
                        "t_risk_bin": t_bin,
                        "t_plus_ad_risk_bin": tad_bin,
                        "case_count": count,
                        "share_of_total": count / len(predictions),
                        "share_within_t_bin": count / int(origin_totals[t_bin]),
                        "target_default_count": defaults,
                        "observed_default_rate": defaults / count if count else np.nan,
                        "bin_assignment": (
                            "separate global stable ranks by selected calibrated PD ascending, "
                            "then frozen base_order ascending; bin 1 lowest risk"
                        ),
                    })
    return pd.DataFrame(rows, columns=MIGRATION_COLUMNS)


def validate_migration(migration: pd.DataFrame, formal_quintiles: pd.DataFrame) -> None:
    expected_rows = 3 * (5 * 5 + 10 * 10 + 20 * 20)
    if len(migration) != expected_rows or list(migration.columns) != MIGRATION_COLUMNS:
        raise ShowcaseDataError(f"Migration shape mismatch: {migration.shape}")
    for (family, n_bins), group in migration.groupby(["algorithm_family", "n_bins"], sort=False):
        if len(group) != n_bins * n_bins or int(group.case_count.sum()) != EXPECTED_FINAL:
            raise ShowcaseDataError(f"Migration cells do not reconcile for {family}, K={n_bins}")
        row_sums = group.groupby("t_risk_bin", sort=False).share_within_t_bin.sum()
        if not np.allclose(row_sums, 1.0, rtol=0, atol=1e-12):
            raise ShowcaseDataError(f"Migration origin shares do not sum to one for {family}, K={n_bins}")
    if (migration[["case_count", "target_default_count"]].to_numpy() < 0).any():
        raise ShowcaseDataError("Migration contains negative counts")
    if (migration.target_default_count > migration.case_count).any():
        raise ShowcaseDataError("Migration default count exceeds its case count")
    shares = migration[["share_of_total", "share_within_t_bin"]].to_numpy(np.float64)
    if (shares < 0).any() or (shares > 1 + 1e-12).any():
        raise ShowcaseDataError("Migration shares fall outside [0, 1]")

    formal = formal_quintiles.loc[formal_quintiles.stratum_dimension.eq("overall"), [
        "algorithm_family", "t_risk_quintile", "t_plus_ad_risk_quintile", "case_count",
    ]].rename(columns={"t_risk_quintile": "t_risk_bin", "t_plus_ad_risk_quintile": "t_plus_ad_risk_bin"})
    generated = migration.loc[migration.n_bins.eq(5), [
        "algorithm_family", "t_risk_bin", "t_plus_ad_risk_bin", "case_count",
    ]]
    merged = formal.merge(
        generated,
        on=["algorithm_family", "t_risk_bin", "t_plus_ad_risk_bin"],
        suffixes=("_formal", "_generated"),
        validate="one_to_one",
    )
    if len(merged) != 75 or not (merged.case_count_formal == merged.case_count_generated).all():
        raise ShowcaseDataError("Generated K=5 migration does not match the formal quintile counts")


def validate_showcase_frames(frames: dict[str, pd.DataFrame]) -> None:
    if set(frames) != OUTPUT_FILES:
        raise ShowcaseDataError(f"Showcase inventory mismatch: {sorted(frames)}")
    for name, expected_rows in COPY_FILES.items():
        if len(frames[name]) != expected_rows:
            raise ShowcaseDataError(f"{name} has {len(frames[name])} rows; expected {expected_rows}")
    forbidden = {"case_id", "base_order", "target"}
    for name, frame in frames.items():
        overlap = sorted(forbidden & set(frame.columns))
        if overlap:
            raise ShowcaseDataError(f"{name} exposes row-level columns: {overlap}")
    economic = frames["economic_incremental_value.csv"]
    if set(economic.comparison_type) != {"within_family_ad"} or economic.duplicated(["algorithm_family", "scenario_id"]).any():
        raise ShowcaseDataError("Showcase economic data is not the repaired within-family 18-row table")


def prepare(task12_dir: Path, output_dir: Path) -> dict[str, object]:
    task12_dir = task12_dir.resolve()
    output_dir = output_dir.resolve()
    frames = {name: pd.read_csv(task12_dir / name) for name in COPY_FILES}
    predictions = pd.read_parquet(task12_dir / "final_evaluation_predictions.parquet")
    migration = build_migration(predictions)
    formal_quintiles = pd.read_csv(task12_dir / "risk_quintile_migration.csv")
    validate_migration(migration, formal_quintiles)
    frames["risk_migration_5_10_20.csv"] = migration
    validate_showcase_frames(frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for name, frame in frames.items():
            target = output_dir / name
            temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            frame.to_csv(temp, index=False, lineterminator="\n")
            staged.append((temp, target))
        for temp, target in staged:
            os.replace(temp, target)
    finally:
        for temp, _target in staged:
            if temp.exists():
                temp.unlink()
    unexpected = {path.name for path in output_dir.iterdir() if path.is_file()} - OUTPUT_FILES
    if unexpected:
        raise ShowcaseDataError(f"Unexpected files already present in showcase output: {sorted(unexpected)}")
    return {
        "status": "COMPLETE",
        "output_directory": str(output_dir),
        "row_counts": {name: len(frame) for name, frame in sorted(frames.items())},
        "migration_k5_matches_formal": True,
        "row_level_identifiers_written": False,
        "model_inference_performed": False,
    }


def main() -> int:
    args = parse_args()
    task12_dir = args.task12_dir or args.data_root / "final_evaluation/task12"
    result = prepare(task12_dir, args.output_dir)
    import json
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
