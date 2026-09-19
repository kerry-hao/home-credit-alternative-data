#!/usr/bin/env python3
"""Train the six prespecified Task 09 baseline model combinations.

The script consumes the frozen Task 08 follow-up matrices and saved membership.
It never regenerates preprocessing, predicts held-out roles, fits calibration, or
selects a decision threshold.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import random
import resource
import shutil
import signal
import sys
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

# Bound native pools before importing numerical libraries.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/home_credit_mpl")

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import torch
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from threadpoolctl import threadpool_info, threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# The installed macOS PyTorch/OpenMP combination segfaults when this mixed
# sklearn/LightGBM process executes the MLP at four intra-op threads.  One
# PyTorch intra-op thread is the verified stable setting and remains within the
# authorized maximum; LightGBM and BLAS retain the configurable outer limit.
torch.set_num_threads(1)


VERSION = "1.0.0"
SEED = 20260921
EXPECTED_N = 1_526_659
EXPECTED_TRAIN = 1_068_661
EXPECTED_TRAIN_POS = 33_596
EXPECTED_TUNING = 114_499
EXPECTED_TUNING_POS = 3_599
EXPECTED_MEMBERSHIP_FP = "8bfb238774655f46a4f21c2668b2815f321c9298cb6d38a8da33f35a1ab91714"
EXPECTED_TRAINING_KEY_FP = "bed8c0241e093a8bacedbfb0406960f2b5a7873db277ba6cf1ab7ce09b90736e"
MODEL_ORDER = ["logit_T", "logit_T_plus_AD", "lightgbm_T", "lightgbm_T_plus_AD", "mlp_T", "mlp_T_plus_AD"]
PROBABILITY_COLUMNS = {name: f"prob_{name}" for name in MODEL_ORDER}
METADATA_EXCLUSIONS = {
    "case_id", "base_order", "target", "date_decision", "WEEK_NUM", "MONTH",
    "outer_split", "validation_role",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or value is None:
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(values: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def log(message: str, log_path: Path | None = None) -> None:
    line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()


def runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(), "python_executable": sys.executable,
        "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__,
        "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__,
        "torch": torch.__version__, "joblib": joblib.__version__,
    }


def ru_maxrss_bytes() -> int:
    # macOS reports bytes; Linux reports KiB. This project is explicitly local macOS.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def configure_threads(threads: int) -> dict[str, Any]:
    effective = max(1, min(int(threads), os.cpu_count() or 1))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(effective)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(effective)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return {
        "requested_threads": threads, "effective_threads": effective,
        "torch_intraop": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
        "torch_thread_adjustment_reason": "Verified local mixed-library segfault at four PyTorch intra-op threads; using one without changing rows, batch size, architecture, objective, or epoch budget.",
        "environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "loaded_threadpools": threadpool_info(),
    }


def official_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("Metrics require equal-length one-dimensional target and probability arrays")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Metrics require both binary classes")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and inside [0, 1]")
    return {
        "N": int(len(y)), "positive_count": int(y.sum()), "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "mean_probability": float(p.mean()), "min_probability": float(p.min()),
        "max_probability": float(p.max()), "exact_zero_count": int(np.count_nonzero(p == 0)),
        "exact_one_count": int(np.count_nonzero(p == 1)),
    }


def select_candidate(rows: list[dict[str, Any]], family: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible = [r for r in rows if r.get("status") == "SUCCESS" and r.get("converged", True)]
    if not eligible:
        raise RuntimeError(f"No eligible candidates for {family}")
    max_ap = max(float(r["average_precision"]) for r in eligible)
    band = [r for r in eligible if float(r["average_precision"]) >= max_ap - 0.0001]
    if family == "logit":
        winner = min(band, key=lambda r: (float(r["log_loss"]), float(r["C"]), str(r["candidate_id"])))
    elif family == "lightgbm":
        winner = min(band, key=lambda r: (float(r["log_loss"]), int(r["num_leaves"]), str(r["candidate_id"])))
    else:
        winner = min(band, key=lambda r: (-float(r["average_precision"]), float(r["log_loss"]), int(r.get("best_epoch", 0))))
    annotated = []
    for row in rows:
        copy_row = dict(row)
        in_band = row in band
        copy_row["ap_band_threshold"] = max_ap - 0.0001
        copy_row["ap_band_eligible"] = bool(in_band)
        copy_row["selected"] = row is winner
        copy_row["selection_reason"] = (
            "SELECTED: within global max-AP band; lowest log loss; deterministic complexity/candidate tie-break"
            if row is winner else ("ELIGIBLE_NOT_SELECTED" if in_band else "OUTSIDE_GLOBAL_AP_BAND_OR_INELIGIBLE")
        )
        annotated.append(copy_row)
    return winner, annotated


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, output_bias: float | None = None):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1),
        )
        if output_bias is not None:
            nn.init.constant_(self.network[-1].bias, float(output_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def torch_load_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def predict_mlp(model: TabularMLP, values: np.ndarray, batch_size: int = 16384) -> np.ndarray:
    model.eval()
    result = np.empty(len(values), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            stop = min(len(values), start + batch_size)
            logits = model(torch.from_numpy(values[start:stop].astype(np.float32, copy=False)))
            result[start:stop] = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float64)
    return result


@dataclass
class Contract:
    manifest: pd.DataFrame
    train_positions: np.ndarray
    tuning_positions: np.ndarray
    train_labels: np.ndarray
    tuning_labels: np.ndarray
    feature_sets: dict[str, Any]
    preprocessor: dict[str, Any]
    membership_fingerprint: str
    training_key_fingerprint: str
    input_hashes: dict[str, str]


def load_contract(data_root: Path) -> Contract:
    prep = data_root / "interim/task08_followup"
    manifest_path = prep / "application_manifest.parquet"
    feature_sets_path = prep / "preprocessing/feature_sets.json"
    preprocessor_path = prep / "preprocessing/preprocessor.json"
    linear_path = prep / "linear_nn_inputs.parquet"
    gbdt_path = prep / "gbdt_inputs.parquet"
    for path in (manifest_path, feature_sets_path, preprocessor_path, linear_path, gbdt_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = pd.read_parquet(manifest_path, columns=["case_id", "base_order", "outer_split", "validation_role"])
    if len(manifest) != EXPECTED_N or manifest["case_id"].isna().any() or manifest["case_id"].duplicated().any():
        raise RuntimeError("Manifest population/key contract failed")
    if not np.array_equal(manifest["base_order"].to_numpy(np.int64), np.arange(EXPECTED_N, dtype=np.int64)):
        raise RuntimeError("Manifest base_order is not canonical original order")
    membership_fp = sha256_text(
        f"{cid}\t{outer}\t{role}" for cid, outer, role in zip(manifest.case_id, manifest.outer_split, manifest.validation_role)
    )
    train_positions = np.flatnonzero(manifest["outer_split"].eq("train").to_numpy())
    tuning_positions = np.flatnonzero(manifest["validation_role"].eq("validation_tuning").to_numpy())
    training_key_fp = sha256_text(manifest.iloc[train_positions]["case_id"])
    if membership_fp != EXPECTED_MEMBERSHIP_FP or training_key_fp != EXPECTED_TRAINING_KEY_FP:
        raise RuntimeError(f"Fingerprint mismatch: membership={membership_fp}, training={training_key_fp}")
    if len(train_positions) != EXPECTED_TRAIN or len(tuning_positions) != EXPECTED_TUNING:
        raise RuntimeError("Saved role counts do not match the frozen contract")
    # Predicate pushdown reads labels only for the two authorized development roles.
    train_labeled = pd.read_parquet(
        manifest_path, columns=["case_id", "base_order", "target", "outer_split"],
        filters=[("outer_split", "==", "train")],
    ).sort_values("base_order", kind="stable")
    tune_labeled = pd.read_parquet(
        manifest_path, columns=["case_id", "base_order", "target", "validation_role"],
        filters=[("validation_role", "==", "validation_tuning")],
    ).sort_values("base_order", kind="stable")
    if not np.array_equal(train_labeled.base_order.to_numpy(), train_positions) or not np.array_equal(tune_labeled.base_order.to_numpy(), tuning_positions):
        raise RuntimeError("Filtered label rows do not align with saved base_order")
    if not np.array_equal(train_labeled.case_id.to_numpy(), manifest.iloc[train_positions].case_id.to_numpy()) or not np.array_equal(tune_labeled.case_id.to_numpy(), manifest.iloc[tuning_positions].case_id.to_numpy()):
        raise RuntimeError("Filtered label keys do not align with manifest")
    train_labels = train_labeled.target.to_numpy(np.int8)
    tuning_labels = tune_labeled.target.to_numpy(np.int8)
    if set(np.unique(train_labels)) != {0, 1} or set(np.unique(tuning_labels)) != {0, 1}:
        raise RuntimeError("TRAIN/tuning labels are not complete binary classes")
    if int(train_labels.sum()) != EXPECTED_TRAIN_POS or int(tuning_labels.sum()) != EXPECTED_TUNING_POS:
        raise RuntimeError("TRAIN/tuning positive-count contract failed")
    feature_sets = json.loads(feature_sets_path.read_text(encoding="utf-8"))
    preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    for representation in ("linear_nn", "gbdt"):
        t = feature_sets[representation]["T"]
        tad = feature_sets[representation]["T_plus_AD"]
        if len(t) != 52 or len(tad) != 107 or tad[:52] != t or len(set(tad)) != 107:
            raise RuntimeError(f"Feature-list contract failed for {representation}")
        if METADATA_EXCLUSIONS & set(tad):
            raise RuntimeError(f"Metadata leakage in {representation}")
        matrix_path = linear_path if representation == "linear_nn" else gbdt_path
        if pq.ParquetFile(matrix_path).schema.names != ["case_id", *tad]:
            raise RuntimeError(f"Saved matrix schema/order disagrees with authoritative feature list: {representation}")
    input_paths = [manifest_path, linear_path, gbdt_path, preprocessor_path, feature_sets_path]
    return Contract(
        manifest=manifest, train_positions=train_positions, tuning_positions=tuning_positions,
        train_labels=train_labels, tuning_labels=tuning_labels, feature_sets=feature_sets,
        preprocessor=preprocessor, membership_fingerprint=membership_fp,
        training_key_fingerprint=training_key_fp,
        input_hashes={str(p): sha256_file(p) for p in input_paths},
    )


def load_matrix_rows(
    path: Path, columns: list[str], positions: np.ndarray, manifest_keys: np.ndarray,
    dtype: np.dtype, batch_size: int = 131072,
) -> tuple[np.ndarray, dict[str, Any]]:
    pf = pq.ParquetFile(path)
    expected_schema = ["case_id", *columns]
    if pf.schema.names[0] != "case_id" or any(column not in pf.schema.names for column in columns) or len(columns) != len(set(columns)):
        raise RuntimeError(f"Requested matrix columns are absent or duplicated: {path}")
    output = np.empty((len(positions), len(columns)), dtype=dtype, order="C")
    cursor = 0
    global_start = 0
    any_inf = False
    any_nan = False
    for batch in pf.iter_batches(batch_size=batch_size, columns=expected_schema):
        frame = batch.to_pandas()
        n = len(frame)
        if not np.array_equal(frame["case_id"].to_numpy(), manifest_keys[global_start:global_start + n]):
            raise RuntimeError(f"Matrix/manifest key order mismatch at row {global_start}: {path}")
        left = np.searchsorted(positions, global_start, side="left")
        right = np.searchsorted(positions, global_start + n, side="left")
        if right > left:
            local = positions[left:right] - global_start
            values = frame.iloc[local][columns].to_numpy(dtype=dtype, copy=True)
            output[cursor:cursor + len(values)] = values
            any_inf |= bool(np.isinf(values).any())
            any_nan |= bool(np.isnan(values).any())
            cursor += len(values)
        global_start += n
    if cursor != len(positions) or global_start != len(manifest_keys):
        raise RuntimeError("Matrix scan did not reconcile row counts")
    return output, {
        "shape": list(output.shape), "dtype": str(output.dtype), "nbytes": int(output.nbytes),
        "contains_infinity": any_inf, "contains_nan": any_nan,
    }


def acquire_lock(path: Path, run_id: str, resume: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "run_id": run_id, "started_at": utc_now(), "command": sys.argv}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        pid = int(existing.get("pid", -1))
        live = False
        if pid > 0:
            try:
                os.kill(pid, 0); live = True
            except ProcessLookupError:
                live = False
            except PermissionError:
                live = True
        if live:
            raise RuntimeError(f"Run lock belongs to live PID {pid}: {path}")
        if not resume:
            raise RuntimeError(f"Stale lock exists; use --resume after review: {path}")
        stale = path.with_name(path.name + f".stale.{int(time.time())}")
        os.replace(path, stale)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream)


def release_lock(path: Path) -> None:
    if path.exists():
        path.unlink()


class Heartbeat:
    def __init__(self, message: str, log_path: Path, seconds: int = 30):
        self.message, self.log_path, self.seconds = message, log_path, seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(self.seconds):
            log(f"HEARTBEAT {self.message}; fit still running (no per-iteration validation metric available)", self.log_path)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set(); self.thread.join(timeout=2)


def parse_models(value: str) -> list[str]:
    if value == "all":
        return list(MODEL_ORDER)
    result = [item.strip() for item in value.split(",") if item.strip()]
    bad = [item for item in result if item not in MODEL_ORDER]
    if bad or not result:
        raise ValueError(f"Unknown model combinations: {bad}")
    return [name for name in MODEL_ORDER if name in result]


def feature_contract_for(name: str, feature_sets: dict[str, Any]) -> tuple[str, str, list[str]]:
    family, info = name.split("_", 1)
    if family == "lightgbm":
        representation = "gbdt"
    else:
        representation = "linear_nn"
    set_name = "T_plus_AD" if info == "T_plus_AD" else "T"
    return family, representation, list(feature_sets[representation][set_name])


def class_one_probability(model: Any, values: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise RuntimeError(f"Unexpected class order: {classes}")
    return model.predict_proba(values)[:, classes.index(1)].astype(np.float64)


def apply_preparation_report_corrections(data_root: Path) -> list[dict[str, Any]]:
    prep = data_root / "interim/task08_followup"
    audit = data_root / "audits/task08_followup"
    immutable = [
        prep / "application_manifest.parquet", prep / "linear_nn_inputs.parquet",
        prep / "gbdt_inputs.parquet", prep / "preprocessing/preprocessor.json",
    ]
    summary_path = audit / "preparation_summary.json"
    report_path = audit / "preparation_report.md"
    before = {str(path): sha256_file(path) for path in [*immutable, summary_path, report_path]}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["constant_train_columns_scope"] = (
        "45 substantive numeric columns only; binary, missing-indicator, and one-hot outputs were outside the historical check. "
        "Task 09 separately audits all prepared TRAIN predictors."
    )
    atomic_json(summary_path, summary)
    report = report_path.read_text(encoding="utf-8")
    scope_line = "- Scope correction: `constant_train_columns` covered only the 45 substantive numeric inputs; Task 09 separately audits constant binary, missing-indicator, and one-hot predictors."
    if scope_line not in report:
        anchor = "- Reload verification:"
        lines = report.splitlines()
        insertion = next((i + 1 for i, line in enumerate(lines) if line.startswith(anchor)), len(lines))
        lines.insert(insertion, scope_line)
        atomic_text(report_path, "\n".join(lines) + "\n")
    after = {str(path): sha256_file(path) for path in [*immutable, summary_path, report_path]}
    rows = []
    for path in [*immutable, summary_path, report_path]:
        rows.append({
            "path": str(path), "sha256_before": before[str(path)], "sha256_after": after[str(path)],
            "bytes_unchanged": before[str(path)] == after[str(path)],
            "expected_change": path in {summary_path, report_path},
            "status": "PASS" if ((path in {summary_path, report_path}) or before[str(path)] == after[str(path)]) else "FAIL",
            "reason": "Scope annotation/prose correction" if path in {summary_path, report_path} else "Prepared matrix/preprocessor must remain byte-identical",
        })
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("Prepared matrix/preprocessor changed during bounded report corrections")
    return rows


def constant_input_audit(data_root: Path, contract: Contract) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    substantive = set(contract.preprocessor["numeric_substantive"])
    original_binary = set(contract.preprocessor["ad_indicators"])
    keys = contract.manifest.case_id.to_numpy()
    for representation, filename in (("linear_nn", "linear_nn_inputs.parquet"), ("gbdt", "gbdt_inputs.parquet")):
        columns = contract.feature_sets[representation]["T_plus_AD"]
        values, _ = load_matrix_rows(
            data_root / f"interim/task08_followup/{filename}", columns,
            contract.train_positions, keys, np.float32,
        )
        for j, column in enumerate(columns):
            vector = values[:, j]
            finite = vector[np.isfinite(vector)]
            all_missing = len(finite) == 0
            constant = all_missing or (len(finite) == len(vector) and float(finite.min()) == float(finite.max()))
            if not constant:
                continue
            if column in substantive:
                kind = "substantive_numeric"
            elif column in original_binary:
                kind = "original_AD_indicator"
            elif column.startswith("missing__"):
                kind = "missing_indicator"
            elif column.startswith("ohe__"):
                kind = "one_hot"
            else:
                kind = "other_prepared"
            value = None if all_missing else float(finite[0])
            explanation = (
                "Unexpected substantive constant; requires diagnosis"
                if kind == "substantive_numeric" else
                "Expected possible reserved/indicator constant; retained by frozen input contract"
            )
            rows.append({
                "representation": representation, "prepared_column": column, "prepared_type": kind,
                "train_rows": len(vector), "finite_count": int(len(finite)), "missing_count": int(np.isnan(vector).sum()),
                "all_missing": all_missing, "constant_value": value, "status": "REVIEW" if kind == "substantive_numeric" else "INFORMATIONAL",
                "explanation": explanation,
            })
        del values; gc.collect()
    if any(row["prepared_type"] == "substantive_numeric" for row in rows):
        raise RuntimeError("Unexpected constant substantive numeric predictor found")
    return rows


def standardized_value_audit(data_root: Path, contract: Contract) -> list[dict[str, Any]]:
    numeric = list(contract.preprocessor["numeric_substantive"])
    flag_items = contract.preprocessor["missing_flags"]
    dependency_to_flag = {dep: item["name"] for item in flag_items for dep in item["dependencies"]}
    flags = sorted(set(dependency_to_flag.values()))
    columns = [*numeric, *flags]
    path = data_root / "interim/task08_followup/linear_nn_inputs.parquet"
    keys = contract.manifest.case_id.to_numpy()
    rows: list[dict[str, Any]] = []
    for role, positions in (("train", contract.train_positions), ("validation_tuning", contract.tuning_positions)):
        matrix, _ = load_matrix_rows(path, columns, positions, keys, np.float32)
        index = {column: j for j, column in enumerate(columns)}
        for column in numeric:
            vector = matrix[:, index[column]].astype(np.float64, copy=False)
            flag = dependency_to_flag.get(column)
            slices: list[tuple[str, np.ndarray, str]] = [("all_rows_after_imputation", np.ones(len(vector), bool), "all role rows")]
            if flag:
                observed_mask = matrix[:, index[flag]] == 0
                slices.append(("originally_observed_content", observed_mask, f"{flag}=0; shared flags may serve mean/max pairs"))
            else:
                slices.append(("originally_observed_content", np.ones(len(vector), bool), "No TRAIN missing flag: field was complete in TRAIN"))
            for slice_name, mask, definition in slices:
                selected = vector[mask]
                if not len(selected):
                    rows.append({
                        "role": role, "input_column": column, "slice": slice_name,
                        "status": "NOT_CHECKED", "denominator": 0, "denominator_definition": definition,
                        "reason": "No rows in derived observed-content slice",
                    })
                    continue
                quantiles = np.quantile(selected, [0.001, 0.01, 0.5, 0.99, 0.999], method="linear")
                rows.append({
                    "role": role, "input_column": column, "slice": slice_name, "status": "MEASURED",
                    "denominator": int(len(selected)), "denominator_definition": definition,
                    "finite_count": int(np.isfinite(selected).sum()), "min": float(selected.min()), "max": float(selected.max()),
                    "q0001": float(quantiles[0]), "q001": float(quantiles[1]), "median": float(quantiles[2]),
                    "q099": float(quantiles[3]), "q0999": float(quantiles[4]),
                    "abs_gt_10_count": int(np.count_nonzero(np.abs(selected) > 10)),
                    "abs_gt_20_count": int(np.count_nonzero(np.abs(selected) > 20)),
                    "quantile_method": "linear", "reason": "",
                })
        del matrix; gc.collect()
    return rows


def preprocessing_examples_supplement(data_root: Path, contract: Contract) -> list[dict[str, Any]]:
    feature_path = data_root / "interim/task08/application_features.parquet"
    linear_path = data_root / "interim/task08_followup/linear_nn_inputs.parquet"
    gbdt_path = data_root / "interim/task08_followup/gbdt_inputs.parquet"
    source_columns = [
        "case_id", "t__recorded_primary_income", "t__current_debt",
        "ad__deposit__amount_416A__finite_mean", "ad__deposit__amtdepositbalance_4809441A",
        "t__credit_product_type", "t__applicant_income_type",
    ]
    source = pd.read_parquet(feature_path, columns=source_columns)
    dev_mask = contract.manifest["outer_split"].eq("train") | contract.manifest["validation_role"].eq("validation_tuning")
    candidate_positions: list[tuple[str, int, str]] = []
    rules = [
        ("real_missing_T", source["t__recorded_primary_income"].isna().to_numpy() & dev_mask.to_numpy(), "t__recorded_primary_income"),
        ("real_missing_AD", source["ad__deposit__amount_416A__finite_mean"].isna().to_numpy() & dev_mask.to_numpy(), "ad__deposit__amount_416A__finite_mean"),
        ("real_observed_zero_T", source["t__current_debt"].eq(0).to_numpy() & dev_mask.to_numpy(), "t__current_debt"),
        ("real_observed_zero_banking_AD", source["ad__deposit__amount_416A__finite_mean"].eq(0).to_numpy() & dev_mask.to_numpy(), "ad__deposit__amount_416A__finite_mean"),
    ]
    for example_type, mask, column in rules:
        positions = np.flatnonzero(mask)
        if len(positions):
            candidate_positions.append((example_type, int(positions[0]), column))
        elif example_type == "real_observed_zero_banking_AD":
            fallback = np.flatnonzero(source["ad__deposit__amtdepositbalance_4809441A"].eq(0).to_numpy() & dev_mask.to_numpy())
            if len(fallback):
                candidate_positions.append((example_type, int(fallback[0]), "ad__deposit__amtdepositbalance_4809441A"))
    numeric_columns = sorted({column for _, _, column in candidate_positions})
    needed_flags = {}
    for item in contract.preprocessor["missing_flags"]:
        for dep in item["dependencies"]:
            if dep in numeric_columns:
                needed_flags[dep] = item["name"]
    prepared_columns = [*numeric_columns, *sorted(set(needed_flags.values()))]
    prepared_linear = pd.read_parquet(linear_path, columns=["case_id", *prepared_columns])
    prepared_gbdt = pd.read_parquet(gbdt_path, columns=["case_id", *prepared_columns])
    rows: list[dict[str, Any]] = []
    for example_type, pos, column in candidate_positions:
        raw = source.at[pos, column]
        missing = pd.isna(raw)
        median = contract.preprocessor["medians"][column]
        mean = contract.preprocessor["scaler_means"][column]
        scale = contract.preprocessor["scaler_scales"][column]
        is_log = column in set(contract.preprocessor["monetary"] + contract.preprocessor["dpd"])
        transformed = None if missing else (math.log1p(float(raw)) if is_log else float(raw))
        filled = float(median) if missing else float(transformed)
        expected_linear = (filled - float(mean)) / float(scale)
        flag = needed_flags.get(column)
        saved_linear = float(prepared_linear.at[pos, column])
        saved_gbdt = prepared_gbdt.at[pos, column]
        flag_value = int(prepared_linear.at[pos, flag]) if flag else 0
        expected_gbdt = None if missing else float(raw)
        status = (
            np.isclose(saved_linear, expected_linear, atol=1e-6, rtol=0)
            and ((missing and pd.isna(saved_gbdt)) or (not missing and float(saved_gbdt) == expected_gbdt))
            and flag_value == int(missing)
        )
        rows.append({
            "example_type": example_type, "case_id_or_synthetic_id": int(source.at[pos, "case_id"]),
            "input_column": column, "original_value": raw, "original_state": "MISSING" if missing else ("OBSERVED_ZERO" if float(raw) == 0 else "OBSERVED"),
            "log_or_raw_stage": transformed, "imputed_stage": filled, "scaler_mean": mean, "scaler_scale": scale,
            "missing_flag_name": flag or "NO_TRAIN_MISSING_FLAG", "missing_flag_value": flag_value,
            "independent_expected_linear": expected_linear, "saved_linear": saved_linear,
            "independent_expected_gbdt": expected_gbdt, "saved_gbdt": saved_gbdt,
            "comparison_status": "PASS" if status else "FAIL",
            "note": "Observed zero is not median replacement" if not missing else "Median is computational placeholder, not inferred true value",
        })
    # Known real category from TRAIN/tuning.
    known_pos = int(np.flatnonzero(dev_mask.to_numpy())[0])
    for example_type, pos, column, synthetic_value in [
        ("real_known_category", known_pos, "t__credit_product_type", None),
    ]:
        value = str(source.at[pos, column])
        mapping = contract.preprocessor["category_outputs"][column]
        out = next(item["output"] for item in mapping if item["category"] == value)
        saved = int(pd.read_parquet(linear_path, columns=[out]).at[pos, out])
        rows.append({
            "example_type": example_type, "case_id_or_synthetic_id": int(source.at[pos, "case_id"]),
            "input_column": column, "original_value": value, "original_state": "KNOWN_CATEGORY",
            "relevant_output_name": out, "independent_expected_output": 1, "saved_output": saved,
            "comparison_status": "PASS" if saved == 1 else "FAIL", "note": "TRAIN vocabulary mapping",
        })
    # Precise lookup of the previously documented evaluation-only value; no label is read.
    eval_unseen = np.flatnonzero(
        source["t__applicant_income_type"].eq("HANDICAPPED").to_numpy()
        & contract.manifest["outer_split"].eq("evaluation").to_numpy()
    )
    if len(eval_unseen) == 1:
        pos = int(eval_unseen[0]); column = "t__applicant_income_type"
        out = next(item["output"] for item in contract.preprocessor["category_outputs"][column] if item["category"] == "__UNSEEN__")
        saved = int(pd.read_parquet(linear_path, columns=[out]).at[pos, out])
        rows.append({
            "example_type": "real_previously_documented_evaluation_only_category", "case_id_or_synthetic_id": int(source.at[pos, "case_id"]),
            "input_column": column, "original_value": "HANDICAPPED", "original_state": "UNSEEN_IN_TRAIN",
            "relevant_output_name": out, "independent_expected_output": 1, "saved_output": saved,
            "comparison_status": "PASS" if saved == 1 else "FAIL",
            "note": "Feature-only documentation exception; evaluation label and model metrics were not read.",
        })
    else:
        rows.append({
            "example_type": "synthetic_transform_only_unseen_category", "case_id_or_synthetic_id": "SYNTHETIC_UNSEEN",
            "input_column": "t__applicant_income_type", "original_value": "SYNTHETIC_OUT_OF_VOCABULARY",
            "original_state": "SYNTHETIC_UNSEEN", "independent_expected_output": 1, "comparison_status": "PASS",
            "note": "Prior real evaluation-only example was not uniquely recovered; no real ID is claimed.",
        })
    missing_out = next(item["output"] for item in contract.preprocessor["category_outputs"]["t__credit_product_type"] if item["category"] == "__MISSING__")
    rows.append({
        "example_type": "synthetic_transform_only_missing_category", "case_id_or_synthetic_id": "SYNTHETIC_MISSING",
        "input_column": "t__credit_product_type", "original_value": None, "original_state": "SYNTHETIC_MISSING",
        "relevant_output_name": missing_out, "independent_expected_output": 1, "saved_output": None,
        "comparison_status": "PASS", "note": "No actual TRAIN categorical missing value was claimed; mapping-only fixture.",
    })
    if any(row["comparison_status"] != "PASS" for row in rows):
        raise RuntimeError("Preprocessing example supplement comparison failed")
    return rows


def save_joblib_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    joblib.dump(value, temp)
    os.replace(temp, path)


def save_torch_atomic(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(state, temp)
    os.replace(temp, path)


def save_booster_atomic(path: Path, booster: lgb.Booster, num_iteration: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    booster.save_model(temp, num_iteration=num_iteration)
    os.replace(temp, path)


def base_candidate_row(combination: str, family: str, feature_set: str, features: int) -> dict[str, Any]:
    return {
        "combination": combination, "family": family, "feature_set": feature_set,
        "train_rows": EXPECTED_TRAIN, "tuning_rows": EXPECTED_TUNING, "feature_count": features,
        "fit_scope": "outer_train", "selection_scope": "validation_tuning",
        "objective_weighting": "natural_prevalence_unweighted",
    }


def run_logit_combination(
    combination: str, features: list[str], x_train: np.ndarray, x_tune: np.ndarray,
    y_train: np.ndarray, y_tune: np.ndarray, train_meta: pd.DataFrame, tune_meta: pd.DataFrame,
    model_dir: Path, interim_dir: Path, log_path: Path, threads: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models: dict[str, LogisticRegression] = {}
    sample_x = np.ascontiguousarray(np.vstack([x_train[:32], x_tune[:32]]), dtype=np.float64)
    sample_meta = pd.concat([train_meta.iloc[:32], tune_meta.iloc[:32]], ignore_index=True)
    for c_value in (0.01, 0.1, 1.0):
        candidate_id = f"C_{str(c_value).replace('.', 'p')}"
        candidate_dir = model_dir / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        attempt = 1
        max_iter = 1000
        final_row = None
        while attempt <= 2:
            row = {
                **base_candidate_row(combination, "logit", "T_plus_AD" if combination.endswith("T_plus_AD") else "T", len(features)),
                "candidate_id": candidate_id, "attempt": attempt,
                "retry_of": candidate_id if attempt == 2 else "", "C": c_value,
                "solver": "lbfgs", "l1_ratio": 0.0, "max_iter": max_iter, "tol": 1e-5,
            }
            started = time.perf_counter(); start_rss = ru_maxrss_bytes()
            estimator = LogisticRegression(
                C=c_value, l1_ratio=0.0, solver="lbfgs", fit_intercept=True,
                class_weight=None, max_iter=max_iter, tol=1e-5, warm_start=False,
            )
            log(f"START {combination} {candidate_id} attempt={attempt} rows={len(x_train)} features={len(features)}", log_path)
            captured: list[warnings.WarningMessage]
            try:
                with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=threads), Heartbeat(f"{combination} {candidate_id}", log_path):
                    warnings.simplefilter("always")
                    estimator.fit(x_train, y_train)
                convergence = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
                other_warnings = [str(w.message) for w in captured if not issubclass(w.category, ConvergenceWarning)]
                p = class_one_probability(estimator, x_tune)
                metrics = official_metrics(y_tune, p)
                finite_parameters = bool(np.isfinite(estimator.coef_).all() and np.isfinite(estimator.intercept_).all())
                converged = not convergence and int(estimator.n_iter_[0]) < max_iter
                row.update(metrics)
                row.update({
                    "status": "SUCCESS" if finite_parameters else "FAILED", "converged": converged,
                    "iterations": int(estimator.n_iter_[0]), "convergence_warnings": " | ".join(convergence),
                    "other_warnings": " | ".join(other_warnings), "finite_parameters": finite_parameters,
                    "elapsed_seconds": time.perf_counter() - started,
                    "ru_maxrss_bytes_end": ru_maxrss_bytes(), "ru_maxrss_bytes_start": start_rss,
                    "budget_limited": not converged,
                })
                checkpoint = candidate_dir / f"attempt_{attempt}.joblib"
                sample_expected = class_one_probability(estimator, sample_x)
                save_joblib_atomic(checkpoint, estimator)
                row["checkpoint_path"] = str(checkpoint)
                row["checkpoint_sha256"] = sha256_file(checkpoint)
                row["sample_expected_probability_json"] = json.dumps(sample_expected.tolist())
                atomic_json(candidate_dir / f"attempt_{attempt}_metrics.json", row)
                rows.append(row)
                if converged and finite_parameters:
                    models[candidate_id] = estimator
                    final_row = row
                    break
                if attempt == 1 and convergence:
                    attempt += 1; max_iter = 2000
                    del estimator; gc.collect(); continue
                final_row = row
                break
            except Exception as exc:
                row.update({
                    "status": "FAILED", "converged": False, "elapsed_seconds": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=20),
                })
                rows.append(row); atomic_json(candidate_dir / f"attempt_{attempt}_metrics.json", row)
                final_row = row; break
        log(f"END {combination} {candidate_id} status={final_row['status']} AP={final_row.get('average_precision')}", log_path)
    final_candidates = []
    for candidate_id in ("C_0p01", "C_0p1", "C_1p0"):
        attempts = [row for row in rows if row["candidate_id"] == candidate_id]
        final_candidates.append(attempts[-1])
    winner, annotated_final = select_candidate(final_candidates, "logit")
    annotations = {(r["candidate_id"], r["attempt"]): r for r in rows}
    for final in annotated_final:
        annotations[(final["candidate_id"], final["attempt"])].update({k: final[k] for k in ("ap_band_threshold", "ap_band_eligible", "selected", "selection_reason")})
    candidate_id = winner["candidate_id"]
    checkpoint = Path(winner["checkpoint_path"])
    selected_dir = model_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_path = selected_dir / "model.joblib"
    temp = selected_path.with_name(selected_path.name + f".tmp.{os.getpid()}")
    shutil.copy2(checkpoint, temp); os.replace(temp, selected_path)
    reloaded = joblib.load(selected_path)
    sample_before = np.array(json.loads(winner["sample_expected_probability_json"]), dtype=np.float64)
    sample_after = class_one_probability(reloaded, sample_x)
    max_diff = float(np.max(np.abs(sample_before - sample_after)))
    if max_diff > 1e-6:
        raise RuntimeError(f"Logit reload sample mismatch: {combination}, {max_diff}")
    full_probability = class_one_probability(reloaded, x_tune)
    selected_metrics = official_metrics(y_tune, full_probability)
    if max(abs(selected_metrics[k] - float(winner[k])) for k in ("roc_auc", "average_precision", "log_loss", "brier_score")) > 1e-10:
        raise RuntimeError("Logit reloaded full-tuning metrics differ from in-memory selection metrics")
    sample_frame = sample_meta.copy()
    for j, column in enumerate(features): sample_frame[column] = sample_x[:, j]
    sample_frame["expected_probability_before_serialization"] = sample_before
    atomic_parquet(interim_dir / "reload_samples" / f"{combination}.parquet", sample_frame)
    part = tune_meta.copy(); part["target"] = y_tune; part[PROBABILITY_COLUMNS[combination]] = full_probability
    atomic_parquet(interim_dir / "prediction_parts" / f"{combination}.parquet", part)
    registry = {
        "model_id": combination, "family": "logit", "feature_set": winner["feature_set"],
        "ordered_predictors": features, "model_path": str(selected_path), "model_sha256": sha256_file(selected_path),
        "config": {"C": winner["C"], "solver": "lbfgs", "l1_ratio": 0.0, "tol": 1e-5, "max_iter": winner["max_iter"]},
        "selected_iteration_or_epoch": int(winner["iterations"]), "class_order": [0, 1],
        "reload_sample_max_abs_difference": max_diff,
    }
    comparison = {**winner, **selected_metrics, "model_path": str(selected_path), "reload_max_abs_difference": max_diff}
    return comparison, list(annotations.values()), [], registry


def run_lightgbm_combination(
    combination: str, features: list[str], x_train: np.ndarray, x_tune: np.ndarray,
    y_train: np.ndarray, y_tune: np.ndarray, train_meta: pd.DataFrame, tune_meta: pd.DataFrame,
    model_dir: Path, interim_dir: Path, log_path: Path, threads: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates = [("compact", 15, 500), ("moderate", 31, 200)]
    rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    sample_x = np.ascontiguousarray(np.vstack([x_train[:32], x_tune[:32]]), dtype=np.float32)
    sample_meta = pd.concat([train_meta.iloc[:32], tune_meta.iloc[:32]], ignore_index=True)
    for candidate_id, leaves, min_child in candidates:
        candidate_dir = model_dir / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        row = {
            **base_candidate_row(combination, "lightgbm", "T_plus_AD" if combination.endswith("T_plus_AD") else "T", len(features)),
            "candidate_id": candidate_id, "attempt": 1, "num_leaves": leaves,
            "min_child_samples": min_child, "n_estimators": 1500,
        }
        started = time.perf_counter(); start_rss = ru_maxrss_bytes(); evaluation: dict[str, Any] = {}
        estimator = LGBMClassifier(
            objective="binary", boosting_type="gbdt", n_estimators=1500, learning_rate=0.05,
            max_depth=-1, max_bin=127, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=5.0, n_jobs=threads,
            random_state=SEED, deterministic=True, force_col_wise=True,
            use_missing=True, zero_as_missing=False, class_weight=None, metric="None",
            num_leaves=leaves, min_child_samples=min_child, verbosity=-1,
        )
        log(f"START {combination} {candidate_id} rows={len(x_train)} features={len(features)}", log_path)
        try:
            callbacks = [
                lgb.early_stopping(stopping_rounds=75, first_metric_only=True, min_delta=0.0, verbose=True),
                lgb.record_evaluation(evaluation), lgb.log_evaluation(period=25),
            ]
            estimator.fit(
                x_train, y_train, eval_X=x_tune, eval_y=y_tune,
                eval_names=["validation_tuning"],
                eval_metric=["average_precision", "auc", "binary_logloss"], callbacks=callbacks,
            )
            metric_order = list(evaluation["validation_tuning"].keys())
            if metric_order[:3] != ["average_precision", "auc", "binary_logloss"]:
                raise RuntimeError(f"Unexpected LightGBM metric order: {metric_order}")
            best_iteration = int(estimator.best_iteration_)
            if best_iteration <= 0:
                raise RuntimeError(f"Invalid best_iteration_: {best_iteration}")
            p = estimator.predict_proba(x_tune, num_iteration=best_iteration)[:, list(estimator.classes_).index(1)].astype(np.float64)
            metrics = official_metrics(y_tune, p)
            native_ap = float(evaluation["validation_tuning"]["average_precision"][best_iteration - 1])
            ap_diff = abs(native_ap - metrics["average_precision"])
            if ap_diff > 1e-10:
                raise RuntimeError(f"LightGBM native/sklearn AP mismatch: {ap_diff}")
            rounds_executed = len(evaluation["validation_tuning"]["average_precision"])
            sample_expected = estimator.predict_proba(sample_x, num_iteration=best_iteration)[:, list(estimator.classes_).index(1)].astype(np.float64)
            checkpoint = candidate_dir / "model.txt"
            save_booster_atomic(checkpoint, estimator.booster_, best_iteration)
            row.update(metrics)
            row.update({
                "status": "SUCCESS", "converged": True, "best_iteration": best_iteration,
                "rounds_executed": rounds_executed, "budget_limited": rounds_executed >= 1500,
                "native_ap": native_ap, "native_sklearn_ap_abs_difference": ap_diff,
                "callback_metric_order": json.dumps(metric_order), "early_stopping_metric": "average_precision",
                "lightgbm_validation_api": "eval_X/eval_y", "elapsed_seconds": time.perf_counter() - started,
                "ru_maxrss_bytes_start": start_rss, "ru_maxrss_bytes_end": ru_maxrss_bytes(),
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                "sample_expected_probability_json": json.dumps(sample_expected.tolist()),
            })
            for metric_name, values in evaluation["validation_tuning"].items():
                for iteration, value in enumerate(values, start=1):
                    history_rows.append({
                        "combination": combination, "family": "lightgbm", "candidate_id": candidate_id,
                        "unit": "boosting_round", "step": iteration, "metric": metric_name,
                        "value": float(value), "selected_checkpoint": iteration == best_iteration,
                    })
            atomic_json(candidate_dir / "metrics.json", row); rows.append(row)
        except Exception as exc:
            row.update({"status": "FAILED", "converged": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=20), "elapsed_seconds": time.perf_counter() - started})
            rows.append(row); atomic_json(candidate_dir / "metrics.json", row)
        log(f"END {combination} {candidate_id} status={row['status']} AP={row.get('average_precision')}", log_path)
        del estimator; gc.collect()
    winner, annotated = select_candidate(rows, "lightgbm")
    checkpoint = Path(winner["checkpoint_path"])
    selected_dir = model_dir / "selected"; selected_dir.mkdir(parents=True, exist_ok=True)
    selected_path = selected_dir / "model.txt"
    temp = selected_path.with_name(selected_path.name + f".tmp.{os.getpid()}")
    shutil.copy2(checkpoint, temp); os.replace(temp, selected_path)
    booster = lgb.Booster(model_file=str(selected_path))
    best_iteration = int(winner["best_iteration"])
    sample_before = np.array(json.loads(winner["sample_expected_probability_json"]), dtype=np.float64)
    sample_after = booster.predict(sample_x, num_iteration=best_iteration).astype(np.float64)
    max_diff = float(np.max(np.abs(sample_before - sample_after)))
    if max_diff > 1e-6:
        raise RuntimeError(f"LightGBM reload sample mismatch: {combination}, {max_diff}")
    full_probability = booster.predict(x_tune, num_iteration=best_iteration).astype(np.float64)
    selected_metrics = official_metrics(y_tune, full_probability)
    if max(abs(selected_metrics[k] - float(winner[k])) for k in ("roc_auc", "average_precision", "log_loss", "brier_score")) > 1e-10:
        raise RuntimeError("LightGBM reloaded metrics differ from selected candidate")
    sample_frame = sample_meta.copy()
    for j, column in enumerate(features): sample_frame[column] = sample_x[:, j]
    sample_frame["expected_probability_before_serialization"] = sample_before
    atomic_parquet(interim_dir / "reload_samples" / f"{combination}.parquet", sample_frame)
    part = tune_meta.copy(); part["target"] = y_tune; part[PROBABILITY_COLUMNS[combination]] = full_probability
    atomic_parquet(interim_dir / "prediction_parts" / f"{combination}.parquet", part)
    registry = {
        "model_id": combination, "family": "lightgbm", "feature_set": winner["feature_set"],
        "ordered_predictors": features, "model_path": str(selected_path), "model_sha256": sha256_file(selected_path),
        "config": {"num_leaves": winner["num_leaves"], "min_child_samples": winner["min_child_samples"], "learning_rate": 0.05, "zero_as_missing": False},
        "selected_iteration_or_epoch": best_iteration, "class_order": [0, 1],
        "reload_sample_max_abs_difference": max_diff,
    }
    comparison = {**winner, **selected_metrics, "model_path": str(selected_path), "reload_max_abs_difference": max_diff}
    return comparison, annotated, history_rows, registry


def run_mlp_combination(
    combination: str, features: list[str], x_train: np.ndarray, x_tune: np.ndarray,
    y_train: np.ndarray, y_tune: np.ndarray, train_meta: pd.DataFrame, tune_meta: pd.DataFrame,
    model_dir: Path, interim_dir: Path, log_path: Path, threads: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    positive_rate = float(y_train.mean())
    output_bias = math.log(positive_rate / (1.0 - positive_rate))
    model = TabularMLP(len(features), output_bias=output_bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(SEED)
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1)))
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, drop_last=False, num_workers=0, generator=generator)
    candidate_id = "mlp_64_32"
    candidate_dir = model_dir / "candidates" / candidate_id; candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = candidate_dir / "best_state.pt"
    history_rows: list[dict[str, Any]] = []
    best_ap = -math.inf; best_loss = math.inf; best_epoch = 0; patience_reference = -math.inf; patience = 0
    best_sample_expected: np.ndarray | None = None
    total_clipped = 0; maximum_grad_norm = 0.0; total_updates = 0
    sample_x = np.ascontiguousarray(np.vstack([x_train[:32], x_tune[:32]]), dtype=np.float32)
    sample_meta = pd.concat([train_meta.iloc[:32], tune_meta.iloc[:32]], ignore_index=True)
    started = time.perf_counter(); start_rss = ru_maxrss_bytes()
    log(f"START {combination} {candidate_id} rows={len(x_train)} features={len(features)}", log_path)
    status = "SUCCESS"; error = ""
    try:
        for epoch in range(1, 31):
            epoch_start = time.perf_counter(); model.train(); weighted_loss = 0.0; seen = 0
            epoch_clipped = 0; epoch_max_grad = 0.0
            for batch_index, (batch_x, batch_y) in enumerate(train_loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                if logits.shape != batch_y.shape:
                    raise RuntimeError(f"MLP shape mismatch: logits={logits.shape}, target={batch_y.shape}")
                loss = loss_fn(logits, batch_y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite batch loss epoch={epoch} batch={batch_index}")
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True))
                epoch_max_grad = max(epoch_max_grad, grad_norm); maximum_grad_norm = max(maximum_grad_norm, grad_norm)
                if grad_norm > 5.0: epoch_clipped += 1; total_clipped += 1
                optimizer.step(); total_updates += 1
                if batch_index % 50 == 0:
                    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                        raise FloatingPointError(f"Nonfinite MLP parameter epoch={epoch} batch={batch_index}")
                weighted_loss += float(loss.item()) * len(batch_x); seen += len(batch_x)
                if time.perf_counter() - epoch_start > 30 and batch_index % 100 == 0:
                    log(f"PROGRESS {combination} epoch={epoch} batch={batch_index}/{len(train_loader)}", log_path)
            train_loss = weighted_loss / seen
            probability = predict_mlp(model, x_tune)
            metrics = official_metrics(y_tune, probability)
            improved_best = metrics["average_precision"] > best_ap or (
                metrics["average_precision"] == best_ap and (metrics["log_loss"] < best_loss or (metrics["log_loss"] == best_loss and epoch < best_epoch))
            )
            if improved_best:
                best_ap = metrics["average_precision"]; best_loss = metrics["log_loss"]; best_epoch = epoch
                model.eval(); best_sample_expected = predict_mlp(model, sample_x)
                frozen = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                save_torch_atomic(checkpoint, frozen)
            if metrics["average_precision"] >= patience_reference + 0.0001:
                patience_reference = metrics["average_precision"]; patience = 0
            else:
                patience += 1
            epoch_values = {
                "train_bce_loss": train_loss, "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"], "log_loss": metrics["log_loss"], "brier_score": metrics["brier_score"],
                "learning_rate": optimizer.param_groups[0]["lr"], "gradient_clip_count": epoch_clipped,
                "max_preclip_gradient_norm": epoch_max_grad, "patience_counter": patience,
            }
            for metric_name, value in epoch_values.items():
                history_rows.append({
                    "combination": combination, "family": "mlp", "candidate_id": candidate_id,
                    "unit": "epoch", "step": epoch, "metric": metric_name, "value": float(value),
                    "selected_checkpoint": False,
                })
            log(
                f"EPOCH {combination} {epoch} train_loss={train_loss:.8f} tuning_AP={metrics['average_precision']:.8f} "
                f"log_loss={metrics['log_loss']:.8f} best_epoch={best_epoch} patience={patience} elapsed={time.perf_counter()-epoch_start:.1f}s",
                log_path,
            )
            if patience >= 5:
                break
        if not checkpoint.is_file() or best_sample_expected is None:
            raise RuntimeError("MLP did not produce a valid best checkpoint")
    except Exception as exc:
        status = "FAILED"; error = f"{type(exc).__name__}: {exc}"
        atomic_text(candidate_dir / "failure_traceback.txt", traceback.format_exc())
    epochs_executed = max([int(r["step"]) for r in history_rows if r["metric"] == "average_precision"], default=0)
    if status != "SUCCESS":
        row = {**base_candidate_row(combination, "mlp", "T_plus_AD" if combination.endswith("T_plus_AD") else "T", len(features)),
               "candidate_id": candidate_id, "attempt": 1, "status": status, "converged": False, "error": error,
               "elapsed_seconds": time.perf_counter()-started}
        atomic_json(candidate_dir / "metrics.json", row)
        raise RuntimeError(error)
    selected_dir = model_dir / "selected"; selected_dir.mkdir(parents=True, exist_ok=True)
    selected_path = selected_dir / "state_dict.pt"
    temp = selected_path.with_name(selected_path.name + f".tmp.{os.getpid()}")
    shutil.copy2(checkpoint, temp); os.replace(temp, selected_path)
    architecture = {
        "input_dim": len(features), "hidden_dims": [64, 32], "activation": "ReLU", "dropout": 0.1,
        "output_dim": 1, "output": "logit", "dtype": "float32", "output_bias_initialization": output_bias,
    }
    atomic_json(selected_dir / "architecture.json", architecture)
    reloaded = TabularMLP(len(features), output_bias=None)
    reloaded.load_state_dict(torch_load_state(selected_path)); reloaded.eval()
    sample_after = predict_mlp(reloaded, sample_x)
    max_diff = float(np.max(np.abs(best_sample_expected - sample_after)))
    if max_diff > 1e-6:
        raise RuntimeError(f"MLP reload sample mismatch: {combination}, {max_diff}")
    full_probability = predict_mlp(reloaded, x_tune)
    selected_metrics = official_metrics(y_tune, full_probability)
    for row in history_rows:
        if row["step"] == best_epoch: row["selected_checkpoint"] = True
    candidate_row = {
        **base_candidate_row(combination, "mlp", "T_plus_AD" if combination.endswith("T_plus_AD") else "T", len(features)),
        "candidate_id": candidate_id, "attempt": 1, "status": "SUCCESS", "converged": True,
        "selected": True, "selection_reason": "Only prespecified MLP configuration; highest exact tuning-AP epoch selected",
        "best_epoch": best_epoch, "epochs_executed": epochs_executed, "budget_limited": epochs_executed >= 30,
        "learning_rate": 0.0003, "weight_decay": 0.0001, "batch_size": 4096,
        "gradient_clip_count": total_clipped, "maximum_preclip_gradient_norm": maximum_grad_norm,
        "optimizer_updates": total_updates, "deterministic_algorithms": True, "output_bias_initialization": output_bias,
        "elapsed_seconds": time.perf_counter()-started, "ru_maxrss_bytes_start": start_rss, "ru_maxrss_bytes_end": ru_maxrss_bytes(),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), **selected_metrics,
    }
    atomic_json(candidate_dir / "metrics.json", candidate_row)
    sample_frame = sample_meta.copy()
    for j, column in enumerate(features): sample_frame[column] = sample_x[:, j]
    sample_frame["expected_probability_before_serialization"] = best_sample_expected
    atomic_parquet(interim_dir / "reload_samples" / f"{combination}.parquet", sample_frame)
    part = tune_meta.copy(); part["target"] = y_tune; part[PROBABILITY_COLUMNS[combination]] = full_probability
    atomic_parquet(interim_dir / "prediction_parts" / f"{combination}.parquet", part)
    registry = {
        "model_id": combination, "family": "mlp", "feature_set": candidate_row["feature_set"],
        "ordered_predictors": features, "model_path": str(selected_path), "model_sha256": sha256_file(selected_path),
        "architecture_path": str(selected_dir / "architecture.json"), "config": architecture,
        "selected_iteration_or_epoch": best_epoch, "class_order": [0, 1],
        "reload_sample_max_abs_difference": max_diff,
    }
    comparison = {**candidate_row, "model_path": str(selected_path), "reload_max_abs_difference": max_diff}
    log(f"END {combination} status=SUCCESS AP={selected_metrics['average_precision']:.8f} best_epoch={best_epoch}", log_path)
    return comparison, [candidate_row], history_rows, registry


def combination_is_complete(status_path: Path, config_hash: str) -> bool:
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE" or status.get("config_hash") != config_hash:
        return False
    for item in status.get("artifacts", []):
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            return False
    return True


def load_completed(status_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return status["comparison"], status["candidates"], status["history"], status["registry"]


def persist_combination_status(
    path: Path, config_hash: str, comparison: dict[str, Any], candidates: list[dict[str, Any]],
    history: list[dict[str, Any]], registry: dict[str, Any], prediction_part: Path,
) -> None:
    artifacts = [
        {"path": registry["model_path"], "sha256": sha256_file(Path(registry["model_path"]))},
        {"path": str(prediction_part), "sha256": sha256_file(prediction_part)},
    ]
    if registry.get("architecture_path"):
        artifacts.append({"path": registry["architecture_path"], "sha256": sha256_file(Path(registry["architecture_path"]))})
    atomic_json(path, {
        "status": "COMPLETE", "completed_at": utc_now(), "config_hash": config_hash,
        "comparison": comparison, "candidates": candidates, "history": history,
        "registry": registry, "artifacts": artifacts,
    })


def rebuild_predictions(
    interim_dir: Path, selected: list[str], tune_meta: pd.DataFrame, y_tune: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    merged = tune_meta.copy(); merged["target"] = y_tune
    audits: list[dict[str, Any]] = []
    for combination in selected:
        path = interim_dir / "prediction_parts" / f"{combination}.parquet"
        part = pd.read_parquet(path)
        prob_col = PROBABILITY_COLUMNS[combination]
        key_ok = np.array_equal(part.case_id.to_numpy(), tune_meta.case_id.to_numpy())
        order_ok = np.array_equal(part.base_order.to_numpy(), tune_meta.base_order.to_numpy())
        target_ok = np.array_equal(part.target.to_numpy(np.int8), y_tune)
        finite = bool(np.isfinite(part[prob_col].to_numpy(np.float64)).all())
        bounds = bool(part[prob_col].between(0, 1, inclusive="both").all())
        complete = bool(part[prob_col].notna().all())
        status = "PASS" if len(part) == EXPECTED_TUNING and key_ok and order_ok and target_ok and finite and bounds and complete and part.case_id.is_unique else "FAIL"
        audits.append({
            "model_id": combination, "check": "prediction_part_contract", "status": status,
            "rows": len(part), "unique_keys": part.case_id.nunique(), "key_order_equal": key_ok,
            "base_order_equal": order_ok, "target_equal": target_ok, "finite": finite,
            "within_0_1": bounds, "complete_column": complete, "path": str(path),
        })
        if status != "PASS":
            raise RuntimeError(f"Prediction part verification failed: {combination}")
        merged[prob_col] = part[prob_col].to_numpy(np.float64)
    atomic_parquet(interim_dir / "tuning_predictions.parquet", merged)
    reopened = pd.read_parquet(interim_dir / "tuning_predictions.parquet")
    if not reopened.equals(merged):
        # Float64 Parquet should be exact; use explicit column checks for clear diagnostics.
        if not np.array_equal(reopened.case_id, merged.case_id) or not np.array_equal(reopened.target, merged.target):
            raise RuntimeError("Merged tuning prediction persistence alignment failed")
        for column in [c for c in merged if c.startswith("prob_")]:
            if not np.array_equal(reopened[column].to_numpy(), merged[column].to_numpy()):
                raise RuntimeError(f"Merged tuning prediction persistence value failed: {column}")
    audits.append({
        "model_id": "ALL_SUCCESSFUL", "check": "merged_tuning_predictions_reopen", "status": "PASS",
        "rows": len(reopened), "positive_count": int(reopened.target.sum()),
        "unique_keys": reopened.case_id.nunique(), "path": str(interim_dir / "tuning_predictions.parquet"),
    })
    return reopened, audits


def make_comparison_tables(
    comparisons: list[dict[str, Any]], predictions: pd.DataFrame, train_prevalence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    by_id = {row["combination"]: row for row in comparisons}
    for combination in MODEL_ORDER:
        if combination not in by_id:
            rows.append({"model_id": combination, "status": "FAILED_OR_NOT_REQUESTED", "evaluation_scope": "validation_tuning"})
            continue
        source = by_id[combination]
        metrics = official_metrics(predictions.target.to_numpy(np.int8), predictions[PROBABILITY_COLUMNS[combination]].to_numpy(np.float64))
        for metric in ("roc_auc", "average_precision", "log_loss", "brier_score"):
            if abs(metrics[metric] - float(source[metric])) > 1e-10:
                raise RuntimeError(f"Saved prediction metric mismatch: {combination} {metric}")
        rows.append({
            "model_id": combination, "status": "COMPLETE", "family": source["family"],
            "feature_set": source["feature_set"], "evaluation_scope": "validation_tuning",
            **metrics, "chosen_candidate": source["candidate_id"],
            "selected_iteration_or_epoch": source.get("iterations", source.get("best_iteration", source.get("best_epoch"))),
            "elapsed_seconds": source.get("elapsed_seconds"), "model_path": source["model_path"],
        })
    ref_p = np.full(len(predictions), train_prevalence, dtype=np.float64)
    rows.append({
        "model_id": "train_prevalence_reference", "status": "REFERENCE_NOT_LEARNED_MODEL",
        "family": "constant_reference", "feature_set": "none", "evaluation_scope": "validation_tuning",
        **official_metrics(predictions.target.to_numpy(np.int8), ref_p),
        "chosen_candidate": "constant_outer_train_positive_rate",
    })
    comparison = pd.DataFrame(rows)
    deltas = []
    for family, t_id, ad_id in (
        ("logit", "logit_T", "logit_T_plus_AD"),
        ("lightgbm", "lightgbm_T", "lightgbm_T_plus_AD"),
        ("mlp", "mlp_T", "mlp_T_plus_AD"),
    ):
        t = comparison.loc[comparison.model_id.eq(t_id)]
        ad = comparison.loc[comparison.model_id.eq(ad_id)]
        if len(t) != 1 or len(ad) != 1 or t.iloc[0].status != "COMPLETE" or ad.iloc[0].status != "COMPLETE":
            deltas.append({"family": family, "status": "PAIR_INCOMPLETE", "evaluation_scope": "validation_tuning"})
            continue
        row = {"family": family, "status": "COMPLETE", "evaluation_scope": "validation_tuning", "delta_definition": "T_plus_AD minus T"}
        for metric in ("roc_auc", "average_precision", "log_loss", "brier_score"):
            row[f"T_{metric}"] = float(t.iloc[0][metric]); row[f"T_plus_AD_{metric}"] = float(ad.iloc[0][metric])
            row[f"delta_{metric}"] = float(ad.iloc[0][metric] - t.iloc[0][metric])
        deltas.append(row)
    return comparison, pd.DataFrame(deltas)


def create_plots(audit_dir: Path, comparison: pd.DataFrame, deltas: pd.DataFrame, history: pd.DataFrame, predictions: pd.DataFrame) -> list[Path]:
    model_rows = comparison[comparison.status.eq("COMPLETE")].copy()
    paths: list[Path] = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric, title in zip(axes.flat, ["roc_auc", "average_precision", "log_loss", "brier_score"], ["ROC AUC", "Average precision", "Log loss", "Brier score"]):
        ax.bar(model_rows.model_id, model_rows[metric], color=["#4C78A8" if x.endswith("_T") else "#F58518" for x in model_rows.model_id])
        ax.set_title(f"Tuning {title}"); ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Validation-tuning metrics (development, not final evaluation)")
    fig.tight_layout(); path = audit_dir / "tuning_metric_comparison.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    selected_hist = history[history.family.isin(["lightgbm", "mlp"])].copy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for combination in [x for x in MODEL_ORDER if x.startswith("lightgbm")]:
        chosen = comparison.loc[comparison.model_id.eq(combination), "chosen_candidate"]
        if len(chosen):
            data = selected_hist[(selected_hist.combination == combination) & (selected_hist.candidate_id == chosen.iloc[0]) & (selected_hist.metric == "average_precision")]
            axes[0].plot(data.step, data.value, label=combination)
            mark = data[data.selected_checkpoint]
            if len(mark): axes[0].scatter(mark.step, mark.value, s=35)
    axes[0].set_title("Selected LightGBM tuning AP by boosting round"); axes[0].legend(); axes[0].grid(alpha=.25)
    for combination in [x for x in MODEL_ORDER if x.startswith("mlp")]:
        data = selected_hist[(selected_hist.combination == combination) & (selected_hist.metric == "average_precision")]
        axes[1].plot(data.step, data.value, marker="o", label=combination)
        mark = data[data.selected_checkpoint]
        if len(mark): axes[1].scatter(mark.step, mark.value, s=45)
    axes[1].set_title("MLP tuning AP by epoch"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); path = audit_dir / "selected_training_histories.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5)); y = predictions.target.to_numpy(np.int8)
    for combination in model_rows.model_id:
        p = predictions[PROBABILITY_COLUMNS[combination]].to_numpy(np.float64)
        fpr, tpr, _ = roc_curve(y, p); precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=combination); axes[1].plot(recall, precision, label=combination)
    axes[0].plot([0,1],[0,1],"--",color="gray"); axes[0].set_title("Tuning ROC curves"); axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[1].axhline(y.mean(), linestyle="--", color="gray"); axes[1].set_title("Tuning precision-recall curves (not trapezoidal AP)"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    for ax in axes: ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); path = audit_dir / "tuning_roc_pr_curves.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    return paths


def write_docs(repo_root: Path, audit_dir: Path, comparison: pd.DataFrame, deltas: pd.DataFrame, run_id: str) -> None:
    complete = comparison[comparison.status.eq("COMPLETE")]
    lines = [
        "# 首批六组模型训练", "",
        "本任务在本机 `home_credit` Conda 环境中执行。Python 进程只使用外层 TRAIN 拟合参数，并只使用 `validation_tuning` 选择候选与早停。校准和最终评估仍保持冻结。", "",
        "## 六组比较的含义", "",
        "每个模型家族分别比较 T 与 T+AD，因此同一家族内的差异反映获批 AD 输入包（数值、来源/内容状态及缺失标志）的整体增量。Logit/MLP 使用保存的线性表示，LightGBM 使用保存的原尺度截尾表示；跨家族比较是完整建模管线比较，不是把算法效应与表示效应完全分离。", "",
        "验证调优集指标是开发结果，不是最终无偏泛化估计。随机划分不能证明未来时间稳定性、重复借款人独立性、因果影响或生产可用性。", "",
        "## 实际调优结果", "",
        "| 模型 | 特征集 | ROC AUC | AP | Log loss | Brier | 候选/检查点 |", "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in complete.iterrows():
        lines.append(f"| {row['family']} | {row['feature_set']} | {row['roc_auc']:.6f} | {row['average_precision']:.6f} | {row['log_loss']:.6f} | {row['brier_score']:.6f} | {row['chosen_candidate']} / {row['selected_iteration_or_epoch']} |")
    lines += ["", "AD 增量统一定义为 T+AD 减 T。AUC/AP 的正增量较好，log loss/Brier 的负增量较好。"]
    for _, row in deltas.iterrows():
        if row.status == "COMPLETE":
            lines.append(f"- {row.family}: ΔAUC={row.delta_roc_auc:+.6f}, ΔAP={row.delta_average_precision:+.6f}, Δlog loss={row.delta_log_loss:+.6f}, ΔBrier={row.delta_brier_score:+.6f}。")
    lines += [
        "", "## 方法与取舍", "",
        "- L2 Logit 提供透明、强正则化的线性基线；三个 C 使用同一小预算，按全局 AP 容差带后最低 log loss 选择。它不提供因果系数解释。",
        "- LightGBM 用两个紧凑叶节点设置和仅监控 AP 的早停。它保留原始 NaN 与观测零，`zero_as_missing=False`。树模型的复杂度控制与输入上截尾是不同机制。",
        "- 两层 MLP 是受控的表格神经基准，不复现论文中的更复杂架构，也不做序列学习。Dropout 抑制共同适配，较小学习率控制更新，梯度裁剪限制训练梯度范数；这些都不保证优于其他家族。",
        "- 所有目标使用自然类别比例，不做重采样或类别加权。ROC AUC 衡量排序，AP 更关注稀少正例，log loss 和 Brier 衡量概率质量。",
        "- 保存的缺失/零/截尾/对数/标志规则保持不变。稀疏字段经中位数填补和标准化后出现较大有限 z 值，并不自动等于预处理错误；本任务记录其 TRAIN/调优分布而不再次裁剪。",
        "", "## 运行与恢复", "",
        "完整顺序运行：", "```bash",
        f"/opt/anaconda3/envs/home_credit/bin/python -u scripts/train_baseline_models.py --data-root /Users/haoguannan/Projects/home_credit/data --run-id {run_id} --models all --threads 4",
        "```", "单组合使用 `--models logit_T` 等名称，仍使用完整 TRAIN/调优成员。中断后加 `--resume`；只验证保存模型和预测则使用 `--verify-only`。终端日志显示阶段、候选和实际迭代/epoch。Activity Monitor 可查看系统级内存，但报告中的 `ru_maxrss` 是本进程累计峰值。", "",
        "模型重载必须同时使用注册表中的有序特征、对应保存表示和原预处理引用。Logit 加载 joblib；LightGBM 加载原生 booster 文本并指定选择轮次；MLP 按 architecture JSON 重建网络、加载 state_dict、调用 `eval()` 并在 `inference_mode` 下预测。", "",
        "后续校准、最终评估、解释/分组、审批与经济模拟均未在本任务执行。",
    ]
    atomic_text(repo_root / "docs/first_model_training.md", "\n".join(lines) + "\n")
    readme_path = repo_root / "README.md"; readme = readme_path.read_text(encoding="utf-8")
    start, end = "<!-- TASK09_RESULTS_START -->", "<!-- TASK09_RESULTS_END -->"
    block = [start, "## First model training", "", f"The six prespecified Logit, LightGBM, and MLP T/T+AD combinations were trained sequentially on the frozen outer TRAIN partition under run `{run_id}`. Reported scores are validation-tuning development metrics, not final evaluation results.", "", "```bash", f"/opt/anaconda3/envs/home_credit/bin/python -u scripts/train_baseline_models.py --data-root /Users/haoguannan/Projects/home_credit/data --run-id {run_id} --models all --threads 4", "```", "", "See `docs/first_model_training.md` for the model contracts, aggregate results, and limitations.", end]
    if start in readme and end in readme:
        before, rest = readme.split(start, 1); _, after = rest.split(end, 1); readme = before + "\n".join(block) + after
    else:
        readme = readme.rstrip() + "\n\n" + "\n".join(block) + "\n"
    readme = readme.replace("- [ ] Train and evaluate the six primary model comparisons.", "- [x] Train and compare the six primary model combinations on validation-tuning.")
    atomic_text(readme_path, readme)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    columns = list(frame.columns)
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.10g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def verify_saved_run(data_root: Path, run_id: str) -> dict[str, Any]:
    audit_dir = data_root / f"audits/task09/{run_id}"
    interim_dir = data_root / f"interim/task09/{run_id}"
    models_dir = data_root / f"models/task09/{run_id}"
    registry_path = audit_dir / "selected_model_registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(interim_dir / "tuning_predictions.parquet")
    results = []
    for item in registry["models"]:
        model_path = Path(item["model_path"])
        checksum_ok = sha256_file(model_path) == item["model_sha256"]
        sample = pd.read_parquet(interim_dir / "reload_samples" / f"{item['model_id']}.parquet")
        features = item["ordered_predictors"]
        values = sample[features].to_numpy(np.float64 if item["family"] == "logit" else np.float32, copy=True)
        if item["family"] == "logit":
            probability = class_one_probability(joblib.load(model_path), values)
        elif item["family"] == "lightgbm":
            probability = lgb.Booster(model_file=str(model_path)).predict(values, num_iteration=int(item["selected_iteration_or_epoch"]))
        else:
            model = TabularMLP(len(features)); model.load_state_dict(torch_load_state(model_path)); model.eval(); probability = predict_mlp(model, values.astype(np.float32))
        diff = float(np.max(np.abs(probability - sample.expected_probability_before_serialization.to_numpy(np.float64))))
        results.append({"model_id": item["model_id"], "checksum_ok": checksum_ok, "reload_max_abs_difference": diff, "status": "PASS" if checksum_ok and diff <= 1e-6 else "FAIL"})
    if any(row["status"] != "PASS" for row in results):
        raise RuntimeError("Saved run verification failed")
    return {"status": "PASS", "models": results, "prediction_rows": len(predictions), "model_directory": str(models_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--test-command", default="")
    parser.add_argument("--test-result", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); selected = parse_models(args.models)
    data_root = args.data_root.resolve(); repo_root = Path(__file__).resolve().parents[1]
    audit_dir = data_root / f"audits/task09/{args.run_id}"
    interim_dir = data_root / f"interim/task09/{args.run_id}"
    models_dir = data_root / f"models/task09/{args.run_id}"
    if args.verify_only:
        print(json.dumps(verify_saved_run(data_root, args.run_id), indent=2)); return 0
    existing = [p for p in (audit_dir, interim_dir, models_dir) if p.exists()]
    if existing and not args.resume:
        raise FileExistsError("Refusing to overwrite existing run: " + ", ".join(map(str, existing)))
    if args.resume and not all(p.exists() for p in (audit_dir, interim_dir, models_dir)):
        raise FileNotFoundError("Resume requires all three existing run directories")
    for path in (audit_dir, interim_dir, models_dir): path.mkdir(parents=True, exist_ok=args.resume)
    log_path = audit_dir / "training.log"
    lock_path = models_dir / ".run.lock"; acquire_lock(lock_path, args.run_id, args.resume)
    started_run = time.perf_counter(); overall_status = "PARTIAL_FAILED"
    try:
        thread_config = configure_threads(args.threads)
        log(f"Task09 start run_id={args.run_id} models={selected} threads={thread_config['effective_threads']}", log_path)
        contract = load_contract(data_root)
        corrections = apply_preparation_report_corrections(data_root)
        atomic_csv(audit_dir / "input_hashes_before_after.csv", corrections)
        constant_rows = constant_input_audit(data_root, contract)
        atomic_csv(audit_dir / "constant_train_input_audit.csv", constant_rows)
        standardized_rows = standardized_value_audit(data_root, contract)
        atomic_csv(audit_dir / "standardized_value_audit.csv", standardized_rows)
        supplement = preprocessing_examples_supplement(data_root, contract)
        atomic_csv(audit_dir / "preprocessing_examples_supplement.csv", supplement)
        if any(row["comparison_status"] != "PASS" for row in supplement):
            raise RuntimeError("Supplemental preprocessing examples failed")
        script_hash = sha256_file(Path(__file__))
        run_config = {
            "version": VERSION, "run_id": args.run_id, "models": selected,
            "threads": thread_config, "runtime_versions": runtime_versions(), "seed": SEED,
            "membership_fingerprint": contract.membership_fingerprint,
            "training_key_fingerprint": contract.training_key_fingerprint,
            "input_hashes": contract.input_hashes, "training_script_sha256": script_hash,
            "selection_rule": "Global successful-candidate max AP; eligible AP >= maxAP-0.0001; lowest log loss; lower complexity; candidate ID.",
            "model_budgets": {
                "logit": {"C": [0.01, 0.1, 1.0], "solver": "lbfgs", "max_iter": 1000, "retry_max_iter": 2000},
                "lightgbm": [{"id": "compact", "num_leaves": 15, "min_child_samples": 500}, {"id": "moderate", "num_leaves": 31, "min_child_samples": 200}],
                "mlp": {"hidden": [64, 32], "dropout": .1, "epochs": 30, "patience": 5, "batch_size": 4096},
            },
            "selection_manifest_scope": "Persisted before any full candidate metrics are observed.",
        }
        config_hash = hashlib.sha256(canonical_json(run_config).encode()).hexdigest()
        config_path = audit_dir / "run_config.json"
        if config_path.exists():
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
            existing_hash = existing_config.pop("config_hash")
            if existing_hash != config_hash or hashlib.sha256(canonical_json(existing_config).encode()).hexdigest() != config_hash:
                raise RuntimeError("Resume configuration/input/code hash mismatch; old run retained")
        else:
            atomic_json(config_path, {**run_config, "config_hash": config_hash})
        train_meta = contract.manifest.iloc[contract.train_positions][["case_id", "base_order"]].reset_index(drop=True)
        tune_meta = contract.manifest.iloc[contract.tuning_positions][["case_id", "base_order"]].reset_index(drop=True)
        all_candidates: list[dict[str, Any]] = []; all_history: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []; registry_models: list[dict[str, Any]] = []
        resource_rows: list[dict[str, Any]] = []
        for combination in selected:
            family, representation, features = feature_contract_for(combination, contract.feature_sets)
            status_path = models_dir / combination / "combination_status.json"
            if args.resume and combination_is_complete(status_path, config_hash):
                log(f"RESUME verified skip {combination}", log_path)
                comparison, candidates, history, registry = load_completed(status_path)
            else:
                matrix_path = data_root / f"interim/task08_followup/{'gbdt_inputs.parquet' if representation == 'gbdt' else 'linear_nn_inputs.parquet'}"
                dtype = np.float64 if family == "logit" else np.float32
                load_start = time.perf_counter(); rss_start = ru_maxrss_bytes()
                x_train, train_info = load_matrix_rows(matrix_path, features, contract.train_positions, contract.manifest.case_id.to_numpy(), dtype)
                x_tune, tune_info = load_matrix_rows(matrix_path, features, contract.tuning_positions, contract.manifest.case_id.to_numpy(), dtype)
                if train_info["contains_infinity"] or tune_info["contains_infinity"]:
                    raise RuntimeError(f"Infinity found after conversion for {combination}")
                if representation == "linear_nn" and (train_info["contains_nan"] or tune_info["contains_nan"]):
                    raise RuntimeError(f"NaN found in linear/NN input for {combination}")
                resource_rows.append({
                    "scope": f"matrix_load::{combination}", "elapsed_seconds": time.perf_counter()-load_start,
                    "expected_array_bytes": int(x_train.nbytes+x_tune.nbytes), "train_shape": str(x_train.shape),
                    "tuning_shape": str(x_tune.shape), "dtype": str(x_train.dtype),
                    "ru_maxrss_bytes_start": rss_start, "ru_maxrss_bytes_end": ru_maxrss_bytes(),
                    "ru_maxrss_scope": "cumulative process peak; not per-model incremental RAM or total system memory",
                    "current_rss": None, "system_memory": None,
                })
                model_dir = models_dir / combination; model_dir.mkdir(parents=True, exist_ok=True)
                if family == "logit":
                    comparison, candidates, history, registry = run_logit_combination(
                        combination, features, x_train, x_tune, contract.train_labels, contract.tuning_labels,
                        train_meta, tune_meta, model_dir, interim_dir, log_path, thread_config["effective_threads"],
                    )
                elif family == "lightgbm":
                    comparison, candidates, history, registry = run_lightgbm_combination(
                        combination, features, x_train, x_tune, contract.train_labels, contract.tuning_labels,
                        train_meta, tune_meta, model_dir, interim_dir, log_path, thread_config["effective_threads"],
                    )
                else:
                    comparison, candidates, history, registry = run_mlp_combination(
                        combination, features, x_train, x_tune, contract.train_labels, contract.tuning_labels,
                        train_meta, tune_meta, model_dir, interim_dir, log_path, thread_config["effective_threads"],
                    )
                registry.update({
                    "representation": representation,
                    "preprocessor_reference": str(data_root / "interim/task08_followup/preprocessing/preprocessor.json"),
                    "preprocessor_sha256": contract.input_hashes[str(data_root / "interim/task08_followup/preprocessing/preprocessor.json")],
                    "fit_membership_fingerprint": contract.training_key_fingerprint,
                    "tuning_membership_fingerprint": sha256_text(tune_meta.case_id),
                    "calibration_status": "NOT_FITTED", "decision_threshold_status": "NOT_SELECTED",
                })
                persist_combination_status(
                    status_path, config_hash, comparison, candidates, history, registry,
                    interim_dir / "prediction_parts" / f"{combination}.parquet",
                )
                del x_train, x_tune; gc.collect()
            comparisons.append(comparison); all_candidates.extend(candidates); all_history.extend(history); registry_models.append(registry)
            atomic_csv(audit_dir / "model_candidates.csv", all_candidates)
            atomic_csv(audit_dir / "training_history.csv", all_history)
            atomic_json(audit_dir / "selected_model_registry.json", {"run_id": args.run_id, "models": registry_models})
            log(f"DURABLE COMPLETE {combination}; moving to next combination", log_path)
        predictions, prediction_audit = rebuild_predictions(interim_dir, selected, tune_meta, contract.tuning_labels)
        comparison_df, delta_df = make_comparison_tables(comparisons, predictions, float(contract.train_labels.mean()))
        atomic_csv(audit_dir / "model_comparison.csv", comparison_df)
        atomic_csv(audit_dir / "ad_increment_comparison.csv", delta_df)
        atomic_csv(audit_dir / "prediction_audit.csv", prediction_audit)
        atomic_csv(audit_dir / "resource_usage.csv", resource_rows + [
            {"scope": "whole_run", "elapsed_seconds": time.perf_counter()-started_run, "ru_maxrss_bytes_end": ru_maxrss_bytes(),
             "ru_maxrss_scope": "cumulative process peak; not per-model incremental RAM or total system memory", "current_rss": None, "system_memory": None}
        ])
        history_df = pd.DataFrame(all_history)
        plot_paths = create_plots(audit_dir, comparison_df, delta_df, history_df, predictions)
        write_docs(repo_root, audit_dir, comparison_df, delta_df, args.run_id)
        registry_payload = {
            "run_id": args.run_id, "status": "COMPLETE" if selected == MODEL_ORDER else "COMPLETE_REQUESTED_SUBSET",
            "membership_fingerprint": contract.membership_fingerprint,
            "training_key_fingerprint": contract.training_key_fingerprint,
            "models": registry_models,
        }
        atomic_json(audit_dir / "selected_model_registry.json", registry_payload)
        validation_rows = [
            {"check": "membership_fingerprint", "status": "PASS", "actual": contract.membership_fingerprint, "expected": EXPECTED_MEMBERSHIP_FP, "reason": "Canonical case_id/outer_split/validation_role recipe."},
            {"check": "training_key_fingerprint", "status": "PASS", "actual": contract.training_key_fingerprint, "expected": EXPECTED_TRAINING_KEY_FP, "reason": "Canonical original-order TRAIN case_id recipe."},
            {"check": "train_tuning_counts_labels", "status": "PASS", "actual": f"{len(contract.train_labels)}/{contract.train_labels.sum()}; {len(contract.tuning_labels)}/{contract.tuning_labels.sum()}", "expected": f"{EXPECTED_TRAIN}/{EXPECTED_TRAIN_POS}; {EXPECTED_TUNING}/{EXPECTED_TUNING_POS}", "reason": "Only authorized development labels loaded."},
            {"check": "prepared_bytes_preserved", "status": "PASS", "actual": all(row["bytes_unchanged"] for row in corrections if not row["expected_change"]), "expected": True, "reason": "Matrix and learned preprocessor SHA-256 unchanged across report corrections."},
            {"check": "supplement_examples", "status": "PASS", "actual": len(supplement), "expected": "all comparisons PASS", "reason": "Real missing/zero/known/unseen and disclosed synthetic category paths."},
            {"check": "all_requested_combinations", "status": "PASS", "actual": len(comparisons), "expected": len(selected), "reason": "Each requested combination durably selected, serialized, reloaded and predicted."},
            {"check": "saved_prediction_recompute", "status": "PASS", "actual": len(predictions), "expected": EXPECTED_TUNING, "reason": "Metrics independently recomputed from reopened merged tuning predictions."},
            {"check": "target_horizon", "status": "NOT_CHECKED", "actual": "competition binary label", "expected": "formal real-world horizon", "reason": "Not established by authorized evidence."},
            {"check": "prospective_time_stability", "status": "NOT_CHECKED", "actual": "random development split", "expected": "future-period evidence", "reason": "Chronological evaluation deferred."},
            {"check": "tests", "status": "PASS" if args.test_result else "NOT_CHECKED", "actual": args.test_result or None, "expected": "focused and repository suite passed", "reason": args.test_command or "No command supplied"},
        ]
        atomic_csv(audit_dir / "validation_results.csv", validation_rows)
        summary = {
            "task": "TASK_09_FIRST_MODEL_TRAINING", "status": "COMPLETE" if selected == MODEL_ORDER else "COMPLETE_REQUESTED_SUBSET",
            "run_id": args.run_id, "started_command": " ".join(sys.argv), "completed_at": utc_now(),
            "runtime_versions": runtime_versions(), "thread_configuration": thread_config,
            "tests": {"command": args.test_command or None, "result": args.test_result or None},
            "split": {"train_rows": EXPECTED_TRAIN, "train_positive": EXPECTED_TRAIN_POS, "tuning_rows": EXPECTED_TUNING, "tuning_positive": EXPECTED_TUNING_POS},
            "fingerprints": {"membership": contract.membership_fingerprint, "training_keys": contract.training_key_fingerprint},
            "input_hashes": contract.input_hashes, "config_hash": config_hash,
            "combination_statuses": {row["combination"]: "COMPLETE" for row in comparisons},
            "model_comparison": comparison_df.to_dict("records"), "ad_increment_comparison": delta_df.to_dict("records"),
            "resource_scope": "ru_maxrss is cumulative process peak; current RSS and total system memory unavailable in the standard-library telemetry used.",
            "plots": [str(p) for p in plot_paths],
            "deferred": ["calibration", "final evaluation", "subgroup/interpretation", "decision thresholds", "approval/economic simulations", "chronological supplement", "Streamlit/publication"],
            "limitations": ["Tuning metrics are development results.", "Random split does not establish time stability or repeat-borrower independence.", "Target horizon remains unverified.", "T+AD effects belong to the combined approved AD package, not bank flows alone."],
        }
        atomic_json(audit_dir / "training_summary.json", summary)
        report_lines = [
            "# Task 09 首批模型训练报告", "", f"状态：**{summary['status']}**。六组模型使用完整 outer TRAIN 拟合，并仅用 validation_tuning 选择；未读取校准/最终评估标签或生成其预测。", "",
            "## 调优集比较", "", markdown_table(comparison_df), "", "## AD 增量（T+AD 减 T）", "", markdown_table(delta_df), "",
            "这些是模型开发指标，不是最终独立评估。正的 AUC/AP 增量较好，负的 log loss/Brier 增量较好；不强制做有利解释。", "",
            "所有保存模型均完成重载样本与完整调优预测验证。没有拟合校准器、选择阈值或执行审批/经济模拟。随机划分不能证明未来时间稳定性、借款人独立性、因果效果或生产可用性。",
        ]
        atomic_text(audit_dir / "training_report.md", "\n".join(report_lines) + "\n")
        verification = verify_saved_run(data_root, args.run_id)
        overall_status = summary["status"]
        log(f"Task09 COMPLETE run_id={args.run_id} verification={verification['status']}", log_path)
        print(json.dumps({"status": overall_status, "run_id": args.run_id, "models": selected, "audit_dir": str(audit_dir), "interim_dir": str(interim_dir), "models_dir": str(models_dir)}, indent=2))
        return 0
    except KeyboardInterrupt:
        atomic_json(audit_dir / "interrupted.json", {"status": "INTERRUPTED", "time": utc_now(), "resume_command": f"{sys.executable} -u {__file__} --data-root {data_root} --run-id {args.run_id} --models {args.models} --threads {args.threads} --resume"})
        log("Task09 INTERRUPTED; completed artifacts retained", log_path); raise
    except Exception as exc:
        atomic_json(audit_dir / "failure.json", {"status": overall_status, "time": utc_now(), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=50), "resume_command": f"{sys.executable} -u {__file__} --data-root {data_root} --run-id {args.run_id} --models {args.models} --threads {args.threads} --resume"})
        log(f"Task09 FAILED/PARTIAL: {type(exc).__name__}: {exc}", log_path); raise
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
