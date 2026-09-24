#!/usr/bin/env python3
"""Calibrate the six frozen Task 09 baseline models on validation_calibration."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _thread_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_name, "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/home_credit_mpl_task10")

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from task10_calibration import (
    COMPLEXITY,
    METHODS,
    CalibrationContractError,
    RawIdentityPredictionError,
    annotate_bootstrap_references,
    apply_candidate,
    atomic_csv,
    atomic_joblib,
    atomic_json,
    atomic_parquet,
    atomic_text,
    brier_guardrail_pass,
    canonical_json,
    choose_numeric_best,
    completion_check,
    fit_isotonic_calibrator,
    fit_logistic_calibrator,
    fold_metric_rows,
    fold_summary,
    isotonic_ranking_guardrail,
    json_safe,
    log_loss_one_se_pass,
    logistic_ranking_guardrail,
    make_common_folds,
    mechanical_selection,
    metric_with_calibration_diagnostics,
    official_metrics,
    paired_standard_error,
    paired_stratified_bootstrap,
    sha256_file,
    sha256_lines,
    stability_warning,
    top_risk_summary,
    validate_oof,
    validate_probabilities,
    validate_raw_identity_predictions,
    verify_v2_selection_table,
)
from train_baseline_models import (
    TabularMLP,
    class_one_probability,
    feature_contract_for,
    load_matrix_rows,
    predict_mlp,
    torch_load_state,
)


VERSION = "task10_probability_calibration_v2"
SELECTION_CONTRACT = "hard_guardrails_then_one_se_v2"
MODEL_ORDER = ["logit_T", "logit_T_plus_AD", "lightgbm_T", "lightgbm_T_plus_AD", "mlp_T", "mlp_T_plus_AD"]
EXPECTED_TOTAL = 1_526_659
EXPECTED_CALIBRATION = 114_500
EXPECTED_POSITIVES = 3_600
EXPECTED_NEGATIVES = 110_900
EXPECTED_MEMBERSHIP_FP = "8bfb238774655f46a4f21c2668b2815f321c9298cb6d38a8da33f35a1ab91714"
EXPECTED_TRAIN_FP = "bed8c0241e093a8bacedbfb0406960f2b5a7873db277ba6cf1ab7ce09b90736e"
CROSSFIT_FOLDS = 5
CROSSFIT_SEED = 20260921
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260922
ISOTONIC_MAX_DROP = 0.0005
LOGISTIC_RANK_TOLERANCE = 1e-10
NUMERIC_TIE_TOLERANCE = 1e-12
BOOTSTRAP_DDOF = 1
FULL_FIT_ROLE = "IN_SAMPLE_ARTIFACT_DIAGNOSTIC_NOT_PERFORMANCE_ESTIMATE"
OOF_ROLE = "OOF_CALIBRATION_DEVELOPMENT_ESTIMATE"
RAW_ROLE = "RAW_BASE_MODEL_DEVELOPMENT_DIAGNOSTIC"
TASK08_HASHES = {
    "application_manifest.parquet": "6c93a384a3714bf4288e7964ba9959f8086127fdb3f38e62552803c163443e23",
    "linear_nn_inputs.parquet": "0bcd60da9864f12e7d822fcc8ab0601b2ee8559f12c1440196378a8e097bc51d",
    "gbdt_inputs.parquet": "306f2734133ef819b179170a72cd379d7856fc754f90234da4d968cb4ea32fd6",
    "preprocessing/preprocessor.json": "be2039e6a04dd48e9f0725b59f6652f494fe0ad2807d8b9a74085bd718b3efd9",
    "preprocessing/feature_sets.json": "bc4b10ed53614243fab6a3161e1cebf422d2a9e7d9af85c65e2a69db32270e39",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ru_maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def log(message: str, path: Path | None = None) -> None:
    line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    if path is not None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n"); stream.flush()


def runtime_versions() -> dict[str, Any]:
    return {"python": platform.python_version(), "python_executable": sys.executable, "numpy": np.__version__,
            "pandas": pd.__version__, "pyarrow": pa.__version__, "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__, "torch": torch.__version__, "joblib": joblib.__version__,
            "matplotlib": plt.matplotlib.__version__}


def configure_threads(threads: int) -> dict[str, Any]:
    effective = max(1, min(int(threads), 4, os.cpu_count() or 1))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(effective)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(effective)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return {"requested_threads": threads, "effective_threads": effective, "torch_intraop": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(), "environment": {name: os.environ[name] for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "threadpools": threadpool_info()}


def parse_models(value: str) -> list[str]:
    if value == "all":
        return list(MODEL_ORDER)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested or any(item not in MODEL_ORDER for item in requested):
        raise ValueError(f"Unknown model selection: {requested}")
    return [item for item in MODEL_ORDER if item in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--base-run-id", default="first_full")
    parser.add_argument("--run-id", default="first_full")
    parser.add_argument("--models", default="all")
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def ensure_external(path: Path, repo_root: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        return
    raise CalibrationContractError(f"Private application-level output cannot target repository: {resolved}")


def protected_paths(data_root: Path, base_run_id: str) -> list[tuple[str, Path]]:
    prep = data_root / "interim/task08_followup"
    task09_audit = data_root / "audits/task09" / base_run_id
    v4 = data_root / "audits/task09/first_full_post_review_v4"
    registry = json.loads((task09_audit / "selected_model_registry.json").read_text(encoding="utf-8"))
    paths: list[tuple[str, Path]] = [
        ("task08_manifest", prep / "application_manifest.parquet"),
        ("task08_linear", prep / "linear_nn_inputs.parquet"),
        ("task08_gbdt", prep / "gbdt_inputs.parquet"),
        ("task08_preprocessor", prep / "preprocessing/preprocessor.json"),
        ("task08_feature_sets", prep / "preprocessing/feature_sets.json"),
        ("task09_run_config", task09_audit / "run_config.json"),
        ("task09_registry", task09_audit / "selected_model_registry.json"),
    ]
    for item in registry["models"]:
        paths.append((f"task09_model::{item['model_id']}", Path(item["model_path"])))
        if item.get("architecture_path"):
            paths.append((f"task09_architecture::{item['model_id']}", Path(item["architecture_path"])))
    v4_provenance = pd.read_csv(v4 / "provenance_hashes.csv")
    for row in v4_provenance.itertuples(index=False):
        paths.append((f"v4_protected::{row.artifact}", Path(row.path)))
    unique: dict[str, tuple[str, Path]] = {}
    for name, path in paths:
        unique[str(path)] = (name, path)
    return list(unique.values())


def hash_rows(paths: list[tuple[str, Path]], stage: str) -> list[dict[str, Any]]:
    rows = []
    for name, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"artifact": name, "path": str(path), "stage": stage, "size_bytes": path.stat().st_size,
                     "sha256": sha256_file(path), "status": "MEASURED"})
    return rows


def verify_protected_hash_tables(before: pd.DataFrame, after: pd.DataFrame, check_current_files: bool = True) -> None:
    merged = before[["path", "sha256"]].merge(after[["path", "sha256"]], on="path", suffixes=("_before", "_after"), validate="one_to_one")
    if len(merged) != len(before) or len(merged) != len(after) or not (merged.sha256_before == merged.sha256_after).all():
        raise CalibrationContractError("Protected input before/after hash mismatch")
    if check_current_files:
        for row in after.itertuples(index=False):
            if sha256_file(Path(row.path)) != row.sha256:
                raise CalibrationContractError(f"Protected input changed after Task 10: {row.path}")


def verify_selected_calibrator_copy(registry_item: dict[str, Any]) -> None:
    selected = Path(registry_item["selected_calibrator_path"])
    source = Path(registry_item["selected_calibrator_source_path"])
    selected_hash = sha256_file(selected)
    source_hash = sha256_file(source)
    if selected_hash != registry_item["selected_calibrator_sha256"] or source_hash != registry_item["selected_calibrator_source_sha256"]:
        raise CalibrationContractError(f"Selected calibrator registered hash mismatch: {registry_item['model_id']}")
    if selected_hash != source_hash:
        raise CalibrationContractError(f"Selected calibrator is not byte-identical to source: {registry_item['model_id']}")


def load_membership(data_root: Path) -> tuple[pd.DataFrame, np.ndarray, str, str]:
    path = data_root / "interim/task08_followup/application_manifest.parquet"
    metadata = pd.read_parquet(path, columns=["case_id", "base_order", "outer_split", "validation_role"])
    if len(metadata) != EXPECTED_TOTAL or metadata.case_id.isna().any() or not metadata.case_id.is_unique:
        raise CalibrationContractError("Complete manifest key/count contract failed")
    if not np.array_equal(metadata.base_order.to_numpy(np.int64), np.arange(EXPECTED_TOTAL, dtype=np.int64)):
        raise CalibrationContractError("Manifest base_order is not canonical")
    membership_fp = sha256_lines(f"{case}\t{outer}\t{role}" for case, outer, role in zip(metadata.case_id, metadata.outer_split, metadata.validation_role))
    train_fp = sha256_lines(metadata.loc[metadata.outer_split.eq("train"), "case_id"])
    if membership_fp != EXPECTED_MEMBERSHIP_FP or train_fp != EXPECTED_TRAIN_FP:
        raise CalibrationContractError("Frozen membership fingerprint mismatch")
    calibration_positions = np.flatnonzero(metadata.validation_role.eq("validation_calibration").to_numpy())
    labels = pd.read_parquet(path, columns=["case_id", "base_order", "target", "validation_role"],
                             filters=[("validation_role", "==", "validation_calibration")]).sort_values("base_order", kind="stable")
    if len(labels) != EXPECTED_CALIBRATION or int(labels.target.sum()) != EXPECTED_POSITIVES:
        raise CalibrationContractError("Calibration role count/positive contract failed")
    if not labels.case_id.notna().all() or not labels.case_id.is_unique or not labels.base_order.notna().all() or not labels.base_order.is_unique:
        raise CalibrationContractError("Calibration keys/orders must be complete and unique")
    if not np.array_equal(labels.case_id.to_numpy(), metadata.iloc[calibration_positions].case_id.to_numpy()) or not np.array_equal(labels.base_order.to_numpy(), calibration_positions):
        raise CalibrationContractError("Calibration target rows do not align with metadata-only membership")
    return labels[["case_id", "base_order", "target"]].reset_index(drop=True), calibration_positions, membership_fp, train_fp


def preflight(data_root: Path, base_run_id: str, selected_models: list[str]) -> dict[str, Any]:
    prep = data_root / "interim/task08_followup"
    for relative, expected in TASK08_HASHES.items():
        path = prep / relative
        if sha256_file(path) != expected:
            raise CalibrationContractError(f"Frozen Task 08 hash mismatch: {path}")
    v4 = data_root / "audits/task09/first_full_post_review_v4"
    v4_summary = json.loads((v4 / "post_review_summary.json").read_text(encoding="utf-8"))
    if v4_summary.get("status") != "PASS" or not v4_summary.get("original_artifacts_unchanged"):
        raise CalibrationContractError("Task 09 v4 review is not authoritative PASS")
    verification_models = v4_summary.get("verification", {}).get("models", [])
    if [row.get("model_id") for row in verification_models] != MODEL_ORDER or any(row.get("status") != "PASS" for row in verification_models):
        raise CalibrationContractError("Task 09 v4 six-model full-tuning verification failed")
    audit = data_root / "audits/task09" / base_run_id
    registry_path = audit / "selected_model_registry.json"; config_path = audit / "run_config.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("status") != "COMPLETE" or [item.get("model_id") for item in registry.get("models", [])] != MODEL_ORDER:
        raise CalibrationContractError("Task 09 registry model order/status mismatch")
    by_id = {item["model_id"]: item for item in registry["models"]}
    for model_id in selected_models:
        item = by_id[model_id]
        if item.get("calibration_status") != "NOT_FITTED" or item.get("decision_threshold_status") != "NOT_SELECTED":
            raise CalibrationContractError(f"Task 09 downstream state changed: {model_id}")
        if item.get("class_order") != [0, 1] or sha256_file(Path(item["model_path"])) != item["model_sha256"]:
            raise CalibrationContractError(f"Task 09 selected model contract failed: {model_id}")
        if item["family"] == "mlp" and not Path(item.get("architecture_path", "")).is_file():
            raise CalibrationContractError(f"Task 09 MLP architecture missing: {model_id}")
    feature_sets = json.loads((prep / "preprocessing/feature_sets.json").read_text(encoding="utf-8"))
    return {"registry": registry, "registry_path": registry_path, "run_config_path": config_path,
            "models": by_id, "feature_sets": feature_sets, "v4_summary_path": v4 / "post_review_summary.json"}


def infer_raw_predictions(
    data_root: Path, membership: pd.DataFrame, positions: np.ndarray, model_registry: dict[str, Any],
    feature_sets: dict[str, Any], selected_models: list[str], log_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    result = membership.copy(); probability_audit = []; resource_rows = []
    manifest_keys = pd.read_parquet(data_root / "interim/task08_followup/application_manifest.parquet", columns=["case_id"])["case_id"].to_numpy()
    for model_id in selected_models:
        item = model_registry[model_id]
        family, representation, expected_features = feature_contract_for(model_id, feature_sets)
        if item["family"] != family or item["representation"] != representation or item["ordered_predictors"] != expected_features:
            raise CalibrationContractError(f"Registered feature contract mismatch: {model_id}")
        matrix_path = data_root / "interim/task08_followup" / ("gbdt_inputs.parquet" if representation == "gbdt" else "linear_nn_inputs.parquet")
        dtype = np.float64 if family == "logit" else np.float32
        started = time.perf_counter(); before_rss = ru_maxrss_bytes()
        values, info = load_matrix_rows(matrix_path, expected_features, positions, manifest_keys, dtype)
        if info["contains_infinity"] or (representation == "linear_nn" and info["contains_nan"]):
            raise CalibrationContractError(f"Prepared matrix numeric contract failed: {model_id}")
        model_path = Path(item["model_path"])
        if family == "logit":
            model = joblib.load(model_path)
            if list(model.classes_) != [0, 1] or model.n_features_in_ != len(expected_features):
                raise CalibrationContractError(f"Logit reload contract failed: {model_id}")
            probability = class_one_probability(model, values)
        elif family == "lightgbm":
            model = lgb.Booster(model_file=str(model_path)); iteration = int(item["selected_iteration_or_epoch"])
            if model.num_feature() != len(expected_features) or model.current_iteration() != iteration:
                raise CalibrationContractError(f"LightGBM reload contract failed: {model_id}")
            probability = np.asarray(model.predict(values, num_iteration=iteration), dtype=np.float64)
        else:
            architecture = json.loads(Path(item["architecture_path"]).read_text(encoding="utf-8"))
            if architecture["input_dim"] != len(expected_features):
                raise CalibrationContractError(f"MLP architecture input mismatch: {model_id}")
            model = TabularMLP(len(expected_features)); model.load_state_dict(torch_load_state(model_path)); model.eval()
            probability = predict_mlp(model, values.astype(np.float32, copy=False))
        audit = validate_raw_identity_predictions(np.asarray(probability, dtype=np.float64), EXPECTED_CALIBRATION)
        result[f"raw__{model_id}"] = np.asarray(probability, dtype=np.float64)
        probability_audit.append({"model_id": model_id, "method": "identity", "probability_role": "raw", "estimate_role": RAW_ROLE, **audit, "status": "PASS"})
        resource_rows.append({"stage": f"base_inference::{model_id}", "elapsed_seconds": time.perf_counter() - started,
                              "array_bytes": int(values.nbytes), "ru_maxrss_bytes_start": before_rss, "ru_maxrss_bytes_end": ru_maxrss_bytes()})
        log(f"Raw inference complete model={model_id} rows={len(probability)}", log_path)
        del values, model, probability; gc.collect()
    expected = ["case_id", "base_order", "target", *[f"raw__{model_id}" for model_id in selected_models]]
    if list(result.columns) != expected:
        raise CalibrationContractError("Raw prediction schema/order mismatch")
    return result, probability_audit, resource_rows


def save_calibrator(
    method: str, calibrator: Any, path: Path, metadata_path: Path, metadata: dict[str, Any],
) -> tuple[str, str]:
    if method == "identity":
        atomic_json(path, {"method": "identity", "mapping": "p_calibrated = p_raw"})
    else:
        atomic_joblib(path, calibrator)
    calibrator_hash = sha256_file(path)
    atomic_json(metadata_path, {**metadata, "calibrator_path": str(path), "calibrator_sha256": calibrator_hash})
    return calibrator_hash, sha256_file(metadata_path)


def load_calibrator(method: str, path: Path) -> Any:
    if method == "identity":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload != {"method": "identity", "mapping": "p_calibrated = p_raw"}:
            raise CalibrationContractError("Identity calibrator JSON mismatch")
        return None
    return joblib.load(path)


def run_test_command(command: list[str], cwd: Path, audit_dir: Path, scope: str) -> dict[str, Any]:
    stdout_path = audit_dir / f"{scope}.stdout.txt"; stderr_path = audit_dir / f"{scope}.stderr.txt"
    junit_path = audit_dir / f"{scope}.junit.xml" if "pytest" in command else None
    if junit_path is not None:
        command = [*command, f"--junitxml={junit_path}"]
    started_at = utc_now(); started = time.perf_counter()
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, check=False)
    atomic_text(stdout_path, result.stdout); atomic_text(stderr_path, result.stderr)
    collected = 1
    if junit_path is not None and junit_path.is_file():
        root = ET.parse(junit_path).getroot()
        collected = int(root.attrib.get("tests", 0)) if root.tag == "testsuite" else sum(int(node.attrib.get("tests", 0)) for node in root.findall("testsuite"))
    status = "PASS" if result.returncode == 0 and collected > 0 else "FAIL"
    return {"scope": scope, "command": " ".join(command), "cwd": str(cwd), "started_at": started_at, "ended_at": utc_now(),
            "elapsed_seconds": time.perf_counter() - started, "exit_code": result.returncode, "collected_tests": collected,
            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path), "status": status}


def build_documentation(repo_root: Path, selections: pd.DataFrame, metrics: pd.DataFrame, run_id: str) -> None:
    selected_metrics = metrics[metrics.probability_version.eq("oof_selected")].set_index("model_id")
    raw_metrics = metrics[metrics.probability_version.eq("raw_identity")].set_index("model_id")
    lines = [
        "# Probability calibration", "",
        "Task 10 calibrated six frozen Task 09 models without retraining them. Calibration used only the `validation_calibration` partition. Each model compared identity, logistic, and isotonic mappings using common five-fold cross-fitting and 2,000 paired target-stratified bootstrap replicates.", "",
        "Hard probability-validity, Brier, and ranking constraints were applied before identifying the admissible numerical Log-loss best. The paired one-standard-error rule then selected the simplest statistically competitive admissible method.", "",
        "## Verified calibration-development results", "", "| Model | Selected method | Raw Log loss | OOF selected Log loss | Raw Brier | OOF selected Brier |", "|---|---|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_ORDER:
        method = selections.loc[selections.model_id.eq(model_id) & selections.selected, "method"].iloc[0]
        lines.append(f"| {model_id} | {method} | {raw_metrics.loc[model_id,'log_loss']:.8f} | {selected_metrics.loc[model_id,'log_loss']:.8f} | {raw_metrics.loc[model_id,'brier_score']:.8f} | {selected_metrics.loc[model_id,'brier_score']:.8f} |")
    lines += [
        "", "## Interpretation limits", "",
        "The OOF metrics are calibration-development estimates, not final unbiased performance estimates. Full-fit calibration probabilities are artifact and structural diagnostics, not evidence of out-of-sample improvement. Identity means the raw base probability was retained because no admissible more complex method demonstrated sufficient stable benefit under the fixed rule; it is not an improvement claim.", "",
        "Final evaluation remained untouched and no lending threshold was selected. The unit is an application because no reliable borrower identifier is available. This random-design analysis does not establish future-time stability, causality, production readiness, borrower-level independence, or counterfactual outcomes for historically rejected applicants.", "",
        "## Reproduction", "", "```bash", f"/opt/anaconda3/envs/home_credit/bin/python -u scripts/calibrate_baseline_models.py --data-root /Users/haoguannan/Projects/home_credit/data --base-run-id first_full --run-id {run_id} --models all --threads 4", "```", "",
        "Add `--verify-only` to run the separate saved-artifact verification. Application-level predictions, labels, calibrators, and private audit evidence remain under the external data root.",
    ]
    atomic_text(repo_root / "docs/probability_calibration.md", "\n".join(lines) + "\n")
    readme_path = repo_root / "README.md"; readme = readme_path.read_text(encoding="utf-8")
    start, end = "<!-- TASK10_RESULTS_START -->", "<!-- TASK10_RESULTS_END -->"
    block = [start, "## Probability calibration", "", f"Task 10 run `{run_id}` calibrated the six frozen Task 09 models without retraining them. It used only `validation_calibration`, with identity, logistic, and isotonic candidates, common five-fold cross-fitting, and 2,000 paired target-stratified bootstrap replicates.", "", "Hard Brier and ranking constraints were applied before choosing the admissible Log-loss reference. The paired one-standard-error rule then selected the simplest statistically competitive admissible method. These are calibration-development results; final evaluation remained untouched and no lending threshold was selected.", "", "See `docs/probability_calibration.md` for verified aggregate results and interpretation limits.", end]
    if start in readme and end in readme:
        before, rest = readme.split(start, 1); _, after = rest.split(end, 1); readme = before + "\n".join(block) + after
    else:
        readme = readme.rstrip() + "\n\n" + "\n".join(block) + "\n"
    atomic_text(readme_path, readme)


def verify_saved_run(data_root: Path, base_run_id: str, run_id: str, selected_models: list[str]) -> dict[str, Any]:
    interim = data_root / "interim/task10" / run_id; models_root = data_root / "models/task10" / run_id; audit = data_root / "audits/task10" / run_id
    required = [audit / name for name in ("run_config.json", "calibration_registry.json", "calibration_summary.json", "calibration_method_selection.csv", "calibration_metrics.csv", "bootstrap_selection_results.csv", "input_artifact_hashes_before.csv", "input_artifact_hashes_after.csv", "task_status.json")]
    required += [interim / name for name in ("calibration_raw_predictions.parquet", "calibration_fold_assignments.parquet", "calibration_oof_predictions.parquet", "calibration_full_fit_selected_predictions.parquet")]
    for path in required:
        if not path.is_file(): raise FileNotFoundError(path)
    config = json.loads((audit / "run_config.json").read_text(encoding="utf-8")); stored_hash = config["config_hash"]
    payload = dict(config); payload.pop("config_hash")
    if hashlib.sha256(canonical_json(payload).encode()).hexdigest() != stored_hash:
        raise CalibrationContractError("Saved Task 10 config hash mismatch")
    if config.get("version") != VERSION or config.get("selection_contract") != SELECTION_CONTRACT:
        raise CalibrationContractError("Saved Task 10 v2 selection contract mismatch")
    membership, positions, membership_fp, _ = load_membership(data_root)
    if membership_fp != config["membership_fingerprint"]:
        raise CalibrationContractError("Saved Task 10 membership fingerprint mismatch")
    raw = pd.read_parquet(interim / "calibration_raw_predictions.parquet")
    fold_frame = pd.read_parquet(interim / "calibration_fold_assignments.parquet")
    oof = pd.read_parquet(interim / "calibration_oof_predictions.parquet")
    expected_raw = ["case_id", "base_order", "target", *[f"raw__{model_id}" for model_id in selected_models]]
    expected_oof = ["case_id", "base_order", "target", "fold_id", *[f"oof__{model_id}__{method}" for model_id in selected_models for method in METHODS]]
    if list(raw.columns) != expected_raw or list(fold_frame.columns) != ["case_id", "base_order", "target", "fold_id"] or list(oof.columns) != expected_oof:
        raise CalibrationContractError("Saved application-level schema mismatch")
    for frame in (raw, fold_frame, oof):
        if not np.array_equal(frame.case_id.to_numpy(), membership.case_id.to_numpy()) or not np.array_equal(frame.target.to_numpy(np.int8), membership.target.to_numpy(np.int8)):
            raise CalibrationContractError("Saved application-level membership/target alignment mismatch")
    folds_expected = make_common_folds(membership.target.to_numpy(np.int8), CROSSFIT_FOLDS, CROSSFIT_SEED)
    if not np.array_equal(fold_frame.fold_id.to_numpy(np.int8), folds_expected) or not np.array_equal(oof.fold_id.to_numpy(np.int8), folds_expected):
        raise CalibrationContractError("Saved fold assignment mismatch")
    selections = pd.read_csv(audit / "calibration_method_selection.csv")
    metrics = pd.read_csv(audit / "calibration_metrics.csv")
    bootstrap = pd.read_csv(audit / "bootstrap_selection_results.csv")
    table_reconstruction = verify_v2_selection_table(selections, bootstrap)
    registry = json.loads((audit / "calibration_registry.json").read_text(encoding="utf-8"))
    if [item["model_id"] for item in registry["models"]] != selected_models:
        raise CalibrationContractError("Task 10 registry model order mismatch")
    verified_models = []
    full_saved = pd.read_parquet(interim / "calibration_full_fit_selected_predictions.parquet")
    expected_full_columns = ["case_id", "base_order", "target", "model_id", "selected_method", "raw_probability", "full_fit_calibrated_probability", "estimate_role"]
    if list(full_saved.columns) != expected_full_columns or len(full_saved) != len(membership) * len(selected_models):
        raise CalibrationContractError("Saved full-fit selected prediction schema/row count mismatch")
    if set(full_saved.estimate_role.unique()) != {FULL_FIT_ROLE}:
        raise CalibrationContractError("Saved full-fit estimate role mismatch")
    for model_id in selected_models:
        selected_rows = selections[selections.model_id.eq(model_id) & selections.selected]
        if len(selected_rows) != 1: raise CalibrationContractError(f"Selected method count mismatch: {model_id}")
        method = selected_rows.method.iloc[0]
        if table_reconstruction.get(model_id) != method:
            raise CalibrationContractError(f"Independent v2 selection-table reconstruction mismatch: {model_id}")
        item = next(row for row in registry["models"] if row["model_id"] == model_id)
        if item["selected_method"] != method or item["calibration_status"] != "FITTED_AND_VERIFIED" or item["final_evaluation_status"] != "NOT_PREDICTED":
            raise CalibrationContractError(f"Task 10 registry state mismatch: {model_id}")
        raw_probability = raw[f"raw__{model_id}"].to_numpy()
        validate_probabilities(raw_probability, EXPECTED_CALIBRATION)
        raw_metric = official_metrics(membership.target.to_numpy(np.int8), raw_probability)
        candidate_metrics: dict[str, dict[str, Any]] = {}
        for candidate in METHODS:
            candidate_probability = oof[f"oof__{model_id}__{candidate}"].to_numpy()
            validate_probabilities(candidate_probability, EXPECTED_CALIBRATION)
            candidate_metrics[candidate] = official_metrics(membership.target.to_numpy(np.int8), candidate_probability)
        for candidate in METHODS:
            metric = candidate_metrics[candidate]
            reported = metrics[(metrics.model_id == model_id) & (metrics.probability_version == f"oof_{candidate}")].iloc[0]
            for name in ("roc_auc", "average_precision", "log_loss", "brier_score"):
                if abs(metric[name] - float(reported[name])) > 1e-12:
                    raise CalibrationContractError(f"Saved OOF metric mismatch: {model_id}/{candidate}/{name}")
        model_boot = bootstrap[bootstrap.model_id.eq(model_id)].copy()
        if len(model_boot) != BOOTSTRAP_REPLICATES * len(METHODS) or model_boot.replicate_id.nunique() != BOOTSTRAP_REPLICATES:
            raise CalibrationContractError(f"Bootstrap row/replicate mismatch: {model_id}")
        if not {"bootstrap_log_loss", "bootstrap_brier_score", "brier_difference_vs_identity",
                "admissible_numeric_best_reference", "log_loss_difference_vs_admissible_numeric_best"}.issubset(model_boot.columns):
            raise CalibrationContractError(f"Bootstrap v2 schema mismatch: {model_id}")
        recomputed_selection_rows = []
        for candidate in METHODS:
            selection = selections[(selections.model_id == model_id) & (selections.method == candidate)].iloc[0]
            candidate_boot = model_boot[model_boot.method.eq(candidate)]
            se_brier = paired_standard_error(candidate_boot.brier_difference_vs_identity.to_numpy())
            delta_brier = candidate_metrics[candidate]["brier_score"] - candidate_metrics["identity"]["brier_score"]
            extension = "json" if candidate == "identity" else "joblib"
            full_candidate_path = models_root / model_id / f"candidates/full_fit/{candidate}/calibrator.{extension}"
            full_candidate = load_calibrator(candidate, full_candidate_path)
            full_probability = apply_candidate(candidate, full_candidate, raw_probability)
            full_metric = official_metrics(membership.target.to_numpy(np.int8), full_probability)
            if candidate == "identity":
                ranking_pass, auc_change, ap_change = True, 0.0, 0.0
            elif candidate == "logistic":
                ranking_pass, auc_change, ap_change = logistic_ranking_guardrail(
                    raw_metric["roc_auc"], raw_metric["average_precision"], full_metric["roc_auc"],
                    full_metric["average_precision"], float(full_candidate.coef_[0, 0]), LOGISTIC_RANK_TOLERANCE,
                )
            else:
                ranking_pass, auc_change, ap_change = isotonic_ranking_guardrail(
                    raw_metric["roc_auc"], raw_metric["average_precision"], full_metric["roc_auc"],
                    full_metric["average_precision"], ISOTONIC_MAX_DROP,
                )
            brier_pass = brier_guardrail_pass(delta_brier, se_brier)
            if abs(se_brier - float(selection.paired_se_brier_vs_identity)) > 1e-15:
                raise CalibrationContractError(f"Bootstrap Brier SE mismatch: {model_id}/{candidate}")
            if abs(delta_brier - float(selection.delta_brier_vs_identity)) > 1e-15:
                raise CalibrationContractError(f"Saved Brier delta mismatch: {model_id}/{candidate}")
            if abs(auc_change - float(selection.auc_change_on_full_fit_mapping)) > 1e-15 or abs(ap_change - float(selection.ap_change_on_full_fit_mapping)) > 1e-15:
                raise CalibrationContractError(f"Saved ranking change mismatch: {model_id}/{candidate}")
            if bool(selection.passes_brier_guardrail) != brier_pass or bool(selection.passes_ranking_guardrail) != ranking_pass:
                raise CalibrationContractError(f"Saved hard-guardrail boolean mismatch: {model_id}/{candidate}")
            recomputed_selection_rows.append({
                "model_id": model_id, "method": candidate, "complexity_rank": COMPLEXITY[candidate],
                "fit_and_probability_valid": True, "oof_log_loss": candidate_metrics[candidate]["log_loss"],
                "oof_brier": candidate_metrics[candidate]["brier_score"], "delta_brier_vs_identity": delta_brier,
                "paired_se_brier_vs_identity": se_brier, "passes_brier_guardrail": brier_pass,
                "ranking_guardrail_basis": "full_fit_single_mapping",
                "auc_change_on_full_fit_mapping": auc_change, "ap_change_on_full_fit_mapping": ap_change,
                "passes_ranking_guardrail": ranking_pass, "stability_warning": selection.stability_warning,
            })
        recomputed_method, recomputed_rows = mechanical_selection(
            recomputed_selection_rows, model_boot, NUMERIC_TIE_TOLERANCE
        )
        if recomputed_method != method:
            raise CalibrationContractError(f"Saved selected method mismatch: {model_id} expected {recomputed_method}, observed {method}")
        recomputed_by_method = {row["method"]: row for row in recomputed_rows}
        expected_reference = recomputed_rows[0]["admissible_numeric_best_reference"]
        if set(model_boot.admissible_numeric_best_reference.unique()) != {expected_reference}:
            raise CalibrationContractError(f"Bootstrap admissible reference mismatch: {model_id}")
        reference_boot = model_boot[model_boot.method.eq(expected_reference)].sort_values("replicate_id")
        for candidate in METHODS:
            saved = selections[(selections.model_id == model_id) & (selections.method == candidate)].iloc[0]
            expected = recomputed_by_method[candidate]
            candidate_boot = model_boot[model_boot.method.eq(candidate)].sort_values("replicate_id")
            reconstructed_difference = (
                candidate_boot.bootstrap_log_loss.to_numpy(np.float64)
                - reference_boot.bootstrap_log_loss.to_numpy(np.float64)
            )
            np.testing.assert_allclose(
                candidate_boot.log_loss_difference_vs_admissible_numeric_best.to_numpy(np.float64),
                reconstructed_difference, rtol=0, atol=1e-15,
            )
            for name in ("fit_and_probability_valid", "passes_brier_guardrail", "passes_ranking_guardrail",
                         "hard_admissible", "passes_log_loss_one_se", "eligible", "selected"):
                if bool(saved[name]) != bool(expected[name]):
                    raise CalibrationContractError(f"Saved v2 selection boolean mismatch: {model_id}/{candidate}/{name}")
            for name in ("delta_log_loss_vs_admissible_numeric_best", "paired_se_log_loss_vs_admissible_numeric_best"):
                saved_value = saved[name]
                expected_value = expected[name]
                if expected_value is None and pd.isna(saved_value):
                    continue
                if expected_value is None or pd.isna(saved_value) or abs(float(saved_value) - float(expected_value)) > 1e-15:
                    raise CalibrationContractError(f"Saved v2 selection statistic mismatch: {model_id}/{candidate}/{name}")
            if saved.admissible_numeric_best_reference != expected_reference:
                raise CalibrationContractError(f"Saved admissible reference mismatch: {model_id}/{candidate}")
        calibrator_path = Path(item["selected_calibrator_path"])
        verify_selected_calibrator_copy(item)
        calibrator = load_calibrator(method, calibrator_path)
        actual = apply_candidate(method, calibrator, raw[f"raw__{model_id}"].to_numpy())
        expected = full_saved.loc[full_saved.model_id.eq(model_id), "full_fit_calibrated_probability"].to_numpy(np.float64)
        difference = float(np.max(np.abs(actual - expected)))
        if difference > 1e-15:
            raise CalibrationContractError(f"Selected calibrator reload mismatch: {model_id}, max={difference}")
        verified_models.append({"model_id": model_id, "selected_method": method, "reload_max_abs_difference": difference, "status": "PASS"})
    before = pd.read_csv(audit / "input_artifact_hashes_before.csv"); after = pd.read_csv(audit / "input_artifact_hashes_after.csv")
    verify_protected_hash_tables(before, after)
    output_hashes = pd.read_csv(audit / "output_artifact_hashes.csv")
    if output_hashes.empty or output_hashes.path.duplicated().any():
        raise CalibrationContractError("Output artifact hash manifest is empty or duplicated")
    for row in output_hashes.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or sha256_file(path) != row.sha256:
            raise CalibrationContractError(f"Output artifact hash mismatch: {path}")
    return {"status": "PASS", "run_id": run_id, "rows": len(raw), "positives": int(raw.target.sum()),
            "models": verified_models, "final_evaluation_status": "NOT_PREDICTED", "config_hash": stored_hash}


def formal_run(args: argparse.Namespace, repo_root: Path) -> int:
    data_root = args.data_root.resolve(); selected_models = parse_models(args.models)
    interim = data_root / "interim/task10" / args.run_id; models_root = data_root / "models/task10" / args.run_id; audit = data_root / "audits/task10" / args.run_id
    for path in (interim, models_root, audit): ensure_external(path, repo_root)
    existing = [path for path in (interim, models_root, audit) if path.exists()]
    if existing and not args.resume:
        raise FileExistsError("Refusing to overwrite existing Task 10 target: " + ", ".join(map(str, existing)))
    if args.resume and not all(path.exists() for path in (interim, models_root, audit)):
        raise FileNotFoundError("Resume requires all three Task 10 output roots")
    task09 = preflight(data_root, args.base_run_id, selected_models)
    membership, positions, membership_fp, train_fp = load_membership(data_root)
    protected = protected_paths(data_root, args.base_run_id); before_hashes = hash_rows(protected, "before")
    thread_config = configure_threads(args.threads)
    run_config = {
        "version": VERSION, "base_run_id": args.base_run_id, "run_id": args.run_id, "models": selected_models,
        "membership_fingerprint": membership_fp, "training_key_fingerprint": train_fp,
        "crossfit_folds": CROSSFIT_FOLDS, "crossfit_seed": CROSSFIT_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "isotonic_max_auc_drop": ISOTONIC_MAX_DROP, "isotonic_max_ap_drop": ISOTONIC_MAX_DROP,
        "logistic_auc_ap_tolerance": LOGISTIC_RANK_TOLERANCE, "numeric_tie_tolerance": NUMERIC_TIE_TOLERANCE,
        "bootstrap_sd_ddof": BOOTSTRAP_DDOF, "methods": list(METHODS), "complexity_order": list(METHODS),
        "selection_contract": SELECTION_CONTRACT,
        "selection_order": "validity -> Brier/ranking hard admissibility -> admissible Log-loss best -> paired one-SE -> simplest eligible",
        "runtime_versions": runtime_versions(), "threads": thread_config,
        "task10_calibration_sha256": sha256_file(repo_root / "scripts/task10_calibration.py"),
        "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
        "task09_registry_path": str(task09["registry_path"]), "task09_registry_sha256": sha256_file(task09["registry_path"]),
        "task09_run_config_path": str(task09["run_config_path"]), "task09_run_config_sha256": sha256_file(task09["run_config_path"]),
    }
    config_hash = hashlib.sha256(canonical_json(run_config).encode()).hexdigest()
    if args.resume:
        config_path = audit / "run_config.json"
        if not config_path.is_file(): raise CalibrationContractError("Resume config missing")
        saved = json.loads(config_path.read_text(encoding="utf-8")); saved_hash = saved.pop("config_hash", None)
        if saved_hash != config_hash or hashlib.sha256(canonical_json(saved).encode()).hexdigest() != config_hash:
            raise CalibrationContractError("Resume config hash mismatch")
        complete, reason = completion_check(audit / "task_status.json", config_hash, [interim / "calibration_oof_predictions.parquet", audit / "calibration_registry.json"])
        if complete:
            print(json.dumps(verify_saved_run(data_root, args.base_run_id, args.run_id, selected_models), indent=2)); return 0
        raise CalibrationContractError(f"Incomplete prior run retained; automatic partial overwrite is not allowed: {reason}")
    for path in (interim, models_root, audit): path.mkdir(parents=True, exist_ok=False)
    log_path = audit / "calibration.log"; started = time.perf_counter(); stage_times = []; status = "FAILED"; current_stage = "post_preflight_initialization"
    atomic_json(audit / "run_config.json", {**run_config, "config_hash": config_hash})
    atomic_csv(audit / "input_artifact_hashes_before.csv", before_hashes)
    atomic_json(audit / "task_status.json", {"status": "RUNNING", "stage": "preflight_complete", "config_hash": config_hash, "started_at": utc_now(), "artifacts": []})
    try:
        log(f"Task10 start run_id={args.run_id} models={selected_models}", log_path)
        current_stage = "common_fold_assignment"
        folds = make_common_folds(membership.target.to_numpy(np.int8), CROSSFIT_FOLDS, CROSSFIT_SEED)
        fold_summary_frame = fold_summary(folds, membership.target.to_numpy(np.int8))
        if not ((fold_summary_frame.rows == 22900) & (fold_summary_frame.positive_count == 720) & (fold_summary_frame.negative_count == 22180)).all():
            raise CalibrationContractError("Fixed fold count contract failed")
        fold_frame = membership.copy(); fold_frame["fold_id"] = folds
        atomic_parquet(interim / "calibration_fold_assignments.parquet", fold_frame)
        atomic_csv(audit / "fold_assignments_summary.csv", fold_summary_frame.assign(estimate_role=OOF_ROLE))

        current_stage = "base_model_inference"; stage = time.perf_counter()
        raw, probability_audit, resource_rows = infer_raw_predictions(data_root, membership, positions, task09["models"], task09["feature_sets"], selected_models, log_path)
        atomic_parquet(interim / "calibration_raw_predictions.parquet", raw)
        reopened_raw = pd.read_parquet(interim / "calibration_raw_predictions.parquet")
        if not reopened_raw.equals(raw): raise CalibrationContractError("Raw predictions did not reload exactly")
        raw_hash = sha256_file(interim / "calibration_raw_predictions.parquet")
        stage_times.append({"stage": "base_model_inference", "elapsed_seconds": time.perf_counter() - stage})

        oof = fold_frame.copy(); fold_rows: list[dict[str, Any]] = []; parameter_rows: list[dict[str, Any]] = []
        candidate_valid: dict[tuple[str, str], bool] = {}; invalid_reasons: dict[tuple[str, str], str] = {}
        full_calibrators: dict[tuple[str, str], Any] = {}; full_probabilities: dict[tuple[str, str], np.ndarray] = {}
        y = raw.target.to_numpy(np.int8)
        current_stage = "crossfit_and_full_candidate_fits"; stage = time.perf_counter()
        for model_id in selected_models:
            raw_probability = reopened_raw[f"raw__{model_id}"].to_numpy(np.float64, copy=True)
            oof[f"oof__{model_id}__identity"] = raw_probability.copy()
            validate_oof(oof[f"oof__{model_id}__identity"].to_numpy(), np.ones(len(y), dtype=np.int8), len(y))
            probability_audit.append({"model_id": model_id, "method": "identity", "probability_role": "oof", "estimate_role": OOF_ROLE,
                                      **validate_probabilities(raw_probability, len(y)), "status": "PASS"})
            fold_rows.extend(fold_metric_rows(model_id, "identity", folds, y, raw_probability))
            candidate_valid[(model_id, "identity")] = True
            identity_path = models_root / model_id / "candidates/full_fit/identity/calibrator.json"
            identity_meta = identity_path.with_name("calibrator_metadata.json")
            save_calibrator("identity", None, identity_path, identity_meta, {
                "model_id": model_id, "method": "identity", "fit_role": "validation_calibration_full_fit",
                "row_count": len(y), "positive_count": int(y.sum()), "base_model_sha256": task09["models"][model_id]["model_sha256"],
                "raw_prediction_sha256": raw_hash, "crossfit_seed": CROSSFIT_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
            })
            full_calibrators[(model_id, "identity")] = None; full_probabilities[(model_id, "identity")] = raw_probability.copy()
            probability_audit.append({"model_id": model_id, "method": "identity", "probability_role": "full_fit", "estimate_role": FULL_FIT_ROLE,
                                      **validate_probabilities(raw_probability, len(y)), "status": "PASS"})
            parameter_rows.append({"model_id": model_id, "method": "identity", "fit_scope": "full_fit", "fold_id": None,
                                   "estimate_role": FULL_FIT_ROLE, "parameters_json": json.dumps({"mapping": "identity"}), "valid": True})
            for method in ("logistic", "isotonic"):
                assigned = np.zeros(len(y), dtype=np.int8); predictions = np.full(len(y), np.nan, dtype=np.float64); valid = True; reason = ""
                for fold_id in range(CROSSFIT_FOLDS):
                    heldout = folds == fold_id; training = ~heldout
                    try:
                        if method == "logistic": calibrator, metadata = fit_logistic_calibrator(raw_probability[training], y[training])
                        else: calibrator, metadata = fit_isotonic_calibrator(raw_probability[training], y[training])
                        values = apply_candidate(method, calibrator, raw_probability[heldout])
                        predictions[heldout] = values; assigned[heldout] += 1
                        extension = "joblib"; path = models_root / model_id / f"folds/fold_{fold_id}/{method}/calibrator.{extension}"
                        save_calibrator(method, calibrator, path, path.with_name("calibrator_metadata.json"), {
                            **metadata, "model_id": model_id, "method": method, "fold_id": fold_id,
                            "fit_role": "crossfit_training_folds", "fit_row_count": int(training.sum()), "fit_positive_count": int(y[training].sum()),
                            "heldout_row_count": int(heldout.sum()), "heldout_positive_count": int(y[heldout].sum()),
                            "base_model_sha256": task09["models"][model_id]["model_sha256"], "raw_prediction_sha256": raw_hash,
                            "crossfit_seed": CROSSFIT_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
                        })
                        parameter_rows.append({"model_id": model_id, "method": method, "fit_scope": "crossfit", "fold_id": fold_id,
                                               "estimate_role": OOF_ROLE, "parameters_json": json.dumps(json_safe(metadata), sort_keys=True), "valid": True})
                    except Exception as exc:
                        valid = False; reason = f"{type(exc).__name__}: {exc}"
                        parameter_rows.append({"model_id": model_id, "method": method, "fit_scope": "crossfit", "fold_id": fold_id,
                                               "estimate_role": OOF_ROLE, "parameters_json": "{}", "valid": False, "failure_reason": reason})
                        break
                if valid:
                    validate_oof(predictions, assigned, len(y)); oof[f"oof__{model_id}__{method}"] = predictions
                    probability_audit.append({"model_id": model_id, "method": method, "probability_role": "oof", "estimate_role": OOF_ROLE,
                                              **validate_probabilities(predictions, len(y)), "status": "PASS"})
                    fold_rows.extend(fold_metric_rows(model_id, method, folds, y, predictions))
                    try:
                        if method == "logistic": full_calibrator, full_metadata = fit_logistic_calibrator(raw_probability, y)
                        else: full_calibrator, full_metadata = fit_isotonic_calibrator(raw_probability, y)
                        full_probability = apply_candidate(method, full_calibrator, raw_probability)
                        full_path = models_root / model_id / f"candidates/full_fit/{method}/calibrator.joblib"
                        save_calibrator(method, full_calibrator, full_path, full_path.with_name("calibrator_metadata.json"), {
                            **full_metadata, "model_id": model_id, "method": method, "fit_role": "validation_calibration_full_fit",
                            "row_count": len(y), "positive_count": int(y.sum()), "base_model_sha256": task09["models"][model_id]["model_sha256"],
                            "raw_prediction_sha256": raw_hash, "crossfit_seed": CROSSFIT_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
                        })
                        reloaded = joblib.load(full_path); reloaded_probability = apply_candidate(method, reloaded, raw_probability)
                        if float(np.max(np.abs(reloaded_probability - full_probability))) > 1e-15:
                            raise CalibrationContractError("Full-fit calibrator reload mismatch")
                        full_calibrators[(model_id, method)] = full_calibrator; full_probabilities[(model_id, method)] = full_probability
                        probability_audit.append({"model_id": model_id, "method": method, "probability_role": "full_fit", "estimate_role": FULL_FIT_ROLE,
                                                  **validate_probabilities(full_probability, len(y)), "status": "PASS"})
                        parameter_rows.append({"model_id": model_id, "method": method, "fit_scope": "full_fit", "fold_id": None,
                                               "estimate_role": FULL_FIT_ROLE, "parameters_json": json.dumps(json_safe(full_metadata), sort_keys=True), "valid": True})
                    except Exception as exc:
                        valid = False; reason = f"{type(exc).__name__}: {exc}"
                candidate_valid[(model_id, method)] = valid
                if not valid:
                    invalid_reasons[(model_id, method)] = reason
                    oof[f"oof__{model_id}__{method}"] = np.full(len(y), np.nan, dtype=np.float64)
            log(f"Cross-fitting complete model={model_id}", log_path)
        expected_oof_columns = ["case_id", "base_order", "target", "fold_id", *[f"oof__{model_id}__{method}" for model_id in selected_models for method in METHODS]]
        oof = oof[expected_oof_columns]
        atomic_parquet(interim / "calibration_oof_predictions.parquet", oof)
        fold_metrics = pd.DataFrame(fold_rows); atomic_csv(audit / "fold_metrics.csv", fold_metrics)
        atomic_csv(audit / "calibration_parameters.csv", parameter_rows)
        stage_times.append({"stage": "crossfit_and_full_candidate_fits", "elapsed_seconds": time.perf_counter() - stage})

        current_stage = "paired_bootstrap_and_v2_selection"; stage = time.perf_counter(); probability_map = {}; base_metric_lookup = {}
        for model_id in selected_models:
            for method in METHODS:
                if candidate_valid.get((model_id, method), False):
                    values = oof[f"oof__{model_id}__{method}"].to_numpy(np.float64)
                    probability_map[(model_id, method)] = values
                    base_metric_lookup[(model_id, method)] = official_metrics(y, values)
        bootstrap, bootstrap_metadata = paired_stratified_bootstrap(
            y, probability_map, None, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
        )
        # Absolute metrics and identity-relative Brier differences are available
        # before the v2 admissible Log-loss reference is determined.
        atomic_csv(audit / "bootstrap_selection_results.csv", bootstrap)

        stage = time.perf_counter(); selection_rows = []; selected_methods = {}; admissible_references = {}
        required_selection_columns = [
            "model_id", "method", "complexity_rank", "fit_and_probability_valid", "oof_log_loss", "oof_brier",
            "delta_brier_vs_identity", "paired_se_brier_vs_identity", "passes_brier_guardrail",
            "ranking_guardrail_basis", "auc_change_on_full_fit_mapping", "ap_change_on_full_fit_mapping",
            "passes_ranking_guardrail", "hard_admissible", "admissible_numeric_best_reference",
            "delta_log_loss_vs_admissible_numeric_best", "paired_se_log_loss_vs_admissible_numeric_best",
            "passes_log_loss_one_se", "stability_warning", "eligible", "selected", "selection_reason",
        ]
        for model_id in selected_models:
            raw_metric = official_metrics(y, reopened_raw[f"raw__{model_id}"].to_numpy(np.float64))
            rows = []
            for method in METHODS:
                valid = candidate_valid.get((model_id, method), False)
                if not valid:
                    rows.append({"model_id": model_id, "method": method, "complexity_rank": COMPLEXITY[method],
                                 "oof_log_loss": None, "oof_brier": None,
                                 "delta_log_loss_vs_admissible_numeric_best": None,
                                 "paired_se_log_loss_vs_admissible_numeric_best": None,
                                 "passes_log_loss_one_se": False, "delta_brier_vs_identity": None, "paired_se_brier_vs_identity": None,
                                 "passes_brier_guardrail": False, "ranking_guardrail_basis": "full_fit_single_mapping",
                                 "auc_change_on_full_fit_mapping": None, "ap_change_on_full_fit_mapping": None,
                                 "passes_ranking_guardrail": False, "fit_and_probability_valid": False, "eligible": False,
                                 "selected": False, "fit_failure_reason": invalid_reasons.get((model_id, method), "INVALID"),
                                 "selection_reason": invalid_reasons.get((model_id, method), "INVALID"),
                                 "stability_warning": "NOT_CHECKABLE"})
                    continue
                metric = base_metric_lookup[(model_id, method)]; identity_metric = base_metric_lookup[(model_id, "identity")]
                boot = bootstrap[(bootstrap.model_id == model_id) & (bootstrap.method == method)]
                se_brier = paired_standard_error(boot.brier_difference_vs_identity.to_numpy())
                delta_brier = metric["brier_score"] - identity_metric["brier_score"]
                full_metric = official_metrics(y, full_probabilities[(model_id, method)])
                if method == "identity":
                    ranking_pass, auc_change, ap_change = True, 0.0, 0.0
                elif method == "logistic":
                    beta = float(full_calibrators[(model_id, method)].coef_[0, 0])
                    ranking_pass, auc_change, ap_change = logistic_ranking_guardrail(raw_metric["roc_auc"], raw_metric["average_precision"], full_metric["roc_auc"], full_metric["average_precision"], beta, LOGISTIC_RANK_TOLERANCE)
                else:
                    ranking_pass, auc_change, ap_change = isotonic_ranking_guardrail(raw_metric["roc_auc"], raw_metric["average_precision"], full_metric["roc_auc"], full_metric["average_precision"], ISOTONIC_MAX_DROP)
                rows.append({"model_id": model_id, "method": method, "complexity_rank": COMPLEXITY[method],
                             "oof_log_loss": metric["log_loss"], "oof_brier": metric["brier_score"],
                             "delta_brier_vs_identity": delta_brier,
                             "paired_se_brier_vs_identity": se_brier, "passes_brier_guardrail": brier_guardrail_pass(delta_brier, se_brier),
                             "ranking_guardrail_basis": "full_fit_single_mapping", "auc_change_on_full_fit_mapping": auc_change,
                             "ap_change_on_full_fit_mapping": ap_change, "passes_ranking_guardrail": ranking_pass,
                             "fit_and_probability_valid": True, "stability_warning": "" if method == "identity" else stability_warning(fold_metrics[fold_metrics.model_id.eq(model_id)], method)})
            model_bootstrap = bootstrap[bootstrap.model_id.eq(model_id)]
            selected, annotated = mechanical_selection(rows, model_bootstrap, NUMERIC_TIE_TOLERANCE)
            selected_methods[model_id] = selected
            admissible_references[model_id] = annotated[0]["admissible_numeric_best_reference"]
            selection_rows.extend(annotated)
            # Failure-safe, atomic evidence after every processed model.
            incremental = pd.DataFrame(selection_rows).reindex(columns=required_selection_columns)
            atomic_csv(audit / "calibration_method_selection.csv", incremental)
        selection = pd.DataFrame(selection_rows)[required_selection_columns]
        bootstrap = annotate_bootstrap_references(bootstrap, admissible_references)
        atomic_csv(audit / "bootstrap_selection_results.csv", bootstrap)
        interval_rows = []
        for (model_id, method), group in bootstrap.groupby(["model_id", "method"], sort=False):
            for metric_name in ("bootstrap_log_loss", "bootstrap_brier_score", "log_loss_difference_vs_admissible_numeric_best", "brier_difference_vs_identity"):
                values = group[metric_name].to_numpy(np.float64); interval = np.quantile(values, [0.025, 0.975], method="linear")
                interval_rows.append({"model_id": model_id, "method": method, "metric": metric_name, "replicates": len(values),
                                      "mean": float(values.mean()), "sd_ddof_1": float(np.std(values, ddof=1)),
                                      "q025": float(interval[0]), "q975": float(interval[1]), "quantile_method": "linear", "estimate_role": OOF_ROLE})
        atomic_csv(audit / "bootstrap_interval_summary.csv", interval_rows)
        stage_times.append({"stage": "paired_bootstrap_and_v2_selection", "elapsed_seconds": time.perf_counter() - stage})

        current_stage = "selected_calibrator_persistence"; stage = time.perf_counter(); full_long_parts = []; registry_models = []
        for model_id in selected_models:
            method = selected_methods[model_id]; extension = "json" if method == "identity" else "joblib"
            source = models_root / model_id / f"candidates/full_fit/{method}/calibrator.{extension}"
            selected_dir = models_root / model_id / "selected"; selected_dir.mkdir(parents=True, exist_ok=True)
            destination = selected_dir / f"calibrator.{extension}"; temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
            shutil.copyfile(source, temporary); os.replace(temporary, destination)
            if sha256_file(source) != sha256_file(destination): raise CalibrationContractError(f"Selected calibrator byte copy mismatch: {model_id}")
            selected_probability = full_probabilities[(model_id, method)]
            metadata = {"model_id": model_id, "selected_method": method, "source_full_fit_path": str(source),
                        "source_full_fit_sha256": sha256_file(source), "selected_calibrator_path": str(destination),
                        "selected_calibrator_sha256": sha256_file(destination), "copy_byte_identical": True,
                        "fit_role": "validation_calibration_full_fit", "row_count": len(y), "positive_count": int(y.sum()),
                        "base_model_sha256": task09["models"][model_id]["model_sha256"], "raw_prediction_sha256": raw_hash,
                        "crossfit_folds": CROSSFIT_FOLDS, "crossfit_seed": CROSSFIT_SEED,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                        "estimate_role": FULL_FIT_ROLE}
            atomic_json(selected_dir / "calibrator_metadata.json", metadata)
            reloaded = load_calibrator(method, destination); reloaded_probability = apply_candidate(method, reloaded, reopened_raw[f"raw__{model_id}"].to_numpy(np.float64))
            reload_difference = float(np.max(np.abs(reloaded_probability - selected_probability)))
            if reload_difference > 1e-15: raise CalibrationContractError(f"Selected calibrator reload failure: {model_id}")
            part = membership.copy(); part["model_id"] = model_id; part["selected_method"] = method
            part["raw_probability"] = reopened_raw[f"raw__{model_id}"].to_numpy(np.float64)
            part["full_fit_calibrated_probability"] = selected_probability; part["estimate_role"] = FULL_FIT_ROLE
            full_long_parts.append(part)
            selected_row = selection[(selection.model_id == model_id) & selection.selected].iloc[0]
            item = task09["models"][model_id]
            registry_models.append({"model_id": model_id, "family": item["family"], "representation": item["representation"],
                "ordered_predictors": item["ordered_predictors"], "task09_model_path": item["model_path"], "task09_model_sha256": item["model_sha256"],
                "task09_registry_path": str(task09["registry_path"]), "task09_registry_sha256": sha256_file(task09["registry_path"]),
                "raw_prediction_path": str(interim / "calibration_raw_predictions.parquet"), "raw_prediction_sha256": raw_hash,
                "selected_method": method, "selected_calibrator_path": str(destination), "selected_calibrator_sha256": sha256_file(destination),
                "selected_calibrator_source_path": str(source), "selected_calibrator_source_sha256": sha256_file(source),
                "reload_max_abs_difference": reload_difference, "membership_fingerprint": membership_fp,
                "crossfit_folds": CROSSFIT_FOLDS, "crossfit_seed": CROSSFIT_SEED, "bootstrap_design": "paired_target_stratified_application",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                "admissible_numeric_best_reference": selected_row.admissible_numeric_best_reference,
                "delta_log_loss_vs_admissible_numeric_best": float(selected_row.delta_log_loss_vs_admissible_numeric_best),
                "paired_se_log_loss_vs_admissible_numeric_best": float(selected_row.paired_se_log_loss_vs_admissible_numeric_best),
                "passes_log_loss_one_se": bool(selected_row.passes_log_loss_one_se),
                "delta_brier_vs_identity": float(selected_row.delta_brier_vs_identity),
                "paired_se_brier_vs_identity": float(selected_row.paired_se_brier_vs_identity),
                "passes_brier_guardrail": bool(selected_row.passes_brier_guardrail),
                "ranking_guardrail_basis": selected_row.ranking_guardrail_basis,
                "auc_change_on_full_fit_mapping": float(selected_row.auc_change_on_full_fit_mapping),
                "ap_change_on_full_fit_mapping": float(selected_row.ap_change_on_full_fit_mapping),
                "passes_ranking_guardrail": bool(selected_row.passes_ranking_guardrail), "calibration_status": "FITTED_AND_VERIFIED",
                "decision_threshold_status": "NOT_SELECTED", "final_evaluation_status": "NOT_PREDICTED"})
            artifacts = [path for path in (models_root / model_id).rglob("*") if path.is_file() and path.name != "calibration_status.json"]
            atomic_json(models_root / model_id / "calibration_status.json", {"status": "COMPLETE", "config_hash": config_hash,
                "model_id": model_id, "selected_method": method, "artifacts": [{"path": str(path), "sha256": sha256_file(path)} for path in sorted(artifacts)]})
        full_long = pd.concat(full_long_parts, ignore_index=True)[["case_id", "base_order", "target", "model_id", "selected_method", "raw_probability", "full_fit_calibrated_probability", "estimate_role"]]
        atomic_parquet(interim / "calibration_full_fit_selected_predictions.parquet", full_long)
        stage_times.append({"stage": "selected_calibrator_persistence", "elapsed_seconds": time.perf_counter() - stage})

        current_stage = "calibration_diagnostics"; stage = time.perf_counter(); metric_rows = []; bins10_rows = []; bins20_rows = []; tail_rows = []
        for model_id in selected_models:
            method = selected_methods[model_id]; raw_values = reopened_raw[f"raw__{model_id}"].to_numpy(np.float64)
            versions = [
                ("raw_identity", "identity", RAW_ROLE, raw_values),
                ("oof_identity", "identity", OOF_ROLE, oof[f"oof__{model_id}__identity"].to_numpy(np.float64)),
                ("oof_logistic", "logistic", OOF_ROLE, oof[f"oof__{model_id}__logistic"].to_numpy(np.float64)),
                ("oof_isotonic", "isotonic", OOF_ROLE, oof[f"oof__{model_id}__isotonic"].to_numpy(np.float64)),
                ("oof_selected", method, OOF_ROLE, oof[f"oof__{model_id}__{method}"].to_numpy(np.float64)),
                ("full_fit_selected_in_sample_diagnostic", method, FULL_FIT_ROLE, full_probabilities[(model_id, method)]),
            ]
            for version, candidate, estimate_role, values in versions:
                _, endpoint = __import__("task10_calibration").endpoint_protected_logit(values)
                metrics_value, bins10, bins20 = metric_with_calibration_diagnostics(y, values, endpoint["protected_endpoint_count"])
                metric_rows.append({"model_id": model_id, "probability_version": version, "method": candidate, "estimate_role": estimate_role, **metrics_value})
                for frame, destination_rows in ((bins10, bins10_rows), (bins20, bins20_rows)):
                    for row in frame.to_dict("records"):
                        destination_rows.append({"model_id": model_id, "probability_version": version, "method": candidate, "estimate_role": estimate_role, **row})
            for version, candidate, values in (("raw_identity", "identity", raw_values), ("oof_selected", method, oof[f"oof__{model_id}__{method}"].to_numpy(np.float64))):
                for row in top_risk_summary(y, values):
                    tail_rows.append({"model_id": model_id, "probability_version": version, "method": candidate,
                                      "estimate_role": RAW_ROLE if version == "raw_identity" else OOF_ROLE, **row,
                                      "diagnostic_type": "TOP_RISK_NOT_POLICY_THRESHOLD"})
        metrics_frame = pd.DataFrame(metric_rows)
        metric_columns = ["model_id", "probability_version", "method", "estimate_role", "N", "positive_count", "positive_rate",
            "roc_auc", "average_precision", "log_loss", "brier_score", "mean_probability", "min_probability", "max_probability",
            "exact_zero_count", "exact_one_count", "nonfinite_count", "out_of_bounds_count", "protected_endpoint_count",
            "calibration_in_the_large_intercept", "calibration_joint_intercept", "calibration_slope", "ece_10", "ece_20",
            "max_abs_bin_gap_10", "max_abs_bin_gap_20"]
        metrics_frame = metrics_frame[metric_columns]
        atomic_csv(audit / "calibration_metrics.csv", metrics_frame); atomic_csv(audit / "calibration_bins_10.csv", bins10_rows)
        atomic_csv(audit / "calibration_bins_20.csv", bins20_rows); atomic_csv(audit / "calibration_tail_summary.csv", tail_rows)
        stage_times.append({"stage": "calibration_diagnostics", "elapsed_seconds": time.perf_counter() - stage})

        current_stage = "aggregate_figure"; stage = time.perf_counter()
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
        bins10_frame = pd.DataFrame(bins10_rows)
        for axis, model_id in zip(axes.flat, selected_models):
            method = selected_methods[model_id]
            raw_bins = bins10_frame[(bins10_frame.model_id == model_id) & (bins10_frame.probability_version == "raw_identity")]
            selected_bins = bins10_frame[(bins10_frame.model_id == model_id) & (bins10_frame.probability_version == "oof_selected")]
            axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="45-degree")
            axis.plot(raw_bins.mean_probability, raw_bins.observed_rate, marker="o", label="raw 10-bin")
            axis.plot(selected_bins.mean_probability, selected_bins.observed_rate, marker="o", label="selected OOF 10-bin")
            axis.set_title(f"{model_id} ({method})"); axis.grid(alpha=.25); axis.set_xlim(0, .35); axis.set_ylim(0, .35)
        axes[1, 0].set_xlabel("Mean predicted probability"); axes[1, 1].set_xlabel("Mean predicted probability"); axes[1, 2].set_xlabel("Mean predicted probability")
        axes[0, 0].set_ylabel("Observed event rate"); axes[1, 0].set_ylabel("Observed event rate")
        axes[0, 0].legend(fontsize=8); fig.suptitle("Validation-calibration curves (OOF selected estimates)"); fig.tight_layout()
        figure_path = audit / "calibration_curves.png"; fig.savefig(figure_path, dpi=160); plt.close(fig)
        stage_times.append({"stage": "aggregate_figure", "elapsed_seconds": time.perf_counter() - stage})

        registry = {"run_id": args.run_id, "base_run_id": args.base_run_id, "status": "VERIFY_PENDING",
                    "version": VERSION, "selection_contract": SELECTION_CONTRACT,
                    "membership_fingerprint": membership_fp, "crossfit_seed": CROSSFIT_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
                    "models": registry_models, "decision_threshold_status": "NOT_SELECTED", "final_evaluation_status": "NOT_PREDICTED"}
        atomic_json(audit / "calibration_registry.json", registry)
        atomic_csv(audit / "probability_audit.csv", probability_audit)
        report_lines = ["# Task 10 概率校准报告", "", "状态：**VERIFY_PENDING**。", "", "六个 Task 09 基础模型均未重新训练。本任务只使用 `validation_calibration` 的 114,500 条申请和 3,600 个正例。", "", "## 机械选择", "", "| 模型 | 方法 | OOF Log loss | OOF Brier | one-SE 差值 / SE | Brier 差值 / SE | 排序护栏 | 警告 |", "|---|---|---:|---:|---:|---:|---|---|"]
        for model_id in selected_models:
            row = selection[(selection.model_id == model_id) & selection.selected].iloc[0]
            report_lines.append(f"| {model_id} | {row.method} | {row.oof_log_loss:.8f} | {row.oof_brier:.8f} | {row.delta_log_loss_vs_admissible_numeric_best:.8g} / {row.paired_se_log_loss_vs_admissible_numeric_best:.8g} | {row.delta_brier_vs_identity:.8g} / {row.paired_se_brier_vs_identity:.8g} | {row.passes_ranking_guardrail} | {row.stability_warning or ''} |")
        report_lines += ["", "OOF 指标是校准开发估计，不是最终无偏评估。`full_fit` 概率只用于工件和结构核验，不能支持性能改善声明。最终评估未预测，阈值未选择。", "", "本结果不能证明未来时间稳定性、借款人独立性、因果关系、生产可用性，也不能识别历史拒绝申请的反事实结果。"]
        atomic_text(audit / "calibration_report_zh.md", "\n".join(report_lines) + "\n")

        current_stage = "formal_tests_and_git_diff_check"; evidence = []
        evidence.append(run_test_command([sys.executable, "-m", "pytest", "-q", "tests/test_task10_calibration.py", "tests/test_calibrate_baseline_models.py"], repo_root, audit, "focused_task10_tests"))
        evidence.append(run_test_command([sys.executable, "-m", "pytest", "-q"], repo_root, audit, "full_repository_tests"))
        evidence.append(run_test_command(["git", "diff", "--check"], repo_root, audit, "git_diff_check"))
        atomic_csv(audit / "execution_evidence.csv", evidence)
        if any(row["status"] != "PASS" for row in evidence): raise CalibrationContractError("Formal execution tests or git diff check failed")

        current_stage = "protected_after_hashes"; after_hashes = hash_rows(protected, "after"); atomic_csv(audit / "input_artifact_hashes_after.csv", after_hashes)
        before_map = {row["path"]: row["sha256"] for row in before_hashes}
        if any(before_map[row["path"]] != row["sha256"] for row in after_hashes):
            raise CalibrationContractError("Protected input changed during Task 10")
        validation_rows = [
            {"check": "calibration_membership", "status": "PASS", "actual": f"{len(y)} rows/{int(y.sum())} positives", "expected": "114500 rows/3600 positives", "reason": "Predicate-filtered validation_calibration labels"},
            {"check": "common_folds", "status": "PASS", "actual": "5 x 22900 rows, 720 positives", "expected": "exact fixed counts", "reason": "One common assignment reused"},
            {"check": "six_selected_methods", "status": "PASS", "actual": int(selection.selected.sum()), "expected": len(selected_models), "reason": "Mechanical selection"},
            {"check": "bootstrap_rows", "status": "PASS", "actual": len(bootstrap), "expected": BOOTSTRAP_REPLICATES * len(probability_map), "reason": "Valid model-method pairs only"},
            {"check": "final_evaluation", "status": "NOT_CHECKED", "actual": "NOT_PREDICTED", "expected": "frozen", "reason": "Evaluation target/predictions prohibited"},
            {"check": "protected_hashes", "status": "PASS", "actual": len(after_hashes), "expected": len(before_hashes), "reason": "Before/after SHA-256 unchanged"},
        ]
        atomic_csv(audit / "validation_results.csv", validation_rows)
        stage_times.append({"stage": "tests_docs_and_hash_reconciliation", "elapsed_seconds": time.perf_counter() - stage})
        resource_rows += stage_times
        resource_rows.append({"stage": "whole_run_start_to_pre_final_verification", "elapsed_seconds": time.perf_counter() - started,
                              "ru_maxrss_bytes_end": ru_maxrss_bytes(), "ru_maxrss_scope": "cumulative process peak"})
        atomic_csv(audit / "resource_usage.csv", resource_rows)
        summary = {"task": "TASK10_PROBABILITY_CALIBRATION", "status": "VERIFY_PENDING", "run_id": args.run_id,
                   "version": VERSION, "selection_contract": SELECTION_CONTRACT,
                   "completed_at": utc_now(), "rows": len(y), "positives": int(y.sum()), "negatives": int((y == 0).sum()),
                   "selected_methods": selected_methods, "bootstrap": bootstrap_metadata, "runtime_versions": runtime_versions(),
                   "thread_configuration": thread_config, "final_evaluation_status": "NOT_PREDICTED",
                   "decision_threshold_status": "NOT_SELECTED", "limitations": ["OOF calibration-development estimates are not final unbiased evaluation.", "Application unit; no reliable borrower identifier.", "Random design does not establish future-time stability."]}
        atomic_json(audit / "calibration_summary.json", summary)

        output_files = [path for root in (interim, models_root, audit) for path in root.rglob("*") if path.is_file()]
        output_files = [path for path in output_files if path.name not in {"output_artifact_hashes.csv", "task_status.json"}]
        atomic_csv(audit / "output_artifact_hashes.csv", [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in sorted(output_files)])
        atomic_json(audit / "task_status.json", {"status": "VERIFY_PENDING", "stage": "saved_artifact_verification", "config_hash": config_hash,
            "artifacts": [{"path": str(path), "sha256": sha256_file(path)} for path in [interim / "calibration_raw_predictions.parquet", interim / "calibration_oof_predictions.parquet", audit / "calibration_registry.json"]]})
        current_stage = "built_in_saved_artifact_verification"; preliminary_verification = verify_saved_run(data_root, args.base_run_id, args.run_id, selected_models)
        registry["status"] = "COMPLETE"; registry["verification"] = {"status": preliminary_verification["status"], "stage": "preliminary_saved_artifact_verification"}; atomic_json(audit / "calibration_registry.json", registry)
        summary["status"] = "COMPLETE"; summary["verification"] = {"status": preliminary_verification["status"], "stage": "preliminary_saved_artifact_verification"}; summary["completed_at"] = utc_now(); atomic_json(audit / "calibration_summary.json", summary)
        report_lines[2] = "状态：**COMPLETE**。保存工件独立核验通过。"; atomic_text(audit / "calibration_report_zh.md", "\n".join(report_lines) + "\n")
        status = "COMPLETE"
        log(f"Task10 COMPLETE selected_methods={selected_methods}", log_path)
        final_output_files = [path for root in (interim, models_root, audit) for path in root.rglob("*") if path.is_file()]
        final_output_files = [path for path in final_output_files if path.name not in {"output_artifact_hashes.csv", "task_status.json"}]
        atomic_csv(audit / "output_artifact_hashes.csv", [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in sorted(final_output_files)])
        verification = verify_saved_run(data_root, args.base_run_id, args.run_id, selected_models)
        status_artifacts = [interim / "calibration_raw_predictions.parquet", interim / "calibration_fold_assignments.parquet", interim / "calibration_oof_predictions.parquet", interim / "calibration_full_fit_selected_predictions.parquet", audit / "calibration_registry.json", audit / "calibration_method_selection.csv"]
        atomic_json(audit / "task_status.json", {"status": "COMPLETE", "stage": "complete", "config_hash": config_hash, "completed_at": utc_now(),
            "artifacts": [{"path": str(path), "sha256": sha256_file(path)} for path in status_artifacts], "verification": verification,
            "protected_input_integrity": {"status": "PASS", "before_count": len(before_hashes), "after_count": len(after_hashes)}})
        print(json.dumps({"status": status, "run_id": args.run_id, "selected_methods": selected_methods, "audit_dir": str(audit), "interim_dir": str(interim), "models_dir": str(models_root)}, indent=2))
        return 0
    except Exception as exc:
        protected_integrity: dict[str, Any]
        try:
            failure_after_hashes = hash_rows(protected, "after")
            atomic_csv(audit / "input_artifact_hashes_after.csv", failure_after_hashes)
            before_map = {row["path"]: row["sha256"] for row in before_hashes}
            changed = [row["path"] for row in failure_after_hashes if before_map.get(row["path"]) != row["sha256"]]
            protected_integrity = {"status": "PASS" if not changed else "FAIL", "before_count": len(before_hashes),
                                   "after_count": len(failure_after_hashes), "changed_paths": changed}
        except Exception as hash_exc:
            protected_integrity = {"status": "SECONDARY_FAILURE", "error_type": type(hash_exc).__name__, "message": str(hash_exc)}
        atomic_json(audit / "task_status.json", {"status": "FAILED", "stage": current_stage, "config_hash": config_hash, "failed_at": utc_now(),
            "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=80), "resume_safe": False,
            "reason": "Partial outputs retained; use a separately authorized new run ID unless exact resume validation succeeds.",
            "protected_input_integrity": protected_integrity})
        log(f"Task10 FAILED {type(exc).__name__}: {exc}", log_path)
        raise


def main() -> int:
    args = parse_args(); repo_root = Path(__file__).resolve().parents[1]; selected_models = parse_models(args.models)
    if args.verify_only:
        data_root = args.data_root.resolve()
        verification = verify_saved_run(data_root, args.base_run_id, args.run_id, selected_models)
        if selected_models == MODEL_ORDER:
            audit = data_root / "audits/task10" / args.run_id
            status_path = audit / "task_status.json"
            selection_path = audit / "calibration_method_selection.csv"
            metrics_path = audit / "calibration_metrics.csv"
            if status_path.is_file() and selection_path.is_file() and metrics_path.is_file():
                selections = pd.read_csv(selection_path)
                metrics = pd.read_csv(metrics_path)
                build_documentation(repo_root, selections, metrics, args.run_id)
                status_payload = json.loads(status_path.read_text(encoding="utf-8"))
                status_payload["separate_verify_only"] = {"status": "PASS", "verified_at": utc_now(), "command_scope": "all"}
                atomic_json(status_path, status_payload)
        print(json.dumps(verification, indent=2)); return 0
    return formal_run(args, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
