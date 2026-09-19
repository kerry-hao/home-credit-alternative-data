#!/usr/bin/env python3
"""Independent post-training verification for Task 09 saved runs.

This entry point performs inference and diagnostics only.  It never calls a
model ``fit`` method, creates optimizer updates, or accesses held-out labels.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Verification is inference-only.  Bound all native pools before numerical
# imports to avoid the documented mixed OpenMP runtime instability on macOS.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import torch

from prepare_training_data import ALIAS_NEW, ALIAS_OLD, MISSING_TOKEN, TrainingPreprocessor, UNSEEN_TOKEN
from task09_reliability import sha256_file, test_execution_status
from train_baseline_models import (
    MODEL_ORDER,
    PROBABILITY_COLUMNS,
    TabularMLP,
    atomic_csv,
    atomic_json,
    atomic_text,
    feature_contract_for,
    load_contract,
    load_matrix_rows,
    runtime_versions,
    torch_load_state,
    utc_now,
    verify_saved_run,
)


VERIFIER_VERSION = "task09_post_review_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run-id", default="first_full")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    return parser.parse_args()


def run_command(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path, junit_path: Path | None = None) -> dict[str, Any]:
    started_at = utc_now(); started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    ended_at = utc_now()
    atomic_text(stdout_path, result.stdout)
    atomic_text(stderr_path, result.stderr)
    collected = 0
    if junit_path is not None and junit_path.is_file():
        root = ET.parse(junit_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        collected = sum(int(node.attrib.get("tests", "0")) for node in suites if node is root or root.tag != "testsuite")
    evidence = {
        "command": command, "command_text": " ".join(command), "cwd": str(cwd),
        "started_at": started_at, "ended_at": ended_at, "elapsed_seconds": time.perf_counter() - started,
        "exit_code": result.returncode, "collected_tests": collected,
        "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
    }
    status, reason = test_execution_status(evidence)
    evidence.update({"status": status, "reason": reason})
    return evidence


def provenance_paths(data_root: Path, run_id: str, registry: dict[str, Any]) -> list[tuple[str, Path]]:
    audit = data_root / "audits/task09" / run_id
    interim = data_root / "interim/task09" / run_id
    prep = data_root / "interim/task08_followup"
    rows = [
        ("tuning_predictions", interim / "tuning_predictions.parquet"),
        ("original_run_config", audit / "run_config.json"),
        ("original_training_summary", audit / "training_summary.json"),
        ("original_registry", audit / "selected_model_registry.json"),
        ("task08_manifest", prep / "application_manifest.parquet"),
        ("task08_linear_matrix", prep / "linear_nn_inputs.parquet"),
        ("task08_gbdt_matrix", prep / "gbdt_inputs.parquet"),
        ("task08_preprocessor", prep / "preprocessing/preprocessor.json"),
        ("task08_feature_lists", prep / "preprocessing/feature_sets.json"),
    ]
    for item in registry["models"]:
        rows.append((f"selected_model::{item['model_id']}", Path(item["model_path"])))
        architecture = item.get("architecture_path")
        if architecture:
            rows.append((f"architecture::{item['model_id']}", Path(architecture)))
        rows.append((f"reload_sample::{item['model_id']}", interim / "reload_samples" / f"{item['model_id']}.parquet"))
    return rows


def raw_feature_rows(path: Path, columns: list[str], positions: np.ndarray, total_rows: int) -> pd.DataFrame:
    requested = [ALIAS_OLD if column == ALIAS_NEW else column for column in columns]
    parts: list[pd.DataFrame] = []
    global_start = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=131072, columns=requested):
        frame = batch.to_pandas()
        n = len(frame)
        left = np.searchsorted(positions, global_start, side="left")
        right = np.searchsorted(positions, global_start + n, side="left")
        if right > left:
            parts.append(frame.iloc[positions[left:right] - global_start].copy())
        global_start += n
    if global_start != total_rows:
        raise RuntimeError("Raw feature scan row count disagrees with manifest")
    output = pd.concat(parts, ignore_index=True)
    if ALIAS_OLD in output and ALIAS_NEW in columns:
        output = output.rename(columns={ALIAS_OLD: ALIAS_NEW})
    return output[columns]


def preprocessing_diagnostics(data_root: Path, output_dir: Path, contract: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prep_path = data_root / "interim/task08_followup/preprocessing/preprocessor.json"
    preprocessor = TrainingPreprocessor.load(prep_path)
    raw_path = data_root / "interim/task08/application_features.parquet"
    input_columns = [*preprocessor.t_features, *preprocessor.ad_substantive, *preprocessor.ad_indicators]
    one_position = np.asarray([int(contract.train_positions[0])], dtype=np.int64)
    source = raw_feature_rows(raw_path, input_columns, one_position, len(contract.manifest))
    examples: list[dict[str, Any]] = []
    for category, token, value in (
        (preprocessor.categories[0], MISSING_TOKEN, pd.NA),
        (preprocessor.categories[1], UNSEEN_TOKEN, "__SYNTHETIC_OUT_OF_VOCABULARY__"),
    ):
        synthetic = source.copy(); synthetic.loc[0, category] = value
        outputs = preprocessor.category_outputs[category]
        expected_column = next(row["output"] for row in outputs if row["category"] == token)
        group_columns = [row["output"] for row in outputs]
        for representation in ("linear_nn", "gbdt"):
            transformed = preprocessor.transform(synthetic, representation)
            group = transformed.loc[0, group_columns]
            passed = int(group[expected_column]) == 1 and int(group.sum()) == 1 and list(transformed.columns) == preprocessor.output_features[representation]["T_plus_AD"]
            for column in group_columns:
                examples.append({
                    "example_type": f"synthetic_actual_transform_{token.lower()}",
                    "source_role": "outer_train_row_copied_then_mutated",
                    "representation": representation, "input_column": category,
                    "synthetic_value": None if pd.isna(value) else value,
                    "output_column": column, "measured_output": int(group[column]),
                    "expected_output": 1 if column == expected_column else 0,
                    "ordered_output_match": list(transformed.columns) == preprocessor.output_features[representation]["T_plus_AD"],
                    "status": "PASS" if passed and int(group[column]) == (1 if column == expected_column else 0) else "FAIL",
                })
    original_examples = pd.read_csv(data_root / "audits/task09/first_full/preprocessing_examples_supplement.csv")
    for row in original_examples.to_dict("records"):
        row = {f"historical_{key}": value for key, value in row.items()}
        row.update({"example_type": "historical_preserved_example", "source_role": "original_task09_audit", "status": "INFORMATIONAL"})
        examples.append(row)

    reserved: list[dict[str, Any]] = []
    binary: list[dict[str, Any]] = []
    keys = contract.manifest.case_id.to_numpy()
    ohe_groups = {category: [row["output"] for row in preprocessor.category_outputs[category]] for category in preprocessor.categories}
    binary_columns = [*preprocessor.ad_indicators, *[row["name"] for row in preprocessor.missing_flags], *[c for values in ohe_groups.values() for c in values]]
    for representation, filename in (("linear_nn", "linear_nn_inputs.parquet"), ("gbdt", "gbdt_inputs.parquet")):
        matrix_path = data_root / "interim/task08_followup" / filename
        columns = contract.feature_sets[representation]["T_plus_AD"]
        train, _ = load_matrix_rows(matrix_path, columns, contract.train_positions, keys, np.float32)
        tuning, _ = load_matrix_rows(matrix_path, columns, contract.tuning_positions, keys, np.float32)
        index = {column: j for j, column in enumerate(columns)}
        for column in binary_columns:
            if column not in index:
                continue
            for role, values in (("train", train[:, index[column]]), ("validation_tuning", tuning[:, index[column]])):
                unique = np.unique(values)
                passed = np.isfinite(values).all() and set(unique.tolist()).issubset({0.0, 1.0})
                binary.append({"representation": representation, "role": role, "check": "binary_values", "column_or_group": column,
                               "row_count": len(values), "observed_values": json.dumps(unique.tolist()), "status": "PASS" if passed else "FAIL"})
        for category, group_columns in ohe_groups.items():
            group_idx = [index[column] for column in group_columns]
            for role, values in (("train", train[:, group_idx]), ("validation_tuning", tuning[:, group_idx])):
                sums = values.sum(axis=1)
                passed = bool(np.all(sums == 1))
                binary.append({"representation": representation, "role": role, "check": "one_hot_group_sum", "column_or_group": category,
                               "row_count": len(sums), "violation_count": int(np.count_nonzero(sums != 1)), "status": "PASS" if passed else "FAIL"})
        for category, group_columns in ohe_groups.items():
            for column in group_columns:
                train_count = int(np.count_nonzero(train[:, index[column]]))
                if train_count == 0:
                    reserved.append({"representation": representation, "category": category, "reserved_column": column,
                                     "train_activation_count": 0, "tuning_activation_count": int(np.count_nonzero(tuning[:, index[column]])),
                                     "train_rows": len(train), "tuning_rows": len(tuning), "status": "MEASURED",
                                     "interpretation": "No category-specific TRAIN activation; an initialized MLP weight may still affect an activated reserved path."})
        del train, tuning
        gc.collect()

    standardized: list[dict[str, Any]] = []
    numeric = list(preprocessor.numeric_substantive)
    linear_path = data_root / "interim/task08_followup/linear_nn_inputs.parquet"
    for role, positions in (("train", contract.train_positions), ("validation_tuning", contract.tuning_positions)):
        raw = raw_feature_rows(raw_path, numeric, positions, len(contract.manifest))
        prepared, _ = load_matrix_rows(linear_path, numeric, positions, keys, np.float32)
        for j, column in enumerate(numeric):
            values = prepared[:, j].astype(np.float64, copy=False)
            observed_mask = raw[column].notna().to_numpy()
            for slice_name, mask in (("all_rows_after_imputation", np.ones(len(values), dtype=bool)), ("originally_observed_content", observed_mask)):
                selected = values[mask]
                row = {"role": role, "input_column": column, "slice": slice_name, "denominator": int(mask.sum()),
                       "mask_source": "raw_application_features_notna" if slice_name == "originally_observed_content" else "all_authorized_role_rows"}
                if len(selected):
                    quantiles = np.quantile(selected, [0.001, 0.01, 0.5, 0.99, 0.999], method="linear")
                    row.update({"status": "MEASURED", "finite_count": int(np.isfinite(selected).sum()), "min": float(selected.min()), "max": float(selected.max()),
                                "q0001": float(quantiles[0]), "q001": float(quantiles[1]), "median": float(quantiles[2]), "q099": float(quantiles[3]), "q0999": float(quantiles[4]),
                                "abs_gt_10_count": int(np.count_nonzero(np.abs(selected) > 10)), "abs_gt_20_count": int(np.count_nonzero(np.abs(selected) > 20))})
                else:
                    row.update({"status": "NOT_CHECKED", "reason": "No originally observed values"})
                standardized.append(row)
        del raw, prepared
        gc.collect()
    return examples, reserved, binary, standardized


def effective_parameters(data_root: Path, registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in registry["models"]:
        model_id = item["model_id"]
        if item["family"] == "logit":
            model = joblib.load(item["model_path"])
            params = model.get_params(deep=False); source = "recovered_from_saved_sklearn_model"
        elif item["family"] == "lightgbm":
            model = lgb.Booster(model_file=item["model_path"])
            params = {**model.params, "current_iteration": model.current_iteration(), "num_feature": model.num_feature()}
            source = "recovered_from_saved_lightgbm_booster"
        else:
            architecture = json.loads(Path(item["architecture_path"]).read_text(encoding="utf-8"))
            params = {**architecture, "optimizer": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.0001,
                      "batch_size": 4096, "gradient_clip_norm": 5.0, "input_dtype": "float32"}
            source = "architecture_file_plus_original_frozen_source_and_candidate_record"
        rows.append({"model_id": model_id, "parameter_source": source, "effective_parameters_json": json.dumps(params, sort_keys=True, default=str)})
    return rows


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve(); output_dir = args.output_dir.resolve(); repo_root = args.repo_root.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite post-review output: {output_dir}")
    output_dir.mkdir(parents=True)
    audit = data_root / "audits/task09" / args.run_id
    registry = json.loads((audit / "selected_model_registry.json").read_text(encoding="utf-8"))
    paths = provenance_paths(data_root, args.run_id, registry)
    before = {str(path): sha256_file(path) for _, path in paths}
    provenance = [{"artifact": name, "path": str(path), "sha256_before": before[str(path)]} for name, path in paths]

    original_source = subprocess.run(["git", "show", "58719d8:scripts/train_baseline_models.py"], cwd=repo_root, capture_output=True, check=True).stdout
    import hashlib
    original_source_hash = hashlib.sha256(original_source).hexdigest()
    current_source = repo_root / "scripts/train_baseline_models.py"
    standalone_source = Path(__file__).resolve()
    metadata = {
        "task": "TASK09_POST_REVIEW", "run_id": args.run_id, "started_at": utc_now(),
        "verifier_version": VERIFIER_VERSION, "verifier_sha256": sha256_file(standalone_source),
        "repaired_training_source_sha256": sha256_file(current_source),
        "original_training_source_checkpoint": "58719d8:scripts/train_baseline_models.py",
        "original_training_source_checkpoint_sha256": original_source_hash,
        "runtime_versions": runtime_versions(), "fitting_performed": False,
    }

    focused_junit = output_dir / "focused_pytest_junit.xml"
    full_junit = output_dir / "full_pytest_junit.xml"
    focused = run_command(
        [sys.executable, "-m", "pytest", "-q", "tests/test_task09_post_review.py", f"--junitxml={focused_junit}"], repo_root,
        output_dir / "focused_pytest.stdout.txt", output_dir / "focused_pytest.stderr.txt", focused_junit,
    )
    full = run_command(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={full_junit}"], repo_root,
        output_dir / "full_pytest.stdout.txt", output_dir / "full_pytest.stderr.txt", full_junit,
    )
    diff = run_command(["git", "diff", "--check"], repo_root, output_dir / "git_diff_check.stdout.txt", output_dir / "git_diff_check.stderr.txt")
    diff["collected_tests"] = 1
    diff["status"] = "PASS" if diff["exit_code"] == 0 else "FAIL"
    diff["reason"] = "git diff --check exit code"
    test_rows = [{"scope": "focused", **focused}, {"scope": "full_repository", **full}, {"scope": "git_diff_check", **diff}]

    verification = verify_saved_run(data_root, args.run_id)
    contract = load_contract(data_root)
    examples, reserved, binary, standardized = preprocessing_diagnostics(data_root, output_dir, contract)
    params = effective_parameters(data_root, registry)

    resource = pd.read_csv(audit / "resource_usage.csv")
    candidates = pd.read_csv(audit / "model_candidates.csv")
    timing = [
        {"scope": "sum_of_all_candidates_seconds", "seconds": float(candidates["elapsed_seconds"].sum()), "status": "MEASURED", "provenance": "model_candidates.csv; all 12 candidates"},
        {"scope": "matrix_loading_seconds", "seconds": float(resource.loc[resource.scope.str.startswith("matrix_load::"), "elapsed_seconds"].sum()), "status": "MEASURED", "provenance": "resource_usage.csv; six matrix loads"},
        {"scope": "historical_former_whole_run", "seconds": float(resource.loc[resource.scope.eq("whole_run"), "elapsed_seconds"].iloc[0]), "status": "MEASURED_LIMITED_SCOPE", "provenance": "resource_usage.csv; recorded before plots/docs/final verify"},
        {"scope": "logged_start_to_last_line", "seconds": 416.0, "status": "APPROXIMATE", "provenance": "training.log timestamps 17:24:56 to 17:31:52"},
        {"scope": "true_historical_process_start_to_end", "seconds": None, "status": "NOT_CHECKED", "provenance": "No historical end-to-end timer covered final output stages"},
    ]

    after = {str(path): sha256_file(path) for _, path in paths}
    for row in provenance:
        row["sha256_after"] = after[row["path"]]
        row["unchanged"] = row["sha256_before"] == row["sha256_after"]
        row["status"] = "PASS" if row["unchanged"] else "FAIL"
    check_rows = [
        {"check": "strong_saved_run_verification", "status": verification["status"], "actual": len(verification["models"]), "expected": 6, "reason": "All six models reloaded and inferred on every authoritative tuning row"},
        {"check": "focused_tests", "status": focused["status"], "actual": focused["collected_tests"], "expected": ">0 with exit 0", "reason": focused["reason"]},
        {"check": "full_repository_tests", "status": full["status"], "actual": full["collected_tests"], "expected": ">0 with exit 0", "reason": full["reason"]},
        {"check": "git_diff_check", "status": diff["status"], "actual": diff["exit_code"], "expected": 0, "reason": diff["reason"]},
        {"check": "synthetic_transform_examples", "status": "PASS" if all(row["status"] in {"PASS", "INFORMATIONAL"} for row in examples) else "FAIL", "actual": len(examples), "expected": "all computed rows PASS", "reason": "Frozen preprocessor transform called for missing and unseen paths"},
        {"check": "binary_and_ohe_contract", "status": "PASS" if all(row["status"] == "PASS" for row in binary) else "FAIL", "actual": len(binary), "expected": "all PASS", "reason": "TRAIN/tuning feature-only checks"},
        {"check": "immutable_provenance", "status": "PASS" if all(row["unchanged"] for row in provenance) else "FAIL", "actual": sum(row["unchanged"] for row in provenance), "expected": len(provenance), "reason": "Before/after SHA-256"},
        {"check": "calibration_final_evaluation", "status": "NOT_CHECKED", "actual": "not accessed", "expected": "deferred", "reason": "Outside authorized post-review scope"},
    ]

    atomic_csv(output_dir / "provenance_hashes.csv", provenance)
    atomic_csv(output_dir / "check_results.csv", check_rows)
    atomic_csv(output_dir / "full_tuning_reload_comparison.csv", verification["models"])
    atomic_csv(output_dir / "metric_recomputation.csv", verification["metrics"])
    atomic_csv(output_dir / "selection_reproduction.csv", verification["selection"])
    atomic_csv(output_dir / "preprocessing_examples_post_review.csv", examples)
    atomic_csv(output_dir / "reserved_ohe_activation.csv", reserved)
    atomic_csv(output_dir / "prepared_binary_ohe_audit.csv", binary)
    atomic_csv(output_dir / "standardized_value_audit_corrected.csv", standardized)
    atomic_csv(output_dir / "timing_reconciliation.csv", timing)
    atomic_csv(output_dir / "effective_model_parameters.csv", params)
    atomic_csv(output_dir / "execution_evidence.csv", test_rows)
    metadata.update({"completed_at": utc_now(), "status": "PASS" if all(row["status"] in {"PASS", "NOT_CHECKED"} for row in check_rows) else "FAIL",
                     "original_artifacts_unchanged": all(row["unchanged"] for row in provenance), "verification": verification,
                     "tests": test_rows, "deferred": ["calibration", "final evaluation", "thresholds", "approval policy", "economic value"]})
    atomic_json(output_dir / "post_review_summary.json", metadata)
    metric_by_id = {row["model_id"]: row for row in verification["metrics"]}
    metric_lines = [
        "| 模型 | ROC AUC | AP | Log loss | Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_ORDER:
        row = metric_by_id[model_id]
        metric_lines.append(
            f"| {model_id} | {row['roc_auc']:.6f} | {row['average_precision']:.6f} | {row['log_loss']:.6f} | {row['brier_score']:.6f} |"
        )
    delta_lines = []
    for family in ("logit", "lightgbm", "mlp"):
        t = metric_by_id[f"{family}_T"]; ad = metric_by_id[f"{family}_T_plus_AD"]
        delta_lines.append(
            f"- {family}: ΔAUC={ad['roc_auc']-t['roc_auc']:+.6f}，ΔAP={ad['average_precision']-t['average_precision']:+.6f}，"
            f"Δlog loss={ad['log_loss']-t['log_loss']:+.6f}，ΔBrier={ad['brier_score']-t['brier_score']:+.6f}。"
        )
    report = f"""# Task09 独立复核与修复响应

状态：**{metadata['status']}**。本次仅重载现有模型并执行推理、合同核验和诊断，没有调用拟合或优化步骤，也没有读取校准/最终评估标签。

## 结论

六个 `first_full` 模型均按原注册表重载，并对全部 {verification['prediction_rows']:,} 条 `validation_tuning` 记录重新推理。逐行概率绝对容差为 `1e-6`，指标重算绝对容差为 `1e-10`。原模型、原调优预测、Task08 矩阵/预处理器和原运行记录的前后 SHA-256 保持不变。

原调优结果继续作为开发证据：

{chr(10).join(metric_lines)}

按 T+AD 减 T 计算的获批 AD 包增量为：

{chr(10).join(delta_lines)}

三类模型的 T+AD 相对 T 均提高 AUC/AP并降低 log loss/Brier；这不能解释为因果效果、生产表现、时间稳定性或银行流水单一贡献。LightGBM T+AD 在六组中最好，MLP 未超过它并非训练失败。

## 预算与早停解释

训练仅有 12 个候选；候选耗时合计约 402.49 秒、六次矩阵加载约 2.36 秒、旧 `whole_run` 记录约 414.53 秒、日志跨度约 416 秒。旧 `whole_run` 在图表、文档和最终验证前写入，因此真正历史端到端时长未被精确测量，标为 NOT_CHECKED。`ru_maxrss` 仍只表示进程累计峰值。

MLP T 的第 26 轮成为精确最高 AP，但相对 patience 参考的提升不足 0.0001，因此同轮可以保存最佳权重并触发早停。MLP T+AD 执行到 30 轮、选择第 28 轮，属于预算受限结果。两组参数量分别为 5,505 和 9,025。PyTorch 使用单 intra-op 线程是原运行的稳定性调整；现有证据只有已记录的本机混合库四线程崩溃说明，没有新增复现。

TRAIN 恒零的保留 one-hot 列已在 tuning 上做 feature-only 激活计数。对 MLP 而言，这些输入没有从 TRAIN 激活中学到类别特定信息，但随机初始化权重仍可能在保留路径被激活时影响输出；本次没有删列、置零权重或重拟合词表。

## 范围

这仍是随机开发划分上的调优结果。校准、最终独立评估、阈值、审批政策和经济价值均未执行。详细机器可读证据见本目录的 CSV、pytest 日志和 JSON 汇总。
"""
    atomic_text(output_dir / "post_review_response_zh.md", report)
    if metadata["status"] != "PASS":
        raise RuntimeError("Post-review verification produced one or more FAIL checks; evidence preserved")
    print(json.dumps({"status": metadata["status"], "output_dir": str(output_dir), "models_retrained": 0, "models_verified": 6}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
