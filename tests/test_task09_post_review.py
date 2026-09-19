import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/home_credit_mpl_tests")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import pytest
import torch
import joblib
from types import SimpleNamespace
from sklearn.linear_model import LogisticRegression

from task09_reliability import (
    EarlyStoppingTracker,
    FamilyImplementationError,
    SharedContractError,
    VerificationError,
    completion_status_check,
    finalize_after_verification,
    run_isolated_combinations,
    test_execution_status as execution_evidence_status,
    validate_prediction_frame,
    validate_registry,
    weighted_epoch_mean,
)
from train_baseline_models import (
    TabularMLP, combination_is_reusable, load_authorized_labels, predict_mlp, sha256_file,
)


def prediction_fixture():
    authoritative = pd.DataFrame({"case_id": [10, 11, 12], "base_order": [0, 1, 2]})
    labels = np.array([0, 1, 0], dtype=np.int8)
    frame = authoritative.copy(); frame["target"] = labels
    frame["prob_logit_T"] = np.array([0.1, 0.8, 0.2], dtype=np.float64)
    return frame, authoritative, labels


@pytest.mark.parametrize("mutation", ["one_row", "wrong_id", "wrong_order", "wrong_target", "missing_probability", "nan", "inf", "out_of_range"])
def test_prediction_contract_rejects_malformed_values(mutation):
    frame, authoritative, labels = prediction_fixture()
    if mutation == "one_row": frame = frame.iloc[:1].copy()
    elif mutation == "wrong_id": frame.loc[0, "case_id"] = 999
    elif mutation == "wrong_order": frame = frame.iloc[[1, 0, 2]].reset_index(drop=True)
    elif mutation == "wrong_target": frame.loc[0, "target"] = 1
    elif mutation == "missing_probability": frame = frame.drop(columns="prob_logit_T")
    elif mutation == "nan": frame.loc[0, "prob_logit_T"] = np.nan
    elif mutation == "inf": frame.loc[0, "prob_logit_T"] = np.inf
    elif mutation == "out_of_range": frame.loc[0, "prob_logit_T"] = 1.1
    with pytest.raises((VerificationError, KeyError)):
        validate_prediction_frame(frame, authoritative, labels, ["logit_T"], {"logit_T": "prob_logit_T"})


def test_empty_registry_is_rejected():
    with pytest.raises(VerificationError):
        validate_registry({"models": []}, ["logit_T"])


def valid_status(tmp_path, family="logit"):
    artifacts = []
    for kind in ("model", "prediction_part", "reload_sample"):
        path = tmp_path / f"{kind}.bin"; path.write_bytes(kind.encode())
        artifacts.append({"kind": kind, "path": str(path), "sha256": sha256_file(path)})
    if family == "mlp":
        path = tmp_path / "architecture.json"; path.write_text("{}")
        artifacts.append({"kind": "architecture", "path": str(path), "sha256": sha256_file(path)})
    payload = {
        "status": "COMPLETE", "config_hash": "cfg", "comparison": {}, "candidates": [], "history": [],
        "registry": {"model_id": f"{family}_T", "family": family, "ordered_predictors": ["x"],
                     "representation": "linear_nn" if family != "lightgbm" else "gbdt", "preprocessor_sha256": "abc"},
        "artifacts": artifacts,
    }
    path = tmp_path / "status.json"; path.write_text(json.dumps(payload))
    return path, payload


@pytest.mark.parametrize("damage", ["empty_artifacts", "missing_model", "missing_reload", "bad_checksum", "wrong_id", "empty_features", "missing_architecture"])
def test_resume_rejects_incomplete_status_schema(tmp_path, damage):
    family = "mlp" if damage == "missing_architecture" else "logit"
    path, payload = valid_status(tmp_path, family)
    if damage == "empty_artifacts": payload["artifacts"] = []
    elif damage == "missing_model": payload["artifacts"] = [x for x in payload["artifacts"] if x["kind"] != "model"]
    elif damage == "missing_reload": payload["artifacts"] = [x for x in payload["artifacts"] if x["kind"] != "reload_sample"]
    elif damage == "bad_checksum": payload["artifacts"][0]["sha256"] = "bad"
    elif damage == "wrong_id": payload["registry"]["model_id"] = "wrong"
    elif damage == "empty_features": payload["registry"]["ordered_predictors"] = []
    elif damage == "missing_architecture": payload["artifacts"] = [x for x in payload["artifacts"] if x["kind"] != "architecture"]
    path.write_text(json.dumps(payload))
    complete, reason = completion_status_check(path, "cfg", f"{family}_T")
    assert not complete and reason


def test_resume_rejects_incompatible_ordered_features_and_checks_reload(tmp_path):
    # Populate the minimal fitted-state attributes directly: this fixture tests
    # serialization/reuse logic without calling fit or performing optimization.
    model = LogisticRegression(solver="lbfgs")
    model.classes_ = np.array([0, 1], dtype=np.int64)
    model.coef_ = np.array([[1.25]], dtype=np.float64)
    model.intercept_ = np.array([-1.0], dtype=np.float64)
    model.n_features_in_ = 1
    model.n_iter_ = np.array([0], dtype=np.int32)
    model_path = tmp_path / "model.joblib"; joblib.dump(model, model_path)
    tune_meta = pd.DataFrame({"case_id": np.arange(10, 74), "base_order": np.arange(64)})
    tuning_y = np.tile(np.array([0, 1], dtype=np.int8), 32)
    reload = tune_meta.copy(); reload["x"] = np.linspace(0, 3, 64)
    reload["expected_probability_before_serialization"] = model.predict_proba(reload[["x"]].to_numpy())[:, 1]
    reload_path = tmp_path / "reload.parquet"; reload.to_parquet(reload_path, index=False)
    part = tune_meta.copy(); part["target"] = tuning_y
    part["prob_logit_T"] = model.predict_proba(reload[["x"]].to_numpy())[:, 1].astype(np.float64)
    part_path = tmp_path / "part.parquet"; part.to_parquet(part_path, index=False)
    registry = {"model_id": "logit_T", "family": "logit", "ordered_predictors": ["x"],
                "representation": "linear_nn", "preprocessor_sha256": "abc", "selected_iteration_or_epoch": 1}
    payload = {"status": "COMPLETE", "config_hash": "cfg", "comparison": {}, "candidates": [], "history": [],
               "registry": registry, "artifacts": [
                   {"kind": "model", "path": str(model_path), "sha256": sha256_file(model_path)},
                   {"kind": "prediction_part", "path": str(part_path), "sha256": sha256_file(part_path)},
                   {"kind": "reload_sample", "path": str(reload_path), "sha256": sha256_file(reload_path)},
               ]}
    status = tmp_path / "status.json"; status.write_text(json.dumps(payload))
    contract = SimpleNamespace(feature_sets={"linear_nn": {"T": ["x"], "T_plus_AD": ["x"]},
                                              "gbdt": {"T": ["x"], "T_plus_AD": ["x"]}}, tuning_labels=tuning_y)
    assert combination_is_reusable(status, "cfg", "logit_T", contract, tune_meta)[0]
    payload["registry"]["ordered_predictors"] = ["wrong"]
    status.write_text(json.dumps(payload))
    reusable, reason = combination_is_reusable(status, "cfg", "logit_T", contract, tune_meta)
    assert not reusable and "ordered features" in reason


def test_role_filtered_target_loading_never_requests_heldout(monkeypatch, tmp_path):
    manifest = pd.DataFrame({"case_id": [10, 11], "base_order": [0, 1]})
    calls = []
    def fake_read(path, columns, filters):
        calls.append((columns, filters))
        assert filters in [[("outer_split", "==", "train")], [("validation_role", "==", "validation_tuning")]]
        assert "target" in columns
        if filters[0][0] == "outer_split":
            return pd.DataFrame({"case_id": [10], "base_order": [0], "target": [0], "outer_split": ["train"]})
        return pd.DataFrame({"case_id": [11], "base_order": [1], "target": [1], "validation_role": ["validation_tuning"]})
    monkeypatch.setattr(pd, "read_parquet", fake_read)
    train, tuning = load_authorized_labels(tmp_path / "manifest.parquet", manifest, np.array([0]), np.array([1]))
    assert train.tolist() == [0] and tuning.tolist() == [1] and len(calls) == 2


def test_real_sample_weighted_epoch_loss_handles_partial_final_batch():
    assert weighted_epoch_mean([1.0, 3.0], [4096, 3701]) == pytest.approx((4096 + 3 * 3701) / 7797)
    with pytest.raises(FloatingPointError):
        weighted_epoch_mean([1.0, np.inf], [4, 1])


def test_new_best_below_patience_delta_is_saved_and_can_stop():
    tracker = EarlyStoppingTracker(patience_limit=5, min_reset_delta=0.0001)
    tracker.update(epoch=1, average_precision=.5, log_loss=.2)
    for epoch, ap in enumerate([.50001, .50002, .50003, .50004], start=2):
        assert not tracker.update(epoch=epoch, average_precision=ap, log_loss=.2)["should_stop"]
    result = tracker.update(epoch=6, average_precision=.50009, log_loss=.19)
    assert result["is_best"] and result["best_epoch"] == 6 and result["should_stop"]


def test_nonfinite_mlp_parameter_and_logits_fail():
    model = TabularMLP(2)
    with torch.no_grad(): next(model.parameters()).view(-1)[0] = float("nan")
    with pytest.raises(FloatingPointError, match="parameter"):
        predict_mlp(model, np.ones((2, 2), dtype=np.float32))
    model = TabularMLP(2)
    def bad_forward(_): return torch.tensor([[float("inf")], [0.0]])
    model.forward = bad_forward
    with pytest.raises(FloatingPointError, match="logits"):
        predict_mlp(model, np.ones((2, 2), dtype=np.float32))


def test_nonfinite_gradient_is_rejected_by_actual_clip_implementation():
    model = torch.nn.Linear(1, 1)
    model.weight.grad = torch.tensor([[float("inf")]])
    with pytest.raises(RuntimeError):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)


def test_verification_failure_cannot_write_complete():
    states = []
    with pytest.raises(RuntimeError, match="boom"):
        finalize_after_verification(lambda: states.append("PENDING"), lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                                    lambda _: states.append("COMPLETE"), lambda _: states.append("VERIFY_FAILED"))
    assert states == ["PENDING", "VERIFY_FAILED"]


def test_combination_local_failure_continues_and_family_failure_blocks_only_family():
    calls = []
    def run_one(name):
        calls.append(name)
        if name == "logit_T": raise RuntimeError("local")
        if name == "mlp_T": raise FamilyImplementationError("family")
        return name
    failures = []
    combinations = ["logit_T", "lightgbm_T", "mlp_T", "mlp_T_plus_AD", "lightgbm_T_plus_AD"]
    result = run_isolated_combinations(combinations, lambda name: name.split("_", 1)[0], run_one,
                                       lambda *row: failures.append(row))
    assert result["logit_T"]["status"] == "FAILED_LOCAL"
    assert result["lightgbm_T"]["status"] == "COMPLETE"
    assert result["mlp_T_plus_AD"]["status"] == "BLOCKED_BY_FAMILY_FAILURE"
    assert result["lightgbm_T_plus_AD"]["status"] == "COMPLETE"


def test_shared_contract_failure_stops_dependent_work():
    calls = []
    def run_one(name):
        calls.append(name)
        raise SharedContractError("shared")
    with pytest.raises(SharedContractError):
        run_isolated_combinations(["logit_T", "lightgbm_T"], lambda x: x.split("_", 1)[0], run_one, lambda *x: None)
    assert calls == ["logit_T"]


def test_interruption_is_recorded_then_propagated():
    failures = []
    def interrupt(_): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_isolated_combinations(["logit_T"], lambda _: "logit", interrupt, lambda *row: failures.append(row))
    assert failures and failures[0][1] == "INTERRUPTED"


def test_free_text_cannot_create_test_pass():
    status, _ = execution_evidence_status({"result": "86 passed"})
    assert status == "FAIL"
    status, _ = execution_evidence_status({})
    assert status == "NOT_CHECKED"
