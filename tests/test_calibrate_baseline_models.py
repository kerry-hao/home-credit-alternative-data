import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import pytest

import calibrate_baseline_models as orchestrator
from task10_calibration import CalibrationContractError, sha256_file


def test_model_parser_fixed_order():
    assert orchestrator.parse_models("mlp_T,logit_T") == ["logit_T", "mlp_T"]
    assert orchestrator.parse_models("all") == orchestrator.MODEL_ORDER
    with pytest.raises(ValueError): orchestrator.parse_models("unknown")


def test_private_output_cannot_target_repository(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    with pytest.raises(CalibrationContractError): orchestrator.ensure_external(repo / "data", repo)
    orchestrator.ensure_external(tmp_path / "external", repo)


def test_membership_target_read_is_role_filtered(monkeypatch, tmp_path):
    metadata = pd.DataFrame({"case_id": np.arange(6), "base_order": np.arange(6),
                             "outer_split": ["train", "train", "validation", "validation", "evaluation", "evaluation"],
                             "validation_role": [None, None, "validation_tuning", "validation_calibration", None, None]})
    labels = pd.DataFrame({"case_id": [3], "base_order": [3], "target": [1], "validation_role": ["validation_calibration"]})
    calls = []
    def fake_read(path, columns, filters=None):
        calls.append((tuple(columns), filters))
        if "target" not in columns:
            return metadata
        assert filters == [("validation_role", "==", "validation_calibration")]
        return labels
    monkeypatch.setattr(pd, "read_parquet", fake_read)
    monkeypatch.setattr(orchestrator, "EXPECTED_TOTAL", 6)
    monkeypatch.setattr(orchestrator, "EXPECTED_CALIBRATION", 1)
    monkeypatch.setattr(orchestrator, "EXPECTED_POSITIVES", 1)
    monkeypatch.setattr(orchestrator, "EXPECTED_MEMBERSHIP_FP", orchestrator.sha256_lines(f"{r.case_id}\t{r.outer_split}\t{r.validation_role}" for r in metadata.itertuples()))
    monkeypatch.setattr(orchestrator, "EXPECTED_TRAIN_FP", orchestrator.sha256_lines([0, 1]))
    result, positions, *_ = orchestrator.load_membership(tmp_path)
    assert len(result) == 1 and positions.tolist() == [3]
    assert all("target" not in columns or filters == [("validation_role", "==", "validation_calibration")] for columns, filters in calls)


def test_verify_only_dispatch_never_calls_formal_run(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["prog", "--data-root", str(tmp_path), "--verify-only"])
    monkeypatch.setattr(orchestrator, "verify_saved_run", lambda *args: {"status": "PASS"})
    monkeypatch.setattr(orchestrator, "formal_run", lambda *args: (_ for _ in ()).throw(AssertionError("formal run called")))
    assert orchestrator.main() == 0


def test_identity_calibrator_round_trip(tmp_path):
    path = tmp_path / "calibrator.json"; meta = tmp_path / "meta.json"
    digest, _ = orchestrator.save_calibrator("identity", None, path, meta, {"role": "test"})
    assert digest == sha256_file(path) and orchestrator.load_calibrator("identity", path) is None


def test_identity_calibrator_rejects_modified_json(tmp_path):
    path = tmp_path / "calibrator.json"; path.write_text(json.dumps({"method": "identity", "mapping": "wrong"}))
    with pytest.raises(CalibrationContractError): orchestrator.load_calibrator("identity", path)


def test_required_application_schemas_are_exact():
    raw = ["case_id", "base_order", "target", *[f"raw__{model}" for model in orchestrator.MODEL_ORDER]]
    fold = ["case_id", "base_order", "target", "fold_id"]
    oof = ["case_id", "base_order", "target", "fold_id", *[f"oof__{model}__{method}" for model in orchestrator.MODEL_ORDER for method in ("identity", "logistic", "isotonic")]]
    full = ["case_id", "base_order", "target", "model_id", "selected_method", "raw_probability", "full_fit_calibrated_probability", "estimate_role"]
    assert raw[:3] == fold[:3] == oof[:3]
    assert len(raw) == 9 and len(fold) == 4 and len(oof) == 22 and len(full) == 8


def test_estimate_roles_are_separate_and_fixed():
    assert len({orchestrator.RAW_ROLE, orchestrator.OOF_ROLE, orchestrator.FULL_FIT_ROLE}) == 3
    assert orchestrator.FULL_FIT_ROLE == "IN_SAMPLE_ARTIFACT_DIAGNOSTIC_NOT_PERFORMANCE_ESTIMATE"


def test_formal_constants_match_specification():
    assert orchestrator.CROSSFIT_FOLDS == 5 and orchestrator.CROSSFIT_SEED == 20260921
    assert orchestrator.BOOTSTRAP_REPLICATES == 2000 and orchestrator.BOOTSTRAP_SEED == 20260922
    assert orchestrator.ISOTONIC_MAX_DROP == .0005 and orchestrator.LOGISTIC_RANK_TOLERANCE == 1e-10
    assert orchestrator.NUMERIC_TIE_TOLERANCE == 1e-12 and orchestrator.BOOTSTRAP_DDOF == 1
    assert orchestrator.VERSION == "task10_probability_calibration_v2"
    assert orchestrator.SELECTION_CONTRACT == "hard_guardrails_then_one_se_v2"


def test_task09_hash_change_fails_preflight(monkeypatch, tmp_path):
    prep = tmp_path / "interim/task08_followup"; prep.mkdir(parents=True)
    for relative in orchestrator.TASK08_HASHES:
        path = prep / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"changed")
    with pytest.raises(CalibrationContractError, match="Task 08 hash mismatch"):
        orchestrator.preflight(tmp_path, "first_full", orchestrator.MODEL_ORDER)


def test_threads_are_capped_at_four():
    config = orchestrator.configure_threads(99)
    assert config["effective_threads"] <= 4 and config["torch_intraop"] == 1


def test_protected_hash_verifier_rejects_altered_hash():
    before = pd.DataFrame([{"path": "/x", "sha256": "a"}])
    after = pd.DataFrame([{"path": "/x", "sha256": "b"}])
    with pytest.raises(CalibrationContractError, match="before/after"):
        orchestrator.verify_protected_hash_tables(before, after, check_current_files=False)


def test_selected_calibrator_copy_must_be_byte_identical(tmp_path):
    source = tmp_path / "source.joblib"; selected = tmp_path / "selected.joblib"
    source.write_bytes(b"source"); selected.write_bytes(b"different")
    item = {"model_id": "m", "selected_calibrator_path": str(selected),
            "selected_calibrator_source_path": str(source),
            "selected_calibrator_sha256": sha256_file(selected),
            "selected_calibrator_source_sha256": sha256_file(source)}
    with pytest.raises(CalibrationContractError, match="byte-identical"):
        orchestrator.verify_selected_calibrator_copy(item)


def test_post_preflight_failure_writes_after_hashes_and_failed_status(monkeypatch, tmp_path):
    data_root = tmp_path / "data"; protected = tmp_path / "protected.bin"; protected.write_bytes(b"frozen")
    registry_path = tmp_path / "registry.json"; registry_path.write_text("{}")
    config_path = tmp_path / "config.json"; config_path.write_text("{}")
    monkeypatch.setattr(orchestrator, "preflight", lambda *args: {
        "registry_path": registry_path, "run_config_path": config_path, "models": {}, "feature_sets": {}})
    membership = pd.DataFrame({"case_id": [1, 2], "base_order": [0, 1], "target": [0, 1]})
    monkeypatch.setattr(orchestrator, "load_membership", lambda *args: (membership, np.array([0, 1]), "membership", "train"))
    monkeypatch.setattr(orchestrator, "protected_paths", lambda *args: [("protected", protected)])
    monkeypatch.setattr(orchestrator, "make_common_folds", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("post-preflight boom")))
    args = SimpleNamespace(data_root=data_root, base_run_id="first_full", run_id="failure_safe", models="all", threads=1, resume=False)
    repo_root = Path(__file__).resolve().parents[1]
    with pytest.raises(RuntimeError, match="post-preflight boom"):
        orchestrator.formal_run(args, repo_root)
    audit = data_root / "audits/task10/failure_safe"
    status = json.loads((audit / "task_status.json").read_text())
    assert status["status"] == "FAILED" and status["stage"] == "common_fold_assignment"
    assert status["protected_input_integrity"]["status"] == "PASS"
    assert (audit / "input_artifact_hashes_after.csv").is_file()
    assert not (audit / "calibration_registry.json").exists()
