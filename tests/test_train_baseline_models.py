import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/home_credit_mpl_tests")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from train_baseline_models import (
    MODEL_ORDER,
    TabularMLP,
    atomic_json,
    class_one_probability,
    combination_is_complete,
    feature_contract_for,
    load_matrix_rows,
    official_metrics,
    parse_models,
    predict_mlp,
    select_candidate,
    sha256_file,
    torch_load_state,
)


def candidate(candidate_id, ap, loss, **kwargs):
    return {"candidate_id": candidate_id, "status": "SUCCESS", "converged": True,
            "average_precision": ap, "log_loss": loss, **kwargs}


def test_official_metrics_uses_sklearn_average_precision_and_natural_counts():
    y = np.array([0, 1, 0, 1], dtype=np.int8)
    p = np.array([0.2, 0.2, 0.8, 0.8])
    result = official_metrics(y, p)
    assert result["average_precision"] == average_precision_score(y, p)
    assert result["N"] == 4 and result["positive_count"] == 2


def test_official_metrics_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="inside"):
        official_metrics(np.array([0, 1]), np.array([0.2, 1.1]))
    with pytest.raises(ValueError, match="finite"):
        official_metrics(np.array([0, 1]), np.array([0.2, np.nan]))


def test_global_ap_band_is_not_pairwise_streaming():
    # A is within 0.0001 of B, B within 0.0001 of C, but A is outside C's global band.
    rows = [
        candidate("A", 0.50000, 0.10, C=0.01),
        candidate("B", 0.50005, 0.20, C=0.1),
        candidate("C", 0.50011, 0.30, C=1.0),
    ]
    winner, annotated = select_candidate(rows, "logit")
    assert winner["candidate_id"] == "B"
    by_id = {x["candidate_id"]: x for x in annotated}
    assert not by_id["A"]["ap_band_eligible"]
    assert by_id["B"]["ap_band_eligible"] and by_id["C"]["ap_band_eligible"]


def test_candidate_ties_choose_lower_declared_complexity():
    winner, _ = select_candidate([
        candidate("large", .5, .2, C=1.0), candidate("small", .5, .2, C=.01)
    ], "logit")
    assert winner["candidate_id"] == "small"
    winner, _ = select_candidate([
        candidate("moderate", .5, .2, num_leaves=31), candidate("compact", .5, .2, num_leaves=15)
    ], "lightgbm")
    assert winner["candidate_id"] == "compact"


def test_failed_and_unconverged_candidates_are_ineligible():
    rows = [candidate("ok", .4, .3, C=.1), candidate("bad", .9, .1, C=.01)]
    rows[1]["converged"] = False
    winner, _ = select_candidate(rows, "logit")
    assert winner["candidate_id"] == "ok"


def test_model_argument_has_no_sampling_mode():
    assert parse_models("all") == MODEL_ORDER
    assert parse_models("mlp_T,logit_T") == ["logit_T", "mlp_T"]
    with pytest.raises(ValueError):
        parse_models("smoke")


def test_feature_contract_selects_exact_saved_lists_and_prefix():
    fs = {
        "linear_nn": {"T": ["a", "b"], "T_plus_AD": ["a", "b", "c"]},
        "gbdt": {"T": ["x", "y"], "T_plus_AD": ["x", "y", "z"]},
    }
    assert feature_contract_for("logit_T", fs) == ("logit", "linear_nn", ["a", "b"])
    assert feature_contract_for("lightgbm_T_plus_AD", fs) == ("lightgbm", "gbdt", ["x", "y", "z"])


def test_matrix_loader_preserves_requested_order_and_exact_keys(tmp_path):
    frame = pd.DataFrame({"case_id": [10, 11, 12], "b": [1., 2., 3.], "a": [4., 5., 6.]})
    path = tmp_path / "matrix.parquet"; frame.to_parquet(path, index=False)
    matrix, info = load_matrix_rows(path, ["a", "b"], np.array([0, 2]), np.array([10, 11, 12]), np.float64, batch_size=2)
    np.testing.assert_array_equal(matrix, np.array([[4., 1.], [6., 3.]]))
    assert info["shape"] == [2, 2]


def test_matrix_loader_rejects_shuffled_key_alignment(tmp_path):
    frame = pd.DataFrame({"case_id": [10, 12, 11], "a": [1., 2., 3.]})
    path = tmp_path / "matrix.parquet"; frame.to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="key order"):
        load_matrix_rows(path, ["a"], np.array([0, 1]), np.array([10, 11, 12]), np.float64, batch_size=2)


def test_actual_logit_serialization_and_class_order(tmp_path):
    rng = np.random.default_rng(1); x = rng.normal(size=(100, 3)); y = (x[:, 0] > 0).astype(np.int8)
    model = LogisticRegression(C=.1, l1_ratio=0., solver="lbfgs", max_iter=100).fit(x, y)
    before = class_one_probability(model, x[:10])
    path = tmp_path / "model.joblib"; joblib.dump(model, path)
    after = class_one_probability(joblib.load(path), x[:10])
    np.testing.assert_allclose(before, after, atol=0, rtol=0)


def test_lightgbm_modern_api_metric_order_best_iteration_and_serialization(tmp_path):
    rng = np.random.default_rng(2); x = rng.normal(size=(150, 4)); x[0, 0] = np.nan; x[1, 0] = 0
    y = (np.nan_to_num(x[:, 0]) + x[:, 1] > 0).astype(np.int8)
    record = {}
    model = lgb.LGBMClassifier(
        n_estimators=30, learning_rate=.1, num_leaves=7, min_child_samples=3,
        metric="None", zero_as_missing=False, use_missing=True, verbosity=-1,
        deterministic=True, force_col_wise=True, n_jobs=1,
    )
    model.fit(x[:100], y[:100], eval_X=x[100:], eval_y=y[100:], eval_names=["validation_tuning"],
              eval_metric=["average_precision", "auc", "binary_logloss"],
              callbacks=[lgb.record_evaluation(record), lgb.early_stopping(5, first_metric_only=True, verbose=False)])
    assert list(record["validation_tuning"])[:3] == ["average_precision", "auc", "binary_logloss"]
    assert model.best_iteration_ > 0 and model.get_params()["zero_as_missing"] is False
    before = model.predict_proba(x[100:], num_iteration=model.best_iteration_)[:, 1]
    assert abs(record["validation_tuning"]["average_precision"][model.best_iteration_ - 1] - average_precision_score(y[100:], before)) <= 1e-10
    path = tmp_path / "model.txt"; model.booster_.save_model(path, num_iteration=model.best_iteration_)
    after = lgb.Booster(model_file=str(path)).predict(x[100:], num_iteration=model.best_iteration_)
    np.testing.assert_allclose(before, after, atol=1e-12, rtol=0)


def test_tied_score_lightgbm_ap_matches_sklearn():
    x = np.array([[0.], [0.], [1.], [1.], [2.], [2.], [3.], [3.]])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    record = {}
    model = lgb.LGBMClassifier(n_estimators=1, num_leaves=2, min_child_samples=1,
                               metric="None", verbosity=-1, deterministic=True, force_col_wise=True, n_jobs=1)
    model.fit(x, y, eval_X=x, eval_y=y, eval_names=["validation_tuning"], eval_metric=["average_precision"], callbacks=[lgb.record_evaluation(record)])
    p = model.predict_proba(x)[:, 1]
    assert np.unique(p).size == 1
    assert record["validation_tuning"]["average_precision"][0] == average_precision_score(y, p)


def test_mlp_bce_shapes_eval_dropout_and_state_dict_reload(tmp_path):
    torch.manual_seed(3); x = np.random.default_rng(3).normal(size=(20, 5)).astype(np.float32)
    y = torch.from_numpy((x[:, :1] > 0).astype(np.float32))
    model = TabularMLP(5, output_bias=-2.0); model.train()
    logits = model(torch.from_numpy(x)); assert logits.shape == y.shape
    loss = torch.nn.BCEWithLogitsLoss()(logits, y); loss.backward()
    frozen = {k: v.detach().clone() for k, v in model.state_dict().items()}
    path = tmp_path / "state.pt"; torch.save(frozen, path)
    model.eval(); first = predict_mlp(model, x)
    second = predict_mlp(model, x); np.testing.assert_array_equal(first, second)
    loaded = TabularMLP(5); loaded.load_state_dict(torch_load_state(path)); loaded.eval()
    np.testing.assert_allclose(first, predict_mlp(loaded, x), atol=0, rtol=0)


def test_frozen_best_weights_do_not_mutate_after_optimizer_step():
    torch.manual_seed(4); model = TabularMLP(2); optimizer = torch.optim.AdamW(model.parameters())
    frozen = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loss = model(torch.ones(3, 2)).sum(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True); optimizer.step()
    assert any(not torch.equal(frozen[k], model.state_dict()[k]) for k in frozen)
    assert all(torch.isfinite(v).all() for v in frozen.values())


def test_sample_weighted_epoch_loss_differs_from_unweighted_batch_mean():
    losses, sizes = np.array([1.0, 3.0]), np.array([4, 1])
    weighted = float(np.sum(losses * sizes) / sizes.sum())
    assert weighted == 1.4 and weighted != losses.mean()


def test_resume_requires_matching_hashes_and_uncorrupted_artifacts(tmp_path):
    model = tmp_path / "model.bin"; model.write_bytes(b"model")
    prediction = tmp_path / "prediction.parquet"; prediction.write_bytes(b"prediction")
    reload_sample = tmp_path / "reload.parquet"; reload_sample.write_bytes(b"reload")
    status = tmp_path / "status.json"
    atomic_json(status, {
        "status": "COMPLETE", "config_hash": "abc", "comparison": {}, "candidates": [], "history": [],
        "registry": {"model_id": "logit_T", "family": "logit", "ordered_predictors": ["x"],
                     "representation": "linear_nn", "preprocessor_sha256": "frozen"},
        "artifacts": [
            {"kind": "model", "path": str(model), "sha256": sha256_file(model)},
            {"kind": "prediction_part", "path": str(prediction), "sha256": sha256_file(prediction)},
            {"kind": "reload_sample", "path": str(reload_sample), "sha256": sha256_file(reload_sample)},
        ],
    })
    assert combination_is_complete(status, "abc", "logit_T")
    assert not combination_is_complete(status, "different")
    model.write_bytes(b"corrupt")
    assert not combination_is_complete(status, "abc")
