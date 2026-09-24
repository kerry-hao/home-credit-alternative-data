import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

import task10_calibration as calibration
from task10_calibration import (
    CalibrationContractError,
    RawIdentityPredictionError,
    apply_candidate,
    annotate_bootstrap_references,
    atomic_json,
    bin_diagnostics,
    bootstrap_draws,
    brier_guardrail_pass,
    calibration_in_the_large,
    calibration_joint_intercept_slope,
    choose_numeric_best,
    completion_check,
    endpoint_protected_logit,
    fit_isotonic_calibrator,
    fit_logistic_calibrator,
    fold_summary,
    isotonic_ranking_guardrail,
    json_safe,
    log_loss_one_se_pass,
    logistic_ranking_guardrail,
    make_common_folds,
    mechanical_selection,
    official_metrics,
    paired_standard_error,
    paired_stratified_bootstrap,
    predict_isotonic_calibrator,
    predict_logistic_calibrator,
    sha256_file,
    stability_warning,
    tie_preserving_bins,
    top_risk_summary,
    validate_oof,
    validate_probabilities,
    validate_raw_identity_predictions,
    verify_v2_selection_table,
)


def fixture_labels():
    return np.array([0] * 100 + [1] * 20, dtype=np.int8)


def fixture_probabilities(y=None):
    labels = fixture_labels() if y is None else np.asarray(y)
    rng = np.random.default_rng(10)
    return np.clip(0.05 + 0.35 * labels + rng.normal(0, 0.04, len(labels)), 0.001, 0.999).astype(np.float64)


def selection_row(method, loss=None, se=0.01, brier=True, ranking=True, valid=True):
    default_loss = {"identity": 0.101, "logistic": 0.100, "isotonic": 0.099}[method]
    return {"method": method, "oof_log_loss": default_loss if loss is None else loss,
            "fit_and_probability_valid": valid, "paired_se_log_loss_vs_admissible_numeric_best": se,
            "passes_brier_guardrail": brier, "passes_ranking_guardrail": ranking}


def test_fold_seed_reproducibility():
    y = fixture_labels()
    np.testing.assert_array_equal(make_common_folds(y), make_common_folds(y))


def test_folds_cover_once_and_are_disjoint():
    folds = make_common_folds(fixture_labels())
    assert len(folds) == 120 and set(folds) == set(range(5)) and np.all(folds >= 0)


def test_fold_exact_stratified_counts_controlled_fixture():
    summary = fold_summary(make_common_folds(fixture_labels()), fixture_labels())
    assert (summary.rows == 24).all() and (summary.positive_count == 4).all() and (summary.negative_count == 20).all()


def test_one_fold_assignment_can_be_reused_for_all_models():
    folds = make_common_folds(fixture_labels())
    assignments = {model: folds for model in ["a", "b", "c"]}
    assert all(np.shares_memory(folds, item) or np.array_equal(folds, item) for item in assignments.values())


def test_oof_alignment_failure():
    p = fixture_probabilities()
    with pytest.raises(CalibrationContractError): validate_oof(p, np.r_[np.ones(119), 0], 120)


def test_probability_validation_rejects_wrong_dtype_order_and_values():
    with pytest.raises(CalibrationContractError): validate_probabilities(np.array([.1], dtype=np.float32))
    with pytest.raises(CalibrationContractError): validate_probabilities(np.array([[.1]], dtype=np.float64))
    with pytest.raises(CalibrationContractError): validate_probabilities(np.array([np.nan], dtype=np.float64))
    with pytest.raises(CalibrationContractError): validate_probabilities(np.array([1.1], dtype=np.float64))


def test_identity_is_exact_elementwise_copy():
    p = fixture_probabilities(); result = apply_candidate("identity", None, p)
    np.testing.assert_array_equal(result, p); assert result is not p


def test_logistic_uses_logit_of_raw_probability():
    y = fixture_labels(); p = fixture_probabilities(y)
    model, _ = fit_logistic_calibrator(p, y); z, _ = endpoint_protected_logit(p)
    expected = model.predict_proba(z.reshape(-1, 1))[:, 1]
    np.testing.assert_allclose(predict_logistic_calibrator(model, p), expected, atol=0, rtol=0)


def test_positive_slope_single_logistic_preserves_auc_and_ap():
    y = np.array([0, 1] * 50, dtype=np.int8)
    p = np.linspace(.02, .62, len(y), dtype=np.float64)
    model, meta = fit_logistic_calibrator(p, y)
    mapped = predict_logistic_calibrator(model, p)
    assert meta["beta"] > 0
    assert abs(roc_auc_score(y, p) - roc_auc_score(y, mapped)) <= 1e-15
    assert abs(average_precision_score(y, p) - average_precision_score(y, mapped)) <= 1e-15


def test_endpoint_protection_is_deterministic_without_raw_overwrite():
    p = np.array([0.0, 0.2, 1.0], dtype=np.float64); before = p.copy()
    z1, audit1 = endpoint_protected_logit(p); z2, audit2 = endpoint_protected_logit(p)
    np.testing.assert_array_equal(p, before); np.testing.assert_array_equal(z1, z2)
    assert audit1 == audit2 and audit1["protected_endpoint_count"] == 2 and np.isfinite(z1).all()


def test_logistic_beta_nonpositive_is_invalid(monkeypatch):
    class Fake:
        max_iter = 10000
        def fit(self, x, y):
            self.intercept_ = np.array([0.]); self.coef_ = np.array([[0.]]); self.n_iter_ = np.array([1]); return self
    monkeypatch.setattr(calibration, "logistic_estimator", lambda: Fake())
    with pytest.raises(CalibrationContractError, match="Invalid Logistic"):
        fit_logistic_calibrator(np.array([.1, .2, .8, .9]), np.array([0, 0, 1, 1]))


def test_logistic_nonfinite_parameter_is_invalid(monkeypatch):
    class Fake:
        max_iter = 10000
        def fit(self, x, y):
            self.intercept_ = np.array([np.nan]); self.coef_ = np.array([[1.]]); self.n_iter_ = np.array([1]); return self
    monkeypatch.setattr(calibration, "logistic_estimator", lambda: Fake())
    with pytest.raises(CalibrationContractError): fit_logistic_calibrator(np.array([.1, .2, .8, .9]), np.array([0, 0, 1, 1]))


def test_logistic_iteration_budget_exhaustion_is_invalid(monkeypatch):
    class Fake:
        max_iter = 2
        def fit(self, x, y):
            self.intercept_ = np.array([0.]); self.coef_ = np.array([[1.]]); self.n_iter_ = np.array([2]); return self
    monkeypatch.setattr(calibration, "logistic_estimator", lambda: Fake())
    with pytest.raises(CalibrationContractError): fit_logistic_calibrator(np.array([.1, .2, .8, .9]), np.array([0, 0, 1, 1]))


def test_isotonic_monotonic_bounds_and_clipping():
    y = fixture_labels(); p = fixture_probabilities(y); model, meta = fit_isotonic_calibrator(p, y)
    grid = np.linspace(-1, 2, 100); mapped = predict_isotonic_calibrator(model, grid)
    assert meta["nondecreasing"] and meta["out_of_range_clipping_ok"]
    assert np.all(np.diff(mapped) >= 0) and np.all((mapped >= 0) & (mapped <= 1))


def test_isotonic_does_not_jitter_equal_inputs():
    p = np.array([.1, .1, .2, .2, .8, .8, .9, .9]); y = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    model, _ = fit_isotonic_calibrator(p, y); mapped = predict_isotonic_calibrator(model, p)
    assert mapped[0] == mapped[1] and mapped[2] == mapped[3] and mapped[4] == mapped[5]


def test_each_oof_row_has_one_excluding_fold():
    folds = make_common_folds(fixture_labels()); assigned = np.zeros(len(folds), dtype=int)
    for fold in range(5):
        heldout = folds == fold; training = folds != fold
        assert not np.any(heldout & training); assigned[heldout] += 1
    assert np.all(assigned == 1)


def test_bootstrap_seed_reproducibility_small():
    y = np.array([0, 0, 0, 1, 1]); p = np.array([.1, .2, .3, .7, .9])
    probs = {("m", "identity"): p, ("m", "logistic"): p ** .9}
    a, ma = paired_stratified_bootstrap(y, probs, {"m": "identity"}, replicates=8, seed=20260922, batch_size=3)
    b, mb = paired_stratified_bootstrap(y, probs, {"m": "identity"}, replicates=8, seed=20260922, batch_size=4)
    pd.testing.assert_frame_equal(a, b); assert ma["seed"] == mb["seed"]


def test_bootstrap_draw_class_sizes_and_replacement():
    y = np.array([0, 0, 0, 1, 1]); _, draw = next(iter(bootstrap_draws(y, replicates=1)))
    assert len(draw) == 5 and np.sum(y[draw] == 1) == 2 and np.sum(y[draw] == 0) == 3
    assert len(set(draw)) <= len(draw)


def test_bootstrap_shared_indices_give_zero_difference_for_equal_methods():
    y = np.array([0, 0, 0, 1, 1]); p = np.array([.1, .2, .3, .7, .9])
    frame, _ = paired_stratified_bootstrap(y, {("m", "identity"): p, ("m", "logistic"): p.copy()}, {"m": "identity"}, replicates=10)
    assert (frame.log_loss_difference_vs_admissible_numeric_best == 0).all()
    assert (frame.brier_difference_vs_identity == 0).all()


def test_paired_standard_error_uses_ddof_one():
    values = np.array([1., 2., 4.])
    assert paired_standard_error(values) == np.std(values, ddof=1)


def test_numeric_tie_tolerance_selects_simpler_reference():
    assert choose_numeric_best({"identity": .1 + 1e-12, "logistic": .1, "isotonic": .2}) == "identity"


def test_one_se_boolean_boundary():
    assert log_loss_one_se_pass(.01 + 1e-12, .01)
    assert not log_loss_one_se_pass(np.nextafter(.01 + 1e-12, np.inf), .01)


def test_identity_selected_when_eligible():
    selected, rows = mechanical_selection([selection_row("identity"), selection_row("logistic"), selection_row("isotonic")])
    assert selected == "identity" and sum(row["selected"] for row in rows) == 1


def test_logistic_selected_when_identity_outside_and_guardrails_pass():
    selected, _ = mechanical_selection([selection_row("identity", loss=.12, se=.001), selection_row("logistic", loss=.10), selection_row("isotonic", loss=.11)])
    assert selected == "logistic"


def test_isotonic_selected_when_simpler_methods_outside():
    selected, _ = mechanical_selection([selection_row("identity", loss=.12, se=.001), selection_row("logistic", brier=False), selection_row("isotonic", loss=.10)])
    assert selected == "isotonic"


def test_brier_deterioration_above_one_se_excludes():
    assert brier_guardrail_pass(.01, .01) and not brier_guardrail_pass(.0100001, .01)


@pytest.mark.parametrize("metric", ["auc", "ap"])
def test_isotonic_drop_exact_boundary_passes_and_larger_fails(metric):
    args = [0.8, 0.2, 0.8, 0.2]
    if metric == "auc": args[2] = 0.7995
    else: args[3] = 0.1995
    assert isotonic_ranking_guardrail(*args)[0]
    if metric == "auc": args[2] = np.nextafter(0.7995, -np.inf)
    else: args[3] = np.nextafter(0.1995, -np.inf)
    assert not isotonic_ranking_guardrail(*args)[0]


def test_logistic_ranking_tolerance_boundary():
    assert logistic_ranking_guardrail(.8, .2, .8 + 1e-10, .2 - 1e-10, 1.0)[0]
    assert not logistic_ranking_guardrail(.8, .2, .8 + 1.00001e-10, .2, 1.0)[0]


def test_stability_warning_does_not_select():
    rows = []
    for fold in range(5):
        rows += [{"method": "identity", "fold_id": fold, "log_loss": .1},
                 {"method": "logistic", "fold_id": fold, "log_loss": .09 if fold == 0 else .11}]
    assert stability_warning(pd.DataFrame(rows), "logistic") == "STABILITY_WARNING"


def test_invalid_identity_fails_not_fallback():
    with pytest.raises(RawIdentityPredictionError, match="INVALID_RAW_IDENTITY"):
        mechanical_selection([selection_row("identity", valid=False), selection_row("logistic")])


def test_official_metrics_match_sklearn_hand_example():
    y = np.array([0, 1, 0, 1]); p = np.array([.1, .7, .4, .8], dtype=np.float64)
    result = official_metrics(y, p)
    assert result["log_loss"] == log_loss(y, p, labels=[0, 1])
    assert result["brier_score"] == brier_score_loss(y, p)
    assert result["roc_auc"] == roc_auc_score(y, p)
    assert result["average_precision"] == average_precision_score(y, p)


def test_calibration_intercept_is_zero_for_mean_matched_constant():
    y = np.array([0, 0, 1, 1]); p = np.full(4, .5)
    assert abs(calibration_in_the_large(y, p)) <= 1e-10


def test_joint_intercept_slope_finite():
    y = fixture_labels(); p = fixture_probabilities(y)
    intercept, slope = calibration_joint_intercept_slope(y, p)
    assert np.isfinite(intercept) and np.isfinite(slope)


def test_tie_preserving_bins_never_split_equal_values():
    y = np.array([0, 1, 0, 1, 0, 1]); p = np.array([.1, .1, .1, .9, .9, .9])
    bins = tie_preserving_bins(y, p, 3)
    assert len(bins) == 2 and sorted(bins.rows.tolist()) == [3, 3]


def test_boundary_distance_tie_selects_lower_cumulative_count():
    y = np.array([0, 0, 1, 1]); p = np.array([.1, .2, .3, .4])
    bins = tie_preserving_bins(y, p, 3)
    assert bins.iloc[0].rows == 1


def test_few_unique_probabilities_reduce_actual_bins():
    y = np.array([0, 1] * 10); p = np.array([.2] * 10 + [.8] * 10)
    bins = tie_preserving_bins(y, p, 20)
    assert bins.actual_bin_count.iloc[0] == 2


def test_bin_diagnostics_formula():
    bins = pd.DataFrame({"rows": [1, 3], "prediction_minus_observed_gap": [.2, -.1]})
    assert bin_diagnostics(bins) == pytest.approx((.125, .2))


def test_top_risk_includes_complete_cutoff_tie_block():
    y = np.array([1, 0, 1, 0, 0]); p = np.array([.9, .8, .8, .8, .1])
    row = top_risk_summary(y, p, shares=[.4])[0]
    assert row["actual_rows"] == 4 and row["cutoff_probability"] == .8


def test_joblib_calibrator_reload_equality(tmp_path):
    y = fixture_labels(); p = fixture_probabilities(y); model, _ = fit_logistic_calibrator(p, y)
    path = tmp_path / "calibrator.joblib"; joblib.dump(model, path)
    np.testing.assert_array_equal(predict_logistic_calibrator(model, p), predict_logistic_calibrator(joblib.load(path), p))


def test_completion_rejects_missing_empty_and_hash_mismatch(tmp_path):
    status = tmp_path / "status.json"; artifact = tmp_path / "a.bin"; artifact.write_bytes(b"a")
    assert not completion_check(status, "cfg", [artifact])[0]
    status.write_text(json.dumps({"status": "COMPLETE", "config_hash": "cfg", "artifacts": []}))
    assert not completion_check(status, "cfg", [artifact])[0]
    status.write_text(json.dumps({"status": "COMPLETE", "config_hash": "cfg", "artifacts": [{"path": str(artifact), "sha256": "bad"}]}))
    assert not completion_check(status, "cfg", [artifact])[0]


def test_completion_accepts_hash_valid_artifact(tmp_path):
    artifact = tmp_path / "a.bin"; artifact.write_bytes(b"a"); status = tmp_path / "status.json"
    status.write_text(json.dumps({"status": "COMPLETE", "config_hash": "cfg", "artifacts": [{"path": str(artifact), "sha256": sha256_file(artifact)}]}))
    assert completion_check(status, "cfg", [artifact])[0]


def test_json_rejects_nan_and_infinity(tmp_path):
    with pytest.raises(ValueError): json_safe({"x": np.nan})
    with pytest.raises(ValueError): atomic_json(tmp_path / "bad.json", {"x": np.inf})


def test_selected_method_unique_across_annotations():
    selected, rows = mechanical_selection([selection_row("identity", loss=.12, se=.001), selection_row("logistic", loss=.10), selection_row("isotonic", loss=.11)])
    assert selected == "logistic" and [row["method"] for row in rows if row["selected"]] == ["logistic"]


def synthetic_bootstrap(losses, replicates=5):
    rows = []
    offsets = np.array([-.002, -.001, 0, .001, .002])[:replicates]
    for replicate_id, offset in enumerate(offsets):
        for method, value in losses.items():
            rows.append({"replicate_id": replicate_id, "model_id": "m", "method": method,
                         "bootstrap_log_loss": value + offset * {"identity": 1.2, "logistic": 1.0, "isotonic": .8}[method],
                         "bootstrap_brier_score": .2, "brier_difference_vs_identity": 0.0,
                         "estimate_role": "OOF_CALIBRATION_DEVELOPMENT_ESTIMATE"})
    return pd.DataFrame(rows)


def test_v2_excludes_unconstrained_logloss_best_before_one_se_reference():
    rows = [selection_row("identity", loss=.100001), selection_row("logistic", loss=.100000),
            selection_row("isotonic", loss=.099000, ranking=False)]
    selected, annotated = mechanical_selection(rows, synthetic_bootstrap({"identity": .100001, "logistic": .100000, "isotonic": .099000}))
    by_method = {row["method"]: row for row in annotated}
    assert selected == "identity"
    assert by_method["isotonic"]["hard_admissible"] is False
    assert by_method["identity"]["admissible_numeric_best_reference"] == "logistic"
    assert by_method["identity"]["passes_log_loss_one_se"] is True


def test_v2_excludes_brier_failure_before_reference_selection():
    rows = [selection_row("identity", loss=.101), selection_row("logistic", loss=.100),
            selection_row("isotonic", loss=.099, brier=False)]
    selected, annotated = mechanical_selection(rows, synthetic_bootstrap({"identity": .101, "logistic": .100, "isotonic": .099}))
    assert selected in {"identity", "logistic"}
    assert {row["method"]: row for row in annotated}["identity"]["admissible_numeric_best_reference"] == "logistic"


def test_valid_identity_is_hard_admissible_and_nonempty():
    selected, annotated = mechanical_selection([selection_row("identity", loss=.1), selection_row("logistic", loss=.2, brier=False), selection_row("isotonic", loss=.3, ranking=False)])
    identity = {row["method"]: row for row in annotated}["identity"]
    assert selected == "identity" and identity["hard_admissible"] and identity["eligible"]


@pytest.mark.parametrize("values", [
    np.array([np.nan], dtype=np.float64), np.array([[.1]], dtype=np.float64),
    np.array([.1], dtype=np.float32), np.array([1.1], dtype=np.float64),
])
def test_invalid_raw_identity_probability_contract_is_specific(values):
    with pytest.raises(RawIdentityPredictionError, match="INVALID_RAW_IDENTITY_PREDICTIONS"):
        validate_raw_identity_predictions(values, 1)


def test_tie_tolerance_applies_only_inside_hard_admissible_set():
    rows = [selection_row("identity", loss=.1 + 1e-12), selection_row("logistic", loss=.1),
            selection_row("isotonic", loss=.05, ranking=False)]
    selected, annotated = mechanical_selection(rows)
    assert selected == "identity"
    assert {row["method"]: row for row in annotated}["identity"]["admissible_numeric_best_reference"] == "identity"


def test_bootstrap_absolute_metrics_reconstruct_both_paired_differences():
    y = np.array([0, 0, 0, 1, 1], dtype=np.int8)
    identity = np.array([.1, .2, .3, .7, .9], dtype=np.float64)
    logistic = np.array([.11, .19, .29, .71, .88], dtype=np.float64)
    frame, _ = paired_stratified_bootstrap(y, {("m", "identity"): identity, ("m", "logistic"): logistic}, None, replicates=10)
    annotated = annotate_bootstrap_references(frame, {"m": "logistic"})
    pivot_ll = annotated.pivot(index="replicate_id", columns="method", values="bootstrap_log_loss")
    pivot_brier = annotated.pivot(index="replicate_id", columns="method", values="bootstrap_brier_score")
    for method in ("identity", "logistic"):
        rows = annotated[annotated.method.eq(method)].sort_values("replicate_id")
        np.testing.assert_allclose(rows.log_loss_difference_vs_admissible_numeric_best, pivot_ll[method] - pivot_ll["logistic"], rtol=0, atol=0)
        np.testing.assert_allclose(rows.brier_difference_vs_identity, pivot_brier[method] - pivot_brier["identity"], rtol=0, atol=0)
        assert paired_standard_error(rows.log_loss_difference_vs_admissible_numeric_best) == np.std(rows.log_loss_difference_vs_admissible_numeric_best, ddof=1)


def valid_saved_selection_fixture():
    losses = {"identity": .100001, "logistic": .100000, "isotonic": .099000}
    bootstrap = synthetic_bootstrap(losses)
    selected, rows = mechanical_selection([
        selection_row("identity", loss=losses["identity"]),
        selection_row("logistic", loss=losses["logistic"]),
        selection_row("isotonic", loss=losses["isotonic"], ranking=False),
    ], bootstrap)
    annotated_bootstrap = annotate_bootstrap_references(bootstrap, {"m": rows[0]["admissible_numeric_best_reference"]})
    frame = pd.DataFrame(rows)
    frame["model_id"] = "m"
    return selected, frame, annotated_bootstrap


def test_saved_selection_independent_reconstruction_rejects_tampering():
    selected, frame, bootstrap = valid_saved_selection_fixture()
    assert verify_v2_selection_table(frame, bootstrap) == {"m": selected}

    bad_reference = frame.copy()
    bad_reference["admissible_numeric_best_reference"] = "isotonic"
    with pytest.raises(CalibrationContractError, match="reference"):
        verify_v2_selection_table(bad_reference, bootstrap)

    bad_selected = frame.copy()
    bad_selected["selected"] = False
    bad_selected.loc[bad_selected.method.eq("isotonic"), "selected"] = True
    with pytest.raises(CalibrationContractError):
        verify_v2_selection_table(bad_selected, bootstrap)

    duplicate_selected = frame.copy()
    duplicate_selected.loc[duplicate_selected.method.eq("logistic"), "selected"] = True
    with pytest.raises(CalibrationContractError, match="Selected method count"):
        verify_v2_selection_table(duplicate_selected, bootstrap)

    inconsistent = frame.copy()
    inconsistent.loc[inconsistent.method.eq("identity"), "eligible"] = False
    with pytest.raises(CalibrationContractError, match="Inconsistent method-selection row"):
        verify_v2_selection_table(inconsistent, bootstrap)
