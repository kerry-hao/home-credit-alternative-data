import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_training_data import (  # noqa: E402
    AD_INDICATORS,
    AD_TAX_COUNT,
    AD_TAX_MONETARY,
    ALIAS_NEW,
    DPD_FIELDS,
    MISSING_TOKEN,
    T_CATEGORICAL,
    T_COUNT,
    T_MONETARY,
    UNSEEN_TOKEN,
    TrainingPreprocessor,
    apply_alias,
    make_splits,
)


T_FEATURES = [
    "t__recorded_primary_income",
    "t__applicant_main_income",
    "t__requested_credit_amount",
    "t__current_application_annuity",
    "t__credit_product_type",
    "t__applicant_income_type",
    "t__applicant_education",
    "t__current_debt",
    "t__total_debt",
    "t__observed_active_credit_count",
    *DPD_FIELDS,
    "t__bureau_queries_30d",
    "t__bureau_queries_360d",
    "t__client_loan_payment_count",
    "t__applications_30d",
    "t__contracts_3m",
    "t__active_revolving_credit_count",
    ALIAS_NEW,
    "t__paid_installments_last_contract",
]
AD_DEBIT = [
    "ad__debit__last30dayturnover_651A__finite_mean",
    "ad__debit__last30dayturnover_651A__finite_max",
    "ad__debit__last180dayturnover_1134A__finite_mean",
    "ad__debit__last180dayturnover_1134A__finite_max",
    "ad__debit__last180dayaveragebalance_704A__finite_mean",
    "ad__debit__last180dayaveragebalance_704A__finite_max",
    "ad__debit__amtdebitincoming_4809443A",
    "ad__debit__amtdebitoutgoing_4809440A",
]
AD_DEPOSIT = [
    "ad__deposit__amount_416A__finite_mean",
    "ad__deposit__amount_416A__finite_max",
    "ad__deposit__amtdepositbalance_4809441A",
    "ad__deposit__amtdepositincoming_4809444A",
    "ad__deposit__amtdepositoutgoing_4809442A",
]
AD_SUBSTANTIVE = [*AD_DEBIT, *AD_DEPOSIT, *AD_TAX_COUNT, *AD_TAX_MONETARY]


def feature_sets():
    return {
        "T": T_FEATURES,
        "AD_substantive_values": AD_SUBSTANTIVE,
        "AD_indicators": AD_INDICATORS,
        "T_plus_AD": [*T_FEATURES, *AD_SUBSTANTIVE, *AD_INDICATORS],
        "AD_substantive_by_family": {
            "debit": AD_DEBIT,
            "deposit": AD_DEPOSIT,
            "tax": [*AD_TAX_COUNT, *AD_TAX_MONETARY],
        },
    }


def synthetic_frame(n=100, seed=1):
    rng = np.random.default_rng(seed)
    data = {}
    for column in [c for c in T_FEATURES if c not in T_CATEGORICAL] + AD_SUBSTANTIVE:
        data[column] = rng.uniform(0, 100, n)
    data[T_CATEGORICAL[0]] = np.resize(np.array(["A", "B"], object), n)
    data[T_CATEGORICAL[1]] = np.resize(np.array(["X", "Y", "Z"], object), n)
    data[T_CATEGORICAL[2]] = np.resize(np.array(["M1", "M2"], object), n)
    for column in AD_INDICATORS:
        data[column] = rng.integers(0, 2, n, dtype=np.int8)
    return pd.DataFrame(data)[T_FEATURES + AD_SUBSTANTIVE + AD_INDICATORS]


def fitted(frame, train_n=70):
    train = np.zeros(len(frame), bool)
    train[:train_n] = True
    return TrainingPreprocessor().fit(frame, train, feature_sets(), "split", "keys")


def test_split_is_reproducible_disjoint_and_nested():
    labels = np.resize(np.array([0] * 97 + [1] * 3), 1000)
    outer1, role1 = make_splits(labels)
    outer2, role2 = make_splits(labels)
    assert np.array_equal(outer1, outer2)
    assert np.array_equal(role1, role2)
    assert set(np.unique(outer1)) == {"train", "validation", "evaluation"}
    assert set(np.flatnonzero(role1 != "")) == set(np.flatnonzero(outer1 == "validation"))


def test_alias_requires_exactly_one_name_and_preserves_values():
    old = pd.DataFrame({"t__last_approved_credit_amount": [1.0, 2.0]})
    renamed = apply_alias(old)
    assert renamed[ALIAS_NEW].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        apply_alias(pd.DataFrame({ALIAS_NEW: [1], "t__last_approved_credit_amount": [1]}))


def test_sparse_cap_uses_observed_train_values_not_zero_fill():
    frame = synthetic_frame(100)
    field, paired_max = AD_DEPOSIT[:2]
    frame[[field, paired_max]] = np.nan
    frame.loc[:9, [field, paired_max]] = np.column_stack([np.arange(1.0, 11.0)] * 2)
    prep = fitted(frame)
    expected = np.quantile(np.arange(1.0, 11.0), 0.999, method="linear")
    assert prep.cap_parameters[field]["train_finite_count"] == 10
    assert prep.cap_parameters[field]["cap"] == pytest.approx(expected)
    assert prep.cap_parameters[field]["cap"] > 0


def test_zero_cap_exception_preserves_positive_information():
    frame = synthetic_frame(1100)
    field = AD_DEPOSIT[0]
    frame[field] = 0.0
    frame.loc[1000, field] = 5.0
    prep = fitted(frame, train_n=1001)
    assert prep.cap_parameters[field]["computed_quantile"] == 0.0
    assert prep.cap_parameters[field]["status"] == "CAP_SKIPPED_DEGENERATE_ZERO"
    assert prep.cap_parameters[field]["cap"] is None
    assert prep.transform(frame.iloc[[1000]], "gbdt")[field].iloc[0] == pytest.approx(5.0)


def test_only_ad_monetary_negatives_are_clipped_and_indicator_is_unchanged():
    frame = synthetic_frame(100)
    field = AD_DEBIT[0]
    frame.loc[0, field] = -7.0
    frame.loc[0, "ad__debit__has_nonzero_content"] = 1
    prep = fitted(frame)
    result = prep.transform(frame.iloc[[0]], "gbdt")
    assert result[field].iloc[0] == 0.0
    assert result["ad__debit__has_nonzero_content"].iloc[0] == 1
    bad = synthetic_frame(100)
    bad.loc[0, DPD_FIELDS[0]] = -1.0
    with pytest.raises(ValueError, match="Unexpected negative"):
        fitted(bad)


def test_log_zero_nan_imputation_and_missing_flag_are_distinct():
    frame = synthetic_frame(100)
    field = T_MONETARY[0]
    frame.loc[0, field] = 0.0
    frame.loc[1, field] = np.nan
    prep = fitted(frame)
    linear = prep.transform(frame.iloc[:2], "linear_nn")
    flag = next(x["name"] for x in prep.missing_flags if field in x["dependencies"])
    assert linear[flag].tolist() == [0, 1]
    zero_expected = (0.0 - prep.scaler_means[field]) / prep.scaler_scales[field]
    assert linear[field].iloc[0] == pytest.approx(zero_expected, abs=1e-6)
    assert np.isfinite(linear[field]).all()


def test_all_missing_and_constant_train_fallbacks_are_safe():
    frame = synthetic_frame(100)
    all_missing, paired_max = AD_DEPOSIT[:2]
    constant = T_COUNT[0]
    frame.loc[:69, [all_missing, paired_max]] = np.nan
    frame.loc[:69, constant] = 3.0
    prep = fitted(frame)
    assert all_missing in prep.all_missing_columns
    assert prep.medians[all_missing] == 0.0
    assert constant in prep.constant_columns
    assert prep.scaler_scales[constant] == 1.0
    assert np.isfinite(prep.transform(frame, "linear_nn").to_numpy()).all()


def test_mean_max_pair_shares_flag_but_unrelated_masks_do_not_merge():
    frame = synthetic_frame(100)
    mean, maximum = AD_DEBIT[:2]
    frame.loc[[0, 2], [mean, maximum]] = np.nan
    unrelated = T_MONETARY[0]
    frame.loc[[0, 2], unrelated] = np.nan
    prep = fitted(frame)
    paired = [x for x in prep.missing_flags if mean in x["dependencies"]][0]
    separate = [x for x in prep.missing_flags if unrelated in x["dependencies"]][0]
    assert paired["dependencies"] == [mean, maximum]
    assert paired["name"] != separate["name"]


def test_mismatched_mean_max_masks_are_blocking():
    frame = synthetic_frame(100)
    frame.loc[0, AD_DEBIT[0]] = np.nan
    with pytest.raises(ValueError, match="mean/max masks differ"):
        fitted(frame)


def test_categories_train_only_missing_unseen_and_token_collision():
    frame = synthetic_frame(100)
    field = T_CATEGORICAL[0]
    frame.loc[70, field] = "EVAL_ONLY"
    frame.loc[71, field] = None
    prep = fitted(frame)
    assert "EVAL_ONLY" not in prep.category_vocabularies[field]
    output = prep.transform(frame.iloc[[70, 71]], "linear_nn")
    unseen_col = next(x["output"] for x in prep.category_outputs[field] if x["category"] == UNSEEN_TOKEN)
    missing_col = next(x["output"] for x in prep.category_outputs[field] if x["category"] == MISSING_TOKEN)
    assert output[unseen_col].tolist() == [1, 0]
    assert output[missing_col].tolist() == [0, 1]
    collision = synthetic_frame(100)
    collision.loc[0, field] = MISSING_TOKEN
    with pytest.raises(ValueError, match="collision"):
        fitted(collision)


def test_fit_parameters_ignore_fixed_nontrain_mutations():
    frame = synthetic_frame(100)
    frame.loc[70:, T_MONETARY[0]] = 1e20
    frame.loc[70:, T_CATEGORICAL[0]] = "OUTSIDE"
    first = fitted(frame).to_dict()
    changed = frame.copy()
    changed.loc[70:, T_MONETARY[0]] = 2e30
    changed.loc[70:, T_CATEGORICAL[0]] = "DIFFERENT_OUTSIDE"
    second = fitted(changed).to_dict()
    for key in ("cap_parameters", "medians", "scaler_means", "scaler_scales", "category_vocabularies"):
        assert first[key] == second[key]


def test_T_is_exact_prefix_and_reload_transform_is_identical(tmp_path):
    frame = synthetic_frame(100)
    prep = fitted(frame)
    for representation in ("linear_nn", "gbdt"):
        t = prep.output_features[representation]["T"]
        assert prep.output_features[representation]["T_plus_AD"][: len(t)] == t
    path = tmp_path / "preprocessor.json"
    prep.save(path)
    reloaded = TrainingPreprocessor.load(path)
    for representation in ("linear_nn", "gbdt"):
        a = prep.transform(frame.iloc[[0, 70, 99]], representation)
        b = reloaded.transform(frame.iloc[[0, 70, 99]], representation)
        np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)
    payload = json.loads(path.read_text())
    payload["unknown_parameter"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown serialized"):
        TrainingPreprocessor.load(path)


def test_transform_rejects_missing_schema_and_infinity():
    frame = synthetic_frame(100)
    prep = fitted(frame)
    with pytest.raises(ValueError, match="Missing approved"):
        prep.transform(frame.drop(columns=[T_MONETARY[0]]), "linear_nn")
    invalid = frame.copy()
    invalid.loc[0, T_MONETARY[0]] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        prep.transform(invalid, "gbdt")
