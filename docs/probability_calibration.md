# Probability calibration

Task 10 calibrated six frozen Task 09 models without retraining them. Calibration used only the `validation_calibration` partition. Each model compared identity, logistic, and isotonic mappings using common five-fold cross-fitting and 2,000 paired target-stratified bootstrap replicates.

Hard probability-validity, Brier, and ranking constraints were applied before identifying the admissible numerical Log-loss best. The paired one-standard-error rule then selected the simplest statistically competitive admissible method.

## Verified calibration-development results

| Model | Selected method | Raw Log loss | OOF selected Log loss | Raw Brier | OOF selected Brier |
|---|---|---:|---:|---:|---:|
| logit_T | identity | 0.13031780 | 0.13031780 | 0.02987300 | 0.02987300 |
| logit_T_plus_AD | identity | 0.12867510 | 0.12867510 | 0.02976030 | 0.02976030 |
| lightgbm_T | identity | 0.12475107 | 0.12475107 | 0.02917307 | 0.02917307 |
| lightgbm_T_plus_AD | identity | 0.12302074 | 0.12302074 | 0.02903414 | 0.02903414 |
| mlp_T | logistic | 0.12592211 | 0.12586080 | 0.02929917 | 0.02929020 |
| mlp_T_plus_AD | identity | 0.12399717 | 0.12399717 | 0.02916028 | 0.02916028 |

## Interpretation limits

The OOF metrics are calibration-development estimates, not final unbiased performance estimates. Full-fit calibration probabilities are artifact and structural diagnostics, not evidence of out-of-sample improvement. Identity means the raw base probability was retained because no admissible more complex method demonstrated sufficient stable benefit under the fixed rule; it is not an improvement claim.

Final evaluation remained untouched and no lending threshold was selected. The unit is an application because no reliable borrower identifier is available. This random-design analysis does not establish future-time stability, causality, production readiness, borrower-level independence, or counterfactual outcomes for historically rejected applicants.

## Reproduction

```bash
/opt/anaconda3/envs/home_credit/bin/python -u scripts/calibrate_baseline_models.py --data-root /Users/haoguannan/Projects/home_credit/data --base-run-id first_full --run-id first_full_v3 --models all --threads 4
```

Add `--verify-only` to run the separate saved-artifact verification. Application-level predictions, labels, calibrators, and private audit evidence remain under the external data root.
