# The Unequal Value and Cost of Alternative Data in Credit Risk Predictions

**Status: Task 12 formal final evaluation complete.** Feature construction, the frozen random
split, six-model training, calibration, contract freeze, and held-out evaluation are complete.

## Interactive showcase

**The Unequal Value and Cost of Alternative Data in Credit Risk Predictions** is a compact Streamlit/Plotly view of
the reviewed aggregate final-evaluation results.

**Live Streamlit URL:** _placeholder — add after manual review and deployment._

<!-- TASK13_SCREENSHOT_PLACEHOLDER: replace with assets/task13_showcase.png after review -->

Explore five views: Model performance, Information heterogeneity, Risk
re-ranking, Credit re-allocation, and Lender-side value. The app uses only
compact aggregate CSV files under `data/showcase/`.

```bash
python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
```

## Project purpose

This project asks whether verified alternative data improve default
prediction beyond traditional credit information, for whom they help, how
they change relative risk rankings and simulated approvals, and what
normalized scenario-based lender value follows.

This repository is being developed as a reproducible credit-risk
research and recruitment showcase.

## First-release scope

- Regularized logistic regression, one gradient-boosted decision-tree
  model, and a small tabular neural network.
- Each model evaluated with traditional information (T) and traditional
  plus verified alternative information (T+AD).
- Candidate alternative-data sources: debit-card transactions,
  deposits and balances, and tax-registry records.
- ROC-AUC, Average Precision, log loss, Brier score, and calibrated
  probability evaluation using each model's frozen selected calibration.
- Separate pre-model account-count, bureau-history, and observed
  alternative-data-richness subgroup dimensions.
- Rank changes and approval-switch summaries.
- Fixed-approval-rate and fixed-portfolio-risk-target simulations.
- Normalized common gain/loss scenario value and break-even
  alternative-data cost; these are not actual accounting profit or currency.
- A compact aggregate-results Streamlit demonstration with five interactive
  views.

Individual alternative-data module ablations, ensembles, sequence
models, and extensive demographic analyses are deferred.

## Evaluation design

The analysis unit is an application record keyed by case_id.

The final analysis population contains 1,526,659 labeled applications under
a frozen, target-stratified random split: 1,068,661 training cases (70%),
228,999 validation cases (15%), and 228,999 final-evaluation cases (15%). The
validation partition contains 114,499 tuning cases and 114,500 calibration
cases. Membership, order, and seeds are frozen by the Task 8 preparation
pipeline.

Six model systems are trained and frozen: regularized Logit, LightGBM, and a
small tabular MLP, each using traditional information (T) and traditional plus
verified alternative data (T+AD). Calibration is complete: identity is
selected for both Logit models, both LightGBM models, and `mlp_T_plus_AD`;
logistic calibration is selected for `mlp_T`.

Tasks 8–10 are the frozen modeling foundation. Task 11 freezes the final
subgroup, policy, output-schema, and normalized gain/loss analysis contract.
The primary comparison is `lightgbm_T` versus `lightgbm_T_plus_AD`. The
secondary T+AD complexity comparison covers Logit, LightGBM, and MLP, while
all six model outputs are retained in final results.

Random-split results describe held-out applications from the mixed
historical sample. They do not establish performance under future
deployment. Borrower identity and repeat-borrower dependence are not
resolved in the first release.

Learned preprocessing, model selection, calibration, and policy
selection must not use final test labels. Both information sets must
be evaluated on the same application sample.

Preprocessing parameters are learned only from the corresponding model
training partition. The fitted transformations are applied to the other
partitions without refitting. Calibration uses its separately designated
calibration partition.

All approvals are simulations. Scenario value is not actual accounting
profit, and the project does not claim causal effects or outcomes for
historically rejected applicants.

## Data

Source: [Home Credit — Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data)

The initial base-table audit reports:

- 1,526,659 labeled records in train_base.parquet.
- Decision dates from 2019-01-01 through 2020-10-05.
- 47,994 target-1 records, an overall event rate of approximately 3.14%.
- No missing base-table fields or duplicate case_id values.
- 10 unlabeled records in the supplied test_base.parquet.

The supplied competition test table is not used for model evaluation.
Research evaluation partitions will be derived from the labeled data.

The analysis uses the provided binary default target as its outcome. It does
not invent or claim a month-specific default horizon that the supplied target
does not establish. This limits claims about real-time deployment.

Raw data and private audit reports are stored outside this repository
and are not distributed here. Obtain data through the official source
and follow its applicable terms.

## Local setup

The current development environment is the home_credit Conda environment,
using Python 3.11.

Activate it with:

```bash
conda activate home_credit
```

The intended local layout uses sibling directories:

- home-credit-alternative-data: this code repository.
- data/raw/parquet_files: local raw Parquet inputs.
- data/audits: private audit outputs.
- research_sources: local research documents.

The project is not yet fully packaged. A reproducible dependency
specification will be added after the modeling stack is selected.

## Run the base audit

From the repository root, with the project environment active:

```bash
python scripts/audit_base.py \
  --train ../data/raw/parquet_files/train/train_base.parquet \
  --test ../data/raw/parquet_files/test/test_base.parquet \
  --json-output ../data/audits/task03_rerun/base_audit.json \
  --weekly-output ../data/audits/task03_rerun/weekly_summary.csv
```

Choose new output paths for each run. The script refuses to overwrite
existing outputs. It audits all rows and verifies input hashes remain
unchanged.

Detailed validation findings are recorded in the JSON report.
Completed execution does not necessarily mean all validation checks passed.

The original Task 03 reports predate the revised status-reporting format
and are retained unchanged.

## Current progress

- [x] Establish execution rules and Git safeguards.
- [x] Complete the initial base-table audit.
- [x] Verify feature-table structure and candidate feature sources.
- [x] Build the application-level feature table.
- [x] Freeze common evaluation partitions and prepare train-only model inputs.
- [x] Train and compare the six primary model combinations on validation-tuning.
- [x] Complete probability calibration and freeze one selected method per model.
- [x] Freeze subgroup, policy, normalized economic, and result-file contracts.
- [x] Execute the formal 228,999-case final evaluation.
- [x] Build and verify the aggregate-results Streamlit demonstration.

The local showcase is complete; the live deployment link remains intentionally blank pending manual review.

<!-- TASK09_RESULTS_START -->
## First model training

The six prespecified Logit, LightGBM, and MLP T/T+AD combinations were trained sequentially on the frozen outer TRAIN partition under run `first_full`. Reported scores are validation-tuning development metrics, not final evaluation results.

```bash
python -u scripts/train_baseline_models.py --data-root ../data --run-id first_full --models all --threads 4
```

See `docs/first_model_training.md` for the model contracts, aggregate results, and limitations.

The bounded post-training review uses `scripts/verify_task09_post_review.py`; it preserves the original `first_full` artifacts and writes separate inference/test evidence under the external audit root.
<!-- TASK09_RESULTS_END -->

<!-- TASK10_RESULTS_START -->
## Probability calibration

Task 10 run `first_full_v3` calibrated the six frozen Task 09 models without retraining them. It used only `validation_calibration`, with identity, logistic, and isotonic candidates, common five-fold cross-fitting, and 2,000 paired target-stratified bootstrap replicates.

Hard Brier and ranking constraints were applied before choosing the admissible Log-loss reference. The paired one-standard-error rule then selected the simplest statistically competitive admissible method. These are calibration-development results; final evaluation remained untouched and no lending threshold was selected.

See `docs/probability_calibration.md` for verified aggregate results and interpretation limits.
<!-- TASK10_RESULTS_END -->

<!-- TASK11_DESIGN_FREEZE_START -->
## Task 11 analysis design freeze

Task 11 design freeze is complete through Part 3. Its three separate subgroup
dimensions are assigned from pre-model values:

- account count: `0`, `1`, `2`, and `3_plus`;
- bureau history: `0_to_0_5`, `0_5_to_1`, `1_to_2`, `2_to_3`, `3_to_4`,
  `4_to_5`, and `gt_5` years, with missing history assigned to the first group;
- observed AD-module richness: `0`, `1`, `2`, or `3` numeric-content modules.

The fixed-approval curve spans 40%–80% in 0.25-point increments, with 40%,
50%, 60%, 70%, and 80% anchors. The portfolio mean calibrated-PD budget
curve spans 0.75%–2.00% in 0.01-point increments, with 0.75%, 1.00%, 1.25%,
1.50%, 1.75%, and 2.00% anchors.

Normalized loss-to-gain ratios are 2, 5, 10, 20, 30, and 50. Ratio 30 is the
primary scenario; 20 and 50 are lower- and higher-loss sensitivities. This is
a normalized scenario analysis, not Home Credit accounting profit.
<!-- TASK11_DESIGN_FREEZE_END -->

<!-- TASK12_FINAL_EVALUATION_START -->
## Formal final evaluation

Task 12 formal final evaluation is complete on 228,999 frozen
applications: 7,199 defaults and 221,800
non-defaults, for an observed default rate of 3.143682%.

| Model | ROC-AUC | Average precision | Log loss | Brier score | Mean calibrated PD |
|---|---:|---:|---:|---:|---:|
| `logit_T` | 0.724902 | 0.084309 | 0.130150 | 0.029876 | 0.031444 |
| `logit_T_plus_AD` | 0.741731 | 0.089229 | 0.128465 | 0.029762 | 0.031442 |
| `lightgbm_T` | 0.762906 | 0.116995 | 0.124582 | 0.029126 | 0.031377 |
| `lightgbm_T_plus_AD` | 0.778664 | 0.124415 | 0.122744 | 0.028980 | 0.031369 |
| `mlp_T` | 0.755623 | 0.109761 | 0.125596 | 0.029254 | 0.031411 |
| `mlp_T_plus_AD` | 0.771470 | 0.115060 | 0.123856 | 0.029137 | 0.031171 |

The prespecified primary comparison is `lightgbm_T` versus
`lightgbm_T_plus_AD`; all six frozen model results are retained. Separate
account-count, bureau-history, and AD-richness results are available alongside
the fixed-approval and fixed-portfolio-risk curves.

Economic results use normalized common G/L scenarios and report both
model-predicted expected value and target-based realised scenario value. They
are not actual accounting profit or currency. The within-family file contains
18 T-versus-T+AD rows; 18 directed T+AD complexity comparisons are retained in
the separate supplemental `economic_complexity_comparisons.csv`. See
[`overall_model_metrics.csv`](../data/final_evaluation/task12/overall_model_metrics.csv) and the
[`Task 12 report`](../data/final_evaluation/task12/task12_final_evaluation_report.md).
<!-- TASK12_FINAL_EVALUATION_END -->
