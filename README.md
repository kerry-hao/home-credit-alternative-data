# The Unequal Value and Cost of Alternative Data in Credit Risk Prediction

**Status: In development.** No models have been trained and no model
performance or lender-value results are available.

## Project purpose

Compare the predictive value of traditional credit information and
alternative data, then examine how model differences translate into
simulated approval decisions and scenario-based lender value.

This repository is being developed as a reproducible credit-risk
research and recruitment showcase.

## First-release scope

- Regularized logistic regression, one gradient-boosted decision-tree
  model, and a small tabular neural network.
- Each model evaluated with traditional information (T) and traditional
  plus verified alternative information (T+AD).
- Candidate alternative-data sources: debit-card transactions,
  deposits and balances, and tax-registry records.
- ROC-AUC, Average Precision, log loss, Brier score, and calibration
  evaluation, retaining raw and calibrated results.
- One credit-information thickness grouping, subject to field verification.
- Rank changes and approval-switch summaries.
- Fixed-approval-rate and fixed-portfolio-risk-target simulations.
- Scenario-based lender value and break-even alternative-data cost.
- A Streamlit demonstration using reviewed aggregate results only.

Individual alternative-data module ablations, ensembles, sequence
models, and extensive demographic analyses are deferred.

## Evaluation design

The analysis unit is an application record keyed by case_id.

The primary design uses researcher-approved target-stratified random
partitions: 70% training, 15% validation, and 15% final evaluation. The
validation partition is reserved approximately equally for tuning and
calibration. Membership and seeds are frozen by the training-preparation
pipeline. The first six TRAIN/validation-tuning model comparisons described
below have now been trained; calibration and final evaluation remain frozen.

A limited chronological supplement will compare GBDT(T) with GBDT(T+AD)
for predictive performance and calibration.

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

The formal target horizon and detailed feature definitions remain to
be verified. This does not prevent development of the application-level
prediction pipeline, but limits claims about real-time deployment.

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
- [ ] Complete the chronological GBDT supplement.
- [ ] Implement approval and scenario-value evaluation.
- [ ] Build and verify the aggregate-results Streamlit demonstration.

There is no live demo or published code link yet.

<!-- TASK09_RESULTS_START -->
## First model training

The six prespecified Logit, LightGBM, and MLP T/T+AD combinations were trained sequentially on the frozen outer TRAIN partition under run `first_full`. Reported scores are validation-tuning development metrics, not final evaluation results.

```bash
/opt/anaconda3/envs/home_credit/bin/python -u scripts/train_baseline_models.py --data-root /Users/haoguannan/Projects/home_credit/data --run-id first_full --models all --threads 4
```

See `docs/first_model_training.md` for the model contracts, aggregate results, and limitations.
<!-- TASK09_RESULTS_END -->
