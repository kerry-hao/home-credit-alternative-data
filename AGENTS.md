# Project execution rules

## Authority
The researcher defines the research design and approves each task.
Execute only the current explicitly authorized task.
The research proposal provides background; it does not authorize
implementing every proposed analysis.

Do not independently select features, time partitions, borrower-group
definitions, calibration methods, model settings, decision thresholds,
or economic assumptions. Report unresolved questions and stop at the
relevant decision boundary.

## Recruitment showcase scope
The first release targets:
- Traditional information (T) versus T plus verified alternative data.
- Regularized logistic regression, one GBDT, and a small tabular MLP,
  each evaluated with both information sets.
- Ranking, probability-quality, and calibration evaluation.
- One verified credit-information thickness grouping.
- Simple rank-change and approval-switch summaries.
- Fixed-approval-rate and fixed-portfolio-risk-target simulations.
- Scenario-based lender value and break-even alternative-data cost.
- Reproducible GitHub code and an aggregate-results Streamlit demo.

Implement logistic regression and GBDT comparisons before the MLP.
The primary random-split pipeline covers all six model/information-set
combinations and the full evaluation-to-decision showcase.
The first chronological supplement is limited to GBDT(T) versus
GBDT(T+AD), including predictive and calibration evaluation.
Do not duplicate the full six-model decision/economic pipeline across
both designs unless separately authorized.
Declare the roles of both designs before evaluating model results;
do not select the primary design based on favorable test performance.

Individual alternative-data module ablations, ensembles, sequence
models, extensive demographic analyses, personalized pricing, and
actual cash-flow reconstruction are deferred unless separately approved.

This scope is not authorization to start these tasks.

## Research integrity
Use application records as the unit of analysis.
Use researcher-approved, target-stratified random partitions as the
primary evaluation design, with a fixed random seed.
Use chronological evaluation as a limited supplementary analysis.
Exact partition proportions, dates, and seeds require separate
researcher approval. Do not implement partitions in advance.

The primary evaluation targets held-out applications from the mixed
historical sample; it is not evidence of prospective deployment.
Do not make borrower-level grouping or repeat-borrower identification
a prerequisite for the first release.
Keep application partitions disjoint. Use case_id as a join key,
not as a predictive feature.
Do not use final test labels for preprocessing, feature selection,
tuning, calibration, group-boundary selection, or policy selection.
Fit preprocessing steps that learn parameters from data only on the
corresponding model-training partition. Apply the fitted transformations
to validation, calibration, policy-selection, and test partitions without
refitting them there. This rule does not prohibit fitting a calibrator
on its separately authorized calibration partition.
Audit feature timing and label availability; dates alone do not
establish freedom from leakage.

Compare information sets on the same approved evaluation sample.
Do not interpret missing records automatically as confirmed absence.
Do not claim outcomes for historically rejected applicants are observed.
Do not describe scenario value as actual accounting profit.
Never invent results or use placeholder performance claims.

## Files and environment
Repository:
  /Users/haoguannan/Projects/home_credit/home-credit-alternative-data
External data:
  /Users/haoguannan/Projects/home_credit/data
Research sources:
  /Users/haoguannan/Projects/home_credit/research_sources
Python:
  /opt/anaconda3/envs/home_credit/bin/python

Use the specified Python executable for project commands.
Treat raw data and research source documents as immutable.
Do not relocate them into the repository.
Do not install packages or change environments without authorization.

## Publication and permissions
Do not commit, push, deploy, or create remote repositories unless
explicitly instructed.
Never publish raw data, borrower-level features, identifiers,
individual predictions, or confidential research sources.
Only researcher-reviewed aggregate results may be published.
Do not assume .gitignore is sufficient to make publication safe.

If access is denied, report the blocked path and stop.
Do not broaden permissions or bypass restrictions.
Do not delete existing files or temporary directories unless the
specific deletion is authorized.

## Reporting
After each task, report:
- What was executed.
- Files created or changed.
- Verification commands and results.
- Errors, limitations, and unresolved decisions.

Then stop. Do not begin the next phase.
