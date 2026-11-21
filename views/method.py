"""Method page: data, models, PD conversion, lending rules and statistics."""

import plotly.graph_objects as go
import streamlit as st

from src.theme import INK, base_layout
from src.ui import CHART_CONFIG, plot, chapter, lead, load, page_title, table

d = load()
page_title("Method", "Data, models and rules")

# ---------------------------------------------------------------- data
chapter("Data and samples", first=True)
lead("Home Credit consumer-loan applications with an observed default outcome, from the Kaggle competition "
     "<i>Credit Risk Model Stability</i>. The unit of analysis is one application. Applications are split at random "
     "into four samples that never overlap. Models, PD conversion and lending rules are all fixed before the final "
     "sample is used.")
parts = [("Training", 1_068_661, "#3D444D"), ("Tuning", 114_499, "#6E7781"),
         ("Calibration", 114_500, "#9AA1A9"), ("Final evaluation", 228_999, INK)]
fig = go.Figure()
for name, n, color in parts:
    fig.add_trace(go.Bar(x=[n], y=[""], orientation="h", marker=dict(color=color, line=dict(color="white", width=2)),
                         hovertemplate=f"{name}: {n:,}<extra></extra>"))
    fig.add_annotation(x=sum(p[1] for p in parts[:parts.index((name, n, color))]) + n / 2, y=0, yshift=-38,
                       text=f"<b>{name}</b><br>{n:,}", showarrow=False, font=dict(size=13, color=INK))
base_layout(fig, height=150, title="1,526,659 applications")
fig.update_layout(barmode="stack", margin=dict(b=60))
fig.update_xaxes(visible=False, range=[0, 1_526_659])
fig.update_yaxes(visible=False)
plot(fig)
lead("The final sample has 228,999 applications and 7,199 defaults (3.14%): 142,910 women with a default rate of "
     "2.63% and 86,089 men with 3.99%. The calibration sample has 114,500 applications and 3,600 defaults.")

# ---------------------------------------------------------------- models
chapter("Models")
lead("Three algorithms, each trained twice. The traditional model uses application and credit-bureau information "
     "(21 inputs). The model with alternative data adds debit-card, deposit and tax records. LightGBM performs best "
     "with either information set, so the Story and Evidence pages use the two LightGBM models.")
six = d["me_six_models"]
rows = [[r.algorithm, "Traditional" if r.information_set == "T" else "With alternative data", f"{r.auc:.4f}",
         f"{r.ap:.4f}", f"{r.log_loss:.5f}", f"{r.brier:.5f}", f"{r.r1_50_approved_default_rate_pct:.2f}%",
         f"{int(r.r3_30_realised_value):,}"] for r in six.itertuples()]
table(["Algorithm", "Information", "AUC", "Average precision", "Log loss", "Brier score",
       "Default rate of approved, 50% approval", "Lender value, L/G = 30"], rows,
      align=["left", "left"] + ["right"] * 6,
      note="Final sample. AUC and average precision: higher is better. Log loss and Brier score: lower is better.")

# ---------------------------------------------------------------- gender
chapter("Gender and PD conversion")
lead("Gender comes from the main applicant's record. It is never an input of any credit model. It is used to "
     "compare outcomes and, in the fix, only inside the step that turns the model's score into a PD.")
table(["PD conversion", "What it does", "Fitted on"], [
    ["Gender-blind", "The frozen calibration of each model. For LightGBM the identity map was selected: the "
                     "model's own probability is the PD.", "Calibration sample"],
    ["Gender-aware, level", "Logit of default on logit(PD) and a female indicator. Same score, lower PD for women.",
     "Calibration sample"],
    ["Gender-aware, level and slope", "Adds female × logit(PD), so the adjustment can vary with the risk level.",
     "Calibration sample"],
], align=["left", "left", "left"],
    note="Each calibrator is saved and frozen before any final-sample outcome is read, then applied to the final "
         "sample once. A gender-free version of the same calibrator leaves AUC and the approvals at 50% unchanged.")

# ---------------------------------------------------------------- lending rules
chapter("Lending rules")
lead("Every rule ranks applications by PD from low to high, with ties broken in a fixed order.")
table(["Rule", "Approves", "Grid", "Main point"], [
    ["Fixed approval rate", "The same number of lowest-PD applications under each model", "161 rates, 40%–80%",
     "50% (114,499 applications)"],
    ["Portfolio risk budget", "Lowest-PD applications until their mean PD reaches the budget",
     "126 budgets, 0.75%–2.00%", "1%"],
    ["Economic value", "Applications with PD below 1 / (1 + L/G). Value = repaid loans − L/G × defaults",
     "L/G = 2, 5, 10, 20, 30, 50", "L/G = 30 (PD below 3.23%)"],
], align=["left", "left", "left", "left"],
    note="L/G is the loss from one default relative to the gain from one repaid loan. Values are in units of the "
         "gain from one repaid loan.")

# ---------------------------------------------------------------- statistics
chapter("Statistics")
table(["Quantity", "Method"], [
    ["Calibration gap by gender", "Logit of default on logit(PD) and a female indicator; HC1 standard errors"],
    ["Contribution of each input", "Exact SHAP decomposition of the LightGBM score (pred_contrib), averaged by gender"],
    ["Difference in AUC", "DeLong test"],
    ["Changes in approvals, defaults and value", "Paired bootstrap, 1,000 resamples of applications, rules re-run in "
                                                 "each resample"],
    ["Default rates", "Wilson 95% intervals"],
], align=["left", "left"], max_width="900px")

st.markdown('<p class="tbl-note" style="margin-top:2rem">Data: Daniel Herman, Tomas Jelinek, Walter Reade, '
            'Maggie Demkin and Addison Howard. Home Credit – Credit Risk Model Stability. Kaggle, 2024. '
            'All figures on this site are aggregates.</p>', unsafe_allow_html=True)
