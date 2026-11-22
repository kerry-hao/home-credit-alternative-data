"""Why AD helps: what alternative data adds to prediction and lending."""

import math

import plotly.graph_objects as go
import streamlit as st

from src.theme import BLIND, INK, INK_2, REFERENCE, base_layout, legend_name
from src.ui import chapter, lead, load, m, page_title, plot, table, tiles

d = load()
TRAD, WITH = BLIND, INK
GROUP_LABEL = {"G0": "No record", "G1": "Tax only", "G2": "Tax and deposit", "G3": "Other"}

page_title("Why AD helps", "What alternative data adds")
cov = d["wa_coverage"].set_index("group")
tiles([(f"{cov.loc[g, 'share_pct']:.1f}%", f"{GROUP_LABEL[g]}") for g in ["G1", "G2", "G0", "G3"]])
lead("Alternative data (AD) here means tax, deposit and debit-card records. Most applicants have tax records only, "
     "so in this sample AD is mainly tax data.")

# ---------------------------------------------------------------- 1. better prediction
chapter("Alternative data improves prediction with every algorithm")
gains = d["wa_gains"]
gb = d["wa_gains_b"].set_index("row")
metric = st.segmented_control("Metric", ["AUC", "Average precision"], default="AUC", key="wa_metric") or "AUC"
key = "AUC" if metric == "AUC" else "AP"
chart_col, side, _ = st.columns([0.5, 0.3, 0.2], gap="large")
with chart_col:
    g = gains[gains.metric == key].set_index("algorithm").loc[["MLP", "LightGBM", "Logit"]]
    fig = go.Figure()
    for alg, r in g.iterrows():
        fig.add_trace(go.Scatter(x=[r.value_T, r.value_T_plus_AD], y=[alg, alg], mode="lines", hoverinfo="skip",
                                 line=dict(color="#D5D9DE", width=3), showlegend=False))
    for col, color, name in [("value_T", TRAD, "Traditional"), ("value_T_plus_AD", WITH, "With alternative data")]:
        fig.add_trace(go.Scatter(x=g[col], y=g.index, mode="markers", name=legend_name(name),
                                 marker=dict(size=12, color=color, line=dict(color="white", width=1.5)),
                                 hovertemplate=f"{name}<br>%{{y}}: %{{x:.4f}}<extra></extra>"))
    for alg, r in g.iterrows():
        fig.add_annotation(x=r.value_T_plus_AD, y=alg, text=f"+{100 * r.estimate:.2f} pp", showarrow=False,
                           xanchor="left", xshift=10, font=dict(size=12, color=INK))
    base_layout(fig, height=260, title=f"{metric} on the final sample")
    fig.update_layout(showlegend=True, margin=dict(t=40), legend=dict(orientation="h", x=0, y=1.02, xanchor="left",
                                                                        yanchor="bottom"))
    lo, hi = (0.71, 0.80) if key == "AUC" else (0.075, 0.135)
    fig.update_xaxes(range=[lo, hi], tickformat=".2f" if key == "AUC" else ".3f")
    fig.update_yaxes(showgrid=False)
    plot(fig)
with side:
    lg = gains[(gains.metric == "AUC") & (gains.algorithm == "LightGBM")].iloc[0]
    it = gb.loc["(6)"]
    tiles([(f"+{100 * lg.estimate:.2f} pp", f"AUC gain with LightGBM (95% CI {100 * lg.ci_lo:.2f}–{100 * lg.ci_hi:.2f})"),
           (m(f"{100 * it.estimate:.2f} pp"), f"Difference in the AUC gain, LightGBM minus Logit (p = {it.p:.2f})")],
          css_class="stack")

# ---------------------------------------------------------------- 2. splits look-alikes
chapter("Among applicants the traditional model sees as equally risky, alternative data finds the safer ones")
lead("Applicants are ranked by the traditional model's PD and split into ten equal groups. Inside each group, "
     "alternative data lowers some PDs and raises others.")
dec = d["wa_deciles"]
dl = d["wa_delta_logit"]
delta = dl[(dl.column == "(3)") & (dl.item == "Delta")].iloc[0]
chart_col, side, _ = st.columns([0.5, 0.3, 0.2], gap="large")
with chart_col:
    fig = go.Figure()
    down = dec[dec.direction == "down"].sort_values("decile")
    up = dec[dec.direction == "up"].sort_values("decile")
    mean_t = dec.groupby("decile").apply(lambda x: (x.mean_pT_pct * x.n).sum() / x.n.sum(), include_groups=False)
    fig.add_trace(go.Scatter(x=mean_t.index, y=mean_t.values, mode="lines", name=legend_name("Traditional PD"),
                             line=dict(color=REFERENCE, width=1.5, dash="dash"),
                             hovertemplate="Tenth %{x}: traditional PD %{y:.2f}%<extra></extra>"))
    for frame, name, sym in [(up, "PD raised by AD", "circle-open"), (down, "PD lowered by AD", "circle")]:
        fig.add_trace(go.Scatter(
            x=frame.decile, y=frame.default_rate_pct, mode="markers", name=legend_name(name),
            marker=dict(size=10, color=INK, symbol=sym, line=dict(color=INK, width=1.5)),
            error_y=dict(type="data", symmetric=False, array=frame.wilson_hi_pct - frame.default_rate_pct,
                         arrayminus=frame.default_rate_pct - frame.wilson_lo_pct, color=INK, thickness=1, width=0),
            customdata=frame[["n", "defaults"]],
            hovertemplate=f"{name}, tenth %{{x}}<br>Actual default rate %{{y:.2f}}%<br>"
                          "%{customdata[1]:,} defaults of %{customdata[0]:,}<extra></extra>"))
    base_layout(fig, height=330, title="Actual default rate within each tenth of the traditional PD")
    fig.update_layout(showlegend=True, margin=dict(t=40), legend=dict(orientation="h", x=0, y=1.02, xanchor="left",
                                                                        yanchor="bottom"))
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    fig.update_xaxes(tickvals=list(range(1, 11)), title_text="Tenth of applicants by traditional PD (1 = safest)")
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"{t:g}%" for t in ticks], range=[math.log10(0.1), math.log10(25)],
                     title_text="Actual default rate")
    plot(fig)
with side:
    five = dec[dec.decile == 5].set_index("direction")
    tiles([(f"{five.loc['down', 'default_rate_pct']:.2f}% vs {five.loc['up', 'default_rate_pct']:.2f}%",
            "Middle tenth: default rate when AD lowers vs raises the PD"),
           (f"{delta.estimate:.2f}", f"Actual change in default per unit of AD's adjustment; 1 means exactly right "
                                     f"(SE {delta.se_hc1:.3f})")], css_class="stack")

# ---------------------------------------------------------------- 3. tax records
chapter("Three-quarters of the adjustment comes from tax records")
lead("Each applicant's score change splits exactly into the contribution of each record type and the change in how "
     "the model uses the traditional inputs (SHAP).")
comp = d["wa_components"].set_index("component")
byg = d["wa_components_by_group"].set_index("group")
NAMES = [("C_tax", "Tax records"), ("C_dep", "Deposit records"), ("C_deb", "Debit-card records"),
         ("C_T", "Traditional inputs")]
left, right, _ = st.columns([0.4, 0.4, 0.2], gap="large")
with left:
    vals = [100 * comp.loc[c, "share"] for c, _ in NAMES]
    fig = go.Figure(go.Bar(
        x=vals, y=[n for _, n in NAMES], orientation="h", width=0.5, marker_color=[INK, INK_2, BLIND, "#D5D9DE"],
        text=[f"{v:.1f}%" for v in vals], textposition="outside", cliponaxis=False,
        customdata=[[100 * comp.loc[c, "share_ci_lo"], 100 * comp.loc[c, "share_ci_hi"]] for c, _ in NAMES],
        hovertemplate="%{y}: %{x:.2f}% (95% CI %{customdata[0]:.2f}–%{customdata[1]:.2f})<extra></extra>"))
    base_layout(fig, height=260, title="Share of the variation in score changes")
    fig.update_layout(margin=dict(l=130))
    fig.update_xaxes(range=[0, 95], ticksuffix="%")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    plot(fig)
with right:
    pick = st.selectbox("Applicant group", ["No record", "Tax only", "Tax and deposit", "Other"], index=0, key="wa_group")
    gid = {v: k for k, v in GROUP_LABEL.items()}[pick]
    r = byg.loc[gid]
    means = [r[f"mean_{c}"] for c, _ in NAMES]
    fig = go.Figure(go.Bar(
        x=means, y=[n for _, n in NAMES], orientation="h", width=0.5,
        marker_color=[INK if v > 0 else BLIND for v in means],
        error_x=dict(type="data", symmetric=False, array=[r[f"ci_hi_{c}"] - r[f"mean_{c}"] for c, _ in NAMES],
                     arrayminus=[r[f"mean_{c}"] - r[f"ci_lo_{c}"] for c, _ in NAMES], color=INK, thickness=1, width=3),
        text=[m(f"{v:+.2f}") for v in means], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    base_layout(fig, height=230, title=m(f"Mean contribution, {pick.lower()} (PD {r.mean_dPD_pp:+.2f} pp on average)"))
    fig.update_layout(margin=dict(l=130))
    fig.update_xaxes(range=[-0.42, 0.62], tickformat="+.1f", title_text="Log-odds")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    plot(fig)

# ---------------------------------------------------------------- 4. presence vs content
chapter("Having a record is a signal in itself, but what the record says matters more")
lead("The traditional model under-predicts applicants with no record and over-predicts those with deposit records. "
     "The model with alternative data prices both groups correctly.")
chart_col, side, _ = st.columns([0.5, 0.3, 0.2], gap="large")
with chart_col:
    order = ["G0", "G1", "G2", "G3"]
    labels = [GROUP_LABEL[g] for g in order]
    c = cov.loc[order]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=c.default_rate_pct, mode="markers", name=legend_name("Actual default rate"),
                             marker=dict(size=12, color=INK, symbol="diamond"),
                             error_y=dict(type="data", symmetric=False, array=c.wilson_hi_pct - c.default_rate_pct,
                                          arrayminus=c.default_rate_pct - c.wilson_lo_pct, color=INK, thickness=1, width=0),
                             hovertemplate="%{x}: actual %{y:.2f}%<extra></extra>"))
    for col, name, color, sym, dx in [("mean_pT_pct", "Traditional PD", TRAD, "circle", -0.18),
                                      ("mean_pA_pct", "PD with AD", INK_2, "circle-open", 0.18)]:
        fig.add_trace(go.Scatter(x=labels, y=c[col], mode="markers", name=legend_name(name),
                                 marker=dict(size=11, color=color, symbol=sym, line=dict(color=color, width=2)),
                                 hovertemplate=f"%{{x}}: {name.lower()} %{{y:.2f}}%<extra></extra>"))
    base_layout(fig, height=300, title="Actual default rate and mean PD by record type")
    fig.update_layout(showlegend=True, margin=dict(t=40), legend=dict(orientation="h", x=0, y=1.02, xanchor="left",
                                                                        yanchor="bottom"))
    fig.update_yaxes(range=[2.0, 4.8], ticksuffix="%")
    fig.update_xaxes(showgrid=False)
    plot(fig)
with side:
    sp = d["wa_auc_split"].set_index("statistic")
    tiles([(f"{100 * sp.loc['D2.share_W', 'estimate']:.0f}%", "Of the AUC gain from what the records say "
            "(pairs of applicants with the same record types)"),
           (f"{100 * sp.loc['D2.share_X', 'estimate']:.0f}%", "From whether records exist "
            "(pairs with different record types)")], css_class="stack")
cl = d["wa_coverage_logit"]
rows = []
for term, lab in [("G0", "No record"), ("G2", "Tax and deposit"), ("G3", "Other")]:
    cells = [lab]
    for col in ["(1)", "(2)"]:
        r = cl[(cl.column == col) & (cl.term == term)].iloc[0]
        stars = "***" if r.p < 0.01 else "**" if r.p < 0.05 else "*" if r.p < 0.10 else ""
        cells.append(m(f"{r.coef:.3f}{stars} ({r.se_hc1:.3f})"))
    rows.append(cells)
w = [cl[cl.column == col].iloc[0] for col in ["(1)", "(2)"]]
rows.append(["Joint test, all groups = 0"] + [("p < 0.001" if r.wald_groups_p < 0.001 else f"p = {r.wald_groups_p:.2f}")
                                              + f" (χ² = {r.wald_groups_chi2:.1f})" for r in w])
table(["Record type (base: tax only)", "With traditional PD", "With PD using AD"], rows, max_width="720px",
      note="Logit of default on logit(PD) and record-type indicators. A non-zero coefficient means the group defaults "
           "more or less than its PD says. HC1 standard errors in parentheses. *** p < 0.01, ** p < 0.05, * p < 0.10.")

# ---------------------------------------------------------------- 5. approvals
chapter("Better ranking means fewer defaults, or more approvals at the same risk")
fc = d["wa_fixed_curve"]; fa = d["wa_fixed_anchor"].set_index("approval_rate_percent")
bc = d["wa_budget_curve"]; ba = d["wa_budget_anchor"].set_index("risk_budget_percent")
left, right, _ = st.columns([0.4, 0.4, 0.2], gap="large")
with left:
    fig = go.Figure()
    for mod, color, dash, name in [("traditional", TRAD, "dash", "Traditional"), ("with_ad", WITH, "solid", "With AD")]:
        q = fc[fc.model == mod]
        fig.add_trace(go.Scatter(x=q.approval_rate_percent, y=q.approved_default_rate_pct, mode="lines",
                                 name=legend_name(name), line=dict(color=color, width=2, dash=dash),
                                 hovertemplate=f"{name}<br>%{{x:.2f}}% approved: %{{y:.3f}}%<extra></extra>"))
    base_layout(fig, height=280, title="Same number approved: default rate of approved")
    fig.update_layout(showlegend=True, margin=dict(t=40), legend=dict(orientation="h", x=0, y=1.02, xanchor="left",
                                                                        yanchor="bottom"))
    fig.update_xaxes(tickvals=[40, 50, 60, 70, 80], ticktext=["40%", "50%", "60%", "70%", "80%"],
                     title_text="Share of applicants approved")
    fig.update_yaxes(ticksuffix="%", title_text="Default rate of approved")
    plot(fig)
with right:
    fig = go.Figure()
    for mod, color, dash, name in [("traditional", TRAD, "dash", "Traditional"), ("with_ad", WITH, "solid", "With AD")]:
        q = bc[bc.model == mod]
        fig.add_trace(go.Scatter(x=q.risk_budget_percent, y=q.approval_rate_pct, mode="lines", name=legend_name(name),
                                 line=dict(color=color, width=2, dash=dash),
                                 hovertemplate=f"{name}<br>Budget %{{x:.2f}}%: %{{y:.1f}}% approved<extra></extra>"))
    base_layout(fig, height=280, title="Same risk budget: share approved")
    fig.update_layout(showlegend=True, margin=dict(t=40), legend=dict(orientation="h", x=0, y=1.02, xanchor="left",
                                                                        yanchor="bottom"))
    fig.update_xaxes(tickvals=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                     ticktext=["0.75%", "1.00%", "1.25%", "1.50%", "1.75%", "2.00%"], range=[0.72, 2.03],
                     title_text="Budget: mean PD of approved")
    fig.update_yaxes(ticksuffix="%", title_text="Share of applicants approved")
    plot(fig)
f50, b1_ = fa.loc[50.0], ba.loc[1.0]
tiles([
    (f"{int(f50.defaults_avoided)}", f"Fewer defaults at 50% approval (95% CI {int(round(f50.defaults_avoided_ci_lo))}–"
                                     f"{int(round(f50.defaults_avoided_ci_hi))})"),
    (f"{f50.approved_default_rate_T_pct:.2f}% → {f50.approved_default_rate_T_plus_AD_pct:.2f}%",
     "Default rate of approved at 50% approval"),
    (f"+{int(b1_.additional_approvals):,}", f"Approvals within a 1% risk budget (95% CI "
                                            f"{int(round(b1_.additional_approvals_ci_lo)):,}–{int(round(b1_.additional_approvals_ci_hi)):,})"),
    (f"{b1_.approved_default_rate_T_pct:.2f}% → {b1_.approved_default_rate_T_plus_AD_pct:.2f}%",
     m(f"Default rate of approved within 1% budget (change not significant, CI {b1_.default_rate_change_pp_ci_lo:.3f} to "
       f"{b1_.default_rate_change_pp_ci_hi:.3f} pp)")),
], css_class="pair")

# ---------------------------------------------------------------- 6. lender value
chapter("For the lender, alternative data pays off when defaults are costly")
lead("Each repaid loan earns 1 and each default loses L/G. The lender approves applicants whose PD is below "
     "1 / (1 + L/G).")
val = d["wa_value"]
chart_col, side, _ = st.columns([0.5, 0.3, 0.2], gap="large")
with chart_col:
    x = [f"{int(v)}" for v in val.loss_to_gain_ratio]
    fig = go.Figure(go.Bar(
        x=x, y=val.value_change, width=0.45, marker_color=[INK if lo > 0 else BLIND for lo in val.value_change_ci_lo],
        error_y=dict(type="data", symmetric=False, array=val.value_change_ci_hi - val.value_change,
                     arrayminus=val.value_change - val.value_change_ci_lo, color=INK_2, thickness=1.2, width=4),
        customdata=val[["value_change_ci_lo", "value_change_ci_hi"]],
        hovertemplate="L/G = %{x}: %{y:+,} (95% CI %{customdata[0]:,.0f} to %{customdata[1]:,.0f})<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=INK, width=1))
    base_layout(fig, height=300, title="Value added by alternative data, LightGBM")
    fig.update_xaxes(type="category", showgrid=False, title_text="Loss from a default relative to profit from a repaid loan (L/G)")
    fig.update_yaxes(tickformat=",", title_text="Units of one loan's profit", range=[-400, 10500])
    plot(fig)
with side:
    v30 = val.set_index("loss_to_gain_ratio").loc[30]
    tiles([(f"+{int(v30.value_change):,}", f"Value added at L/G = 30 (95% CI {int(round(v30.value_change_ci_lo)):,}–"
                                           f"{int(round(v30.value_change_ci_hi)):,})"),
           (f"{v30.break_even:.4f}", "Break-even cost of AD per scored application at L/G = 30, in units of one "
                                     "loan's profit")], css_class="stack")
rules = d["wa_rules"].set_index("rule")
RULE = {"R1": "Fixed approval rate, 50%", "R2": "Portfolio risk budget, 1%", "R3": "Economic value rule"}
rows = [[RULE[k], f"{int(r.approved_T):,} → {int(r.approved_T_plus_AD):,}", f"{int(r.defaults_T):,} → {int(r.defaults_T_plus_AD):,}",
         f"+{int(r.value_change):,}", f"{int(round(r.value_change_ci_lo)):,}–{int(round(r.value_change_ci_hi)):,}"]
        for k, r in rules.iterrows()]
table(["Approval rule", "Approved", "Defaults approved", "Value added, L/G = 30", "95% CI"], rows, max_width="820px",
      note="LightGBM, traditional → with alternative data. Paired bootstrap of the 228,999 final-sample applications with "
           "1,000 draws; the approval rules are re-run in every draw.")

# ---------------------------------------------------------------- link to the gender result
chapter("Why this matters for the gender result")
lead("Record type is a model input, so once alternative data is added the model prices it correctly. Gender is not an "
     "input: the model cannot price a group it cannot see. The Story page shows what that does to women.")
