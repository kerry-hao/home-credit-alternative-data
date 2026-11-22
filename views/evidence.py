"""Evidence page: the tests behind each step of the story."""

import math

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.theme import BLIND, GENDER_COLOR, INK, INK_2, MEN, REFERENCE, WOMEN, base_layout, legend_name
from src.ui import CHART_CONFIG, plot, chapter, lead, load, m, page_title, table, tiles

Z = 1.959964
d = load()

page_title("Evidence", "The tests behind each step")
choice = st.segmented_control("Model", ["Traditional model", "With alternative data"],
                              default="With alternative data", key="ev_model") or "With alternative data"
model = "traditional" if choice == "Traditional model" else "with_ad"
other = "with_ad" if model == "traditional" else "traditional"

# ---------------------------------------------------------------- 1. composition
chapter("Within every credit-history group, women lose approvals and men gain", first=True)
lead("Women and men have almost the same credit-history profile. The shift appears inside each group, "
     "so it does not come from who applies.")
hist = d["ev_history_shift"]
order = hist.drop_duplicates("bureau_history_group").sort_values("bureau_history_order")
labels = {"0_to_0_5": "0–0.5 years", "0_5_to_1": "0.5–1 years", "1_to_2": "1–2 years", "2_to_3": "2–3 years",
          "3_to_4": "3–4 years", "4_to_5": "4–5 years", "gt_5": "More than 5 years"}
ylab = [labels.get(g, g) for g in order.bureau_history_group]
chart_col, side, _ = st.columns([0.5, 0.3, 0.2], gap="large")
with chart_col:
    fig = go.Figure()
    w = hist[hist.gender == "Women"].set_index("bureau_history_group").loc[order.bureau_history_group]
    mn = hist[hist.gender == "Men"].set_index("bureau_history_group").loc[order.bureau_history_group]
    for i, y in enumerate(ylab):
        fig.add_trace(go.Scatter(x=[w.delta_approval_rate_pp.iloc[i], mn.delta_approval_rate_pp.iloc[i]], y=[y, y],
                                 mode="lines", line=dict(color="#D5D9DE", width=3), hoverinfo="skip", showlegend=False))
    for sex, frame in [("Women", w), ("Men", mn)]:
        fig.add_trace(go.Scatter(x=frame.delta_approval_rate_pp, y=ylab, mode="markers", name=legend_name(sex),
                                 marker=dict(size=12, color=GENDER_COLOR[sex], line=dict(color="white", width=1.5)),
                                 customdata=100 * frame.share_of_gender_population,
                                 hovertemplate=f"{sex}, %{{y}}<br>%{{x:+.2f}} pp<br>%{{customdata:.1f}}% of {sex.lower()}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    base_layout(fig, height=320, title="Change in approval rate after adding alternative data, 50% approved")
    fig.update_layout(showlegend=True, margin=dict(t=40, l=150),
                      legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom"))
    fig.update_xaxes(range=[-3, 5], tickvals=[-2, 0, 2, 4], ticktext=["−2 pp", "0", "+2 pp", "+4 pp"])
    fig.update_yaxes(autorange="reversed", showgrid=False, title_text="Credit-history length")
    plot(fig)
with side:
    cells = d["ev_cells"].iloc[0]
    tiles([(f"{cells.men_gain_more} of {cells.cells}", "Groups by credit history and number of AD record types "
            "where men gain more")], css_class="stack")

# ---------------------------------------------------------------- 2. stable gap
chapter("At the same PD, women's default odds are about two-thirds of men's in every specification")
lead("Logit of default on the model's log-odds and a female indicator. If the model were accurate for both "
     "genders, the female coefficient would be zero and the odds ratio one.")
spec = d["ev_female_coef_specs"]
sub = spec[spec.model == model].iloc[::-1]
fig = go.Figure(go.Scatter(
    x=sub.coef.map(math.exp), y=sub.spec, mode="markers",
    marker=dict(size=11, color=INK),
    error_x=dict(type="data", symmetric=False, array=(sub.coef + Z * sub.se).map(math.exp) - sub.coef.map(math.exp),
                 arrayminus=sub.coef.map(math.exp) - (sub.coef - Z * sub.se).map(math.exp), color=INK, thickness=1.5, width=0),
    customdata=sub[["coef", "se", "n"]],
    hovertemplate="%{y}<br>Odds ratio %{x:.3f}<br>Coefficient %{customdata[0]:.3f} (%{customdata[1]:.3f})"
                  "<br>N = %{customdata[2]:,}<extra></extra>"))
fig.add_vline(x=1, line=dict(color=REFERENCE, width=1.5, dash="dash"))
fig.add_annotation(x=1, y=1.0, yref="paper", text="No gap", showarrow=False, xanchor="right", xshift=-6,
                   yanchor="bottom", font=dict(size=12, color=INK_2))
base_layout(fig, height=360, title=f"Women's default odds vs men's at the same PD, {choice.lower()}")
fig.update_layout(margin=dict(l=240))
fig.update_xaxes(range=[0.55, 1.05], tickformat=".2f", title_text="Odds ratio, exp(female coefficient), with 95% CI")
fig.update_yaxes(showgrid=False)
narrow, _ = st.columns([0.72, 0.28])
with narrow:
    plot(fig)

# ---------------------------------------------------------------- 3. four tests
chapter("Four tests point to a uniform gender discount the model cannot learn")
lead("Two explanations fit an over-estimate of women's risk. In the first, women look like men on every input and "
     "their lower risk sits in information the data do not record. In the second, some inputs, such as past "
     "delinquency, carry less risk for women than the model assumes.")
shape = d["ev_shape"].set_index(["model", "symbol"])
ct = d["ev_component_tests"].set_index(["model", "row"]).value
shap = d["ev_shap_families"]
gap = d["odds_gap"].set_index("item")
loc = d["ev_location"]
est = loc[loc.status == "estimated"]
push_up = shap[(shap.model == "traditional") & (shap.delta > 0)].delta.sum()
e_t, e_a = est[est.model == "traditional"], est[est.model == "with_ad"]
table(
    ["Test", "If women look like men", "If an input means less for women", "Result", "Points to"],
    [
        ["What the model's scores show", "Small gap in scores", "Some input families push women up",
         m(f"Score gap {gap.loc['Traditional model', 'log_odds_gap']:.3f} vs actual {gap.loc['Actual', 'log_odds_gap']:.3f}; "
           f"borrowing records push women up by {push_up:.3f}"), "Both possible"],
        ["Shape along the PD scale", "Parallel lines: h = 0", "Gap widens with risk: h &lt; 0",
         f"h = {shape.loc[('traditional', 'h'), 'coef']:.3f} and {shape.loc[('with_ad', 'h'), 'coef']:.3f}; never negative",
         "Look-alike"],
        ["Location across groups", "Similar gap everywhere", "Gap where inputs make women look risky",
         f"Negative in all {len(e_t)} groups; odds ratio {math.exp(e_t.g.min()):.2f}–{math.exp(e_t.g.max()):.2f}",
         "Look-alike"],
        ["Meaning of each input family", "h = 0 for every family", "h &lt; 0 where women are pushed up",
         f"No h significant at 5%; joint p = {ct.loc[('traditional', 'wald_all_h_zero_p')]:.2f} and "
         f"{ct.loc[('with_ad', 'wald_all_h_zero_p')]:.2f}", "Look-alike"],
    ],
    align=["left"] * 5,
    note="Results for the traditional model, then the model with alternative data. h is the female × score "
         "interaction in the logit of default. Scores and gaps are in log-odds.")

st.markdown('<div class="sub">What the model\'s scores show</div>', unsafe_allow_html=True)
lead("Each score splits exactly into a common start plus one contribution per input (SHAP). Averaging by gender "
     "shows which input families move women's scores relative to men's.")
chart_col, side, _ = st.columns([0.5, 0.3, 0.2], gap="large")
with chart_col:
    sh = shap[shap.model == model].sort_values("family_order", ascending=False)
    fig = go.Figure(go.Bar(
        x=sh.delta, y=sh.family, orientation="h", width=0.6,
        marker_color=[INK if v < 0 else BLIND for v in sh.delta],
        error_x=dict(type="data", array=Z * sh.se, color=INK_2, thickness=1.2, width=3),
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    base_layout(fig, height=360 if model == "with_ad" else 290,
                title="Mean contribution, women minus men (log-odds)")
    fig.update_layout(margin=dict(l=150))
    fig.update_xaxes(range=[-0.22, 0.08], tickformat="+.2f", title_text="Lowers women's score ← → raises it")
    fig.update_yaxes(showgrid=False)
    plot(fig)
with side:
    item = "Traditional model" if model == "traditional" else "With alternative data"
    tiles([(m(f"{gap.loc[item, 'odds_gap_pct']:.0f}%"), "Gap in default odds the model sees"),
           (m(f"{gap.loc['Actual', 'odds_gap_pct']:.0f}%"), "Actual gap in default odds")], css_class="stack")

st.markdown('<div class="sub">Location across groups</div>', unsafe_allow_html=True)
lead("The calibration logit is re-estimated inside each group. Groups are formed on all applicants together.")
tests = d["ev_location_tests"]
groupings = loc.drop_duplicates("grouping_id")[["grouping_id", "grouping"]]
default = groupings.grouping.tolist().index("Max days past due, last 24 months") \
    if "Max days past due, last 24 months" in groupings.grouping.tolist() else 0
pick = st.selectbox("Grouping", groupings.grouping.tolist(), index=default)
gid = groupings.set_index("grouping").loc[pick, "grouping_id"]
g = loc[(loc.model == model) & (loc.grouping_id == gid)].sort_values("group_order")
ge = g[g.status == "estimated"]
p = tests[(tests.model == model) & (tests.grouping_id == gid)].p.iloc[0]
base_g = spec[(spec.model == model) & (spec.spec == "Baseline")].coef.iloc[0]
fig = go.Figure(go.Scatter(
    x=ge.g.map(math.exp), y=ge.group.astype(str), mode="markers", marker=dict(size=11, color=INK),
    error_x=dict(type="data", symmetric=False, array=(ge.g + Z * ge.se).map(math.exp) - ge.g.map(math.exp),
                 arrayminus=ge.g.map(math.exp) - (ge.g - Z * ge.se).map(math.exp), color=INK, thickness=1.5, width=0),
    customdata=ge[["n", "defaults_women", "defaults_men"]],
    hovertemplate="%{y}<br>Odds ratio %{x:.2f}<br>N = %{customdata[0]:,}; defaults: women %{customdata[1]:,}, "
                  "men %{customdata[2]:,}<extra></extra>"))
fig.add_vline(x=1, line=dict(color=REFERENCE, width=1.5, dash="dash"))
fig.add_vline(x=math.exp(base_g), line=dict(color=INK, width=1, dash="dot"))
fig.add_annotation(x=math.exp(base_g), y=1.0, yref="paper", text="All applicants", showarrow=False, xanchor="left",
                   xshift=4, yanchor="bottom", font=dict(size=12, color=INK_2))
base_layout(fig, height=max(240, 60 + 38 * len(ge)),
            title=f"Women's default odds vs men's at the same PD, by {pick.lower()} (equal across groups: p = {p:.3f})")
fig.update_xaxes(range=[0.2, 1.4], tickformat=".1f", title_text="Odds ratio with 95% CI")
fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
narrow, _ = st.columns([0.72, 0.28])
with narrow:
    plot(fig)
skipped = g[g.status != "estimated"].group.astype(str).tolist()
if skipped:
    st.markdown(f'<p class="tbl-note">Not estimated (fewer than 30 defaults among women or men): {", ".join(skipped)}.</p>',
                unsafe_allow_html=True)

st.markdown('<div class="sub">Meaning of each input family</div>', unsafe_allow_html=True)
lead("Each family's part of the score enters the logit of default with its own coefficient for men and for women. "
     "A value of one means a push from that family is fully borne out in actual defaults.")
comp = d["ev_component_logit"]
cm = comp[comp.model == model].iloc[::-1]
fig = go.Figure()
for sex, col, se, shift in [("Men", "c", "c_se", -0.17), ("Women", "c_plus_h", "c_plus_h_se", 0.17)]:
    fig.add_trace(go.Scatter(
        x=cm[col], y=[i + shift for i in range(len(cm))], mode="markers", name=legend_name(sex),
        marker=dict(size=10, color=GENDER_COLOR[sex], line=dict(color="white", width=1)),
        error_x=dict(type="data", array=Z * cm[se], color=GENDER_COLOR[sex], thickness=1.5, width=0),
        customdata=cm.family, hovertemplate=f"{sex}, %{{customdata}}<br>%{{x:.3f}}<extra></extra>"))
fig.add_vline(x=1, line=dict(color=REFERENCE, width=1.5, dash="dash"))
gf = ct.loc[(model, "g_female")]
base_layout(fig, height=110 + 36 * len(cm),
            title=m(f"Coefficient on each family's score (joint test that men = women: p = "
                    f"{ct.loc[(model, 'wald_all_h_zero_p')]:.2f}; female coefficient {gf:.3f})"))
fig.update_layout(showlegend=True, margin=dict(t=40, l=150), legend=dict(orientation="h", x=0, y=1.02, xanchor="left",
                                                                           yanchor="bottom", traceorder="reversed"))
fig.update_xaxes(range=[-0.6, 3.6] if model == "with_ad" else [0.4, 1.6], title_text="Coefficient with 95% CI")
fig.update_yaxes(tickvals=list(range(len(cm))), ticktext=cm.family.tolist(), showgrid=False)
narrow, _ = st.columns([0.72, 0.28])
with narrow:
    plot(fig)

# ---------------------------------------------------------------- 4. fix
chapter("The fix is fitted on a separate sample and improves every lending rule")
lead("The calibrator is a logit of default on the model's log-odds and a female indicator, with an optional "
     "interaction. It is fitted on the calibration sample of 114,500 applications, frozen, and then applied once "
     "to the final sample of 228,999.")
cal = d["ev_calibrators"]
rows = []
for mod, name in [("traditional", "Traditional model"), ("with_ad", "With alternative data")]:
    rows.append(name)
    for sym, lab in [("alpha", "Constant, α"), ("beta", "logit(PD), β"), ("gamma", "Female, γ"),
                     ("delta", "Female × logit(PD), δ")]:
        cells = [lab]
        for ver in ["A1", "A2"]:
            r = cal[(cal.model == mod) & (cal.version == ver) & (cal.symbol == sym)]
            cells.append(m(f"{r.coef.iloc[0]:.3f} ({r.se.iloc[0]:.3f})") if len(r) else "")
        rows.append(cells)
table(["", "Level", "Level and slope"], rows, max_width="620px", note="Calibration sample: 114,500 applications, 3,600 defaults. "
      "Standard errors in parentheses.")

lend = d["ev_lending"]
rows = []
for mod, name in [("traditional", "Traditional model"), ("with_ad", "With alternative data")]:
    rows.append(name)
    for _, r in lend[lend.model == mod].iterrows():
        rows.append([r.version, m(f"{r.female_coef:.3f}"), f"{r.auc:.4f}",
                     f"{r.approval_women_50:.1f}% / {r.approval_men_50:.1f}%", f"{int(r.approved_defaults_50):,}",
                     f"{int(r.approved_1pct):,}", f"{int(r.value_lg30):,}"])
table(["PD conversion", "Female coef.", "AUC", "Approval rate, women / men", "Defaults approved",
       "Approvals within 1% risk budget", "Lender value, L/G = 30"], rows,
      note="Final sample. Female coefficient from the calibration logit with the PD of each version. Approval rates "
           "and defaults approved at a 50% approval rate. The 1% risk budget approves the lowest-PD applicants until "
           "their mean PD reaches 1%. Lender value: +1 per repaid loan and −30 per default among approved applicants.")

# ---------------------------------------------------------------- 5. switched
chapter("After the fix, the approvals that alternative data moves to men go to safer men")
sw = d["ev_switched"]
fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                    subplot_titles=("Gender-blind PDs", "Gender-aware PDs"))
counts = {}
for j, cal_ in enumerate(["blind", "aware"], start=1):
    s_ = sw[sw.calibration == cal_]
    piv = s_.pivot(index="approval_rate_pct", columns="group", values="default_rate_pct")
    counts[cal_] = int((piv["Men switched in"] > piv["Women switched out"]).sum())
    for grp, color in [("Women switched out", WOMEN), ("Men switched in", MEN)]:
        q = s_[s_.group == grp].sort_values("approval_rate_pct")
        fig.add_trace(go.Scatter(x=q.approval_rate_pct, y=q.default_rate_pct, mode="lines", name=legend_name(grp),
                                 line=dict(color=color, width=2), showlegend=j == 1,
                                 hovertemplate=f"{grp}<br>%{{x:.2f}}% approved: %{{y:.2f}}%<extra></extra>"), row=1, col=j)
base_layout(fig, height=330, title="Default rate of the applicants switched by alternative data")
fig.update_layout(showlegend=True, margin=dict(t=64), legend=dict(orientation="h", x=0, y=1.1, xanchor="left",
                                                                     yanchor="bottom"))
fig.update_annotations(font=dict(size=14, color=INK))
fig.update_xaxes(tickvals=[40, 50, 60, 70, 80], ticktext=["40%", "50%", "60%", "70%", "80%"], title_text="Share of applicants approved")
fig.update_yaxes(ticksuffix="%", row=1, col=1, title_text="Default rate")
chart_col, side, _ = st.columns([0.62, 0.24, 0.14], gap="large")
with chart_col:
    plot(fig)
with side:
    tiles([(f"{counts['blind']} → {counts['aware']}", "Of 161 approval rates where men switched in default more "
            "than women switched out")], css_class="stack")
