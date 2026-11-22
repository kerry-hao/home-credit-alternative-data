"""Story page: the gender result in five steps."""

import math

import plotly.graph_objects as go
import streamlit as st

from src.ui import CHART_CONFIG, plot, chapter, load, m, tiles
from src.theme import (AWARE, BLIND, GENDER_COLOR, INK, INK_2, MEN, REFERENCE, WOMEN, base_layout, legend_name)

d = load()
metrics = d["headline_metrics"].set_index(["model", "calibration"])
shift = d["ad_approval_shift"]
calib = d["calibration_by_decile"]
rates = d["default_rate_by_gender"].set_index("gender")
gap = d["odds_gap"].set_index("item")
ad_value = d["ad_value"].set_index(["calibration", "loss_to_gain_ratio"])


def moved(calibration: str, rate: float) -> int:
    row = shift[(shift.calibration == calibration) & (shift.approval_rate_pct == rate) & (shift.gender == "Men")]
    return int(row.net_added_approvals.iloc[0])


LOG_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]


def calibration_figure(model: str, calibration: str, title: str) -> go.Figure:
    sub = calib[(calib.model == model) & (calib.calibration == calibration)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.1, 20], y=[0.1, 20], mode="lines", hoverinfo="skip", name="Predicted = actual\u00a0\u00a0\u00a0",
                             line=dict(color=REFERENCE, width=1.5, dash="dash")))
    for sex in ["Men", "Women"]:
        g = sub[sub.gender == sex].sort_values("decile")
        fig.add_trace(go.Scatter(
            x=g.mean_pd_pct, y=g.default_rate_pct, mode="markers+lines", name=legend_name(sex),
            line=dict(color=GENDER_COLOR[sex], width=1.5),
            marker=dict(size=9, color=GENDER_COLOR[sex], line=dict(color="white", width=1.5)),
            error_y=dict(type="data", symmetric=False, array=g.ci_high_pct - g.default_rate_pct,
                         arrayminus=g.default_rate_pct - g.ci_low_pct, color=GENDER_COLOR[sex], thickness=1, width=0),
            customdata=g[["decile", "defaults", "n"]],
            hovertemplate=f"<b>{sex}</b>, decile %{{customdata[0]}}<br>Predicted %{{x:.2f}}%<br>"
                          "Actual %{y:.2f}%<br>%{customdata[1]:,} defaults of %{customdata[2]:,}<extra></extra>"))
    fig.add_annotation(x=math.log10(0.13), y=math.log10(12), text="Defaults more<br>than predicted", showarrow=False,
                       xanchor="left", align="left", font=dict(size=12, color=INK_2))
    fig.add_annotation(x=math.log10(15), y=math.log10(0.15), text="Defaults less<br>than predicted", showarrow=False,
                       xanchor="right", align="right", font=dict(size=12, color=INK_2))
    base_layout(fig, height=380, title=title)
    fig.update_layout(showlegend=True, margin=dict(t=40),
                      legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                                  traceorder="reversed", font=dict(size=13)))
    labels = [f"{v:g}%" for v in LOG_TICKS]
    fig.update_xaxes(type="log", range=[-1, math.log10(20)], tickvals=LOG_TICKS, ticktext=labels,
                     title_text="Predicted PD, by tenth of applicants")
    fig.update_yaxes(type="log", range=[-1, math.log10(20)], tickvals=LOG_TICKS, ticktext=labels,
                     title_text="Actual default rate")
    return fig


# ---------------------------------------------------------------- hero
blind_ad, aware_ad = metrics.loc[("with_ad", "blind")], metrics.loc[("with_ad", "aware")]
st.markdown('<div class="kicker">Gender bias in credit scoring</div><div class="hero-title">Blind is not neutral</div>',
            unsafe_allow_html=True)
tiles([
    (f"{rates.loc['Women', 'default_rate_pct']:.2f}% vs {rates.loc['Men', 'default_rate_pct']:.2f}%",
     "Default rate, women vs men"),
    (f"{moved('blind', 50.0):,}", "Approvals moved from women to men by alternative data"),
    (f"{math.exp(blind_ad.female_coef):.2f}×", "Women's default odds vs men's at the same PD"),
    ("~90%", "Of the bias removed by the fix"),
])
st.markdown(
    '<div class="terms">'
    '<div class="t"><b>Traditional model</b><br>LightGBM on application and credit-bureau data</div>'
    '<div class="t"><b>Alternative data (AD)</b><br>Tax, deposit and debit-card records added</div>'
    '<div class="t"><b>PD</b><br>The model\'s predicted probability of default</div>'
    "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- 1. puzzle
chapter("Adding alternative data moves approvals from women to men")
rate = st.slider("Share of applicants approved", min_value=40.0, max_value=80.0, value=50.0, step=0.25,
                 format="%.2f%%")
left, right, _ = st.columns([0.36, 0.44, 0.2], gap="large")
with left:
    fig = go.Figure(go.Bar(
        x=[rates.loc[s, "default_rate_pct"] for s in ["Men", "Women"]], y=["Men", "Women"], orientation="h",
        marker_color=[MEN, WOMEN], width=0.5, text=[f"{rates.loc[s, 'default_rate_pct']:.2f}%" for s in ["Men", "Women"]],
        textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    base_layout(fig, height=220, title="Actual default rate")
    fig.update_xaxes(range=[0, 5], ticksuffix="%", title_text="Default rate")
    fig.update_yaxes(showgrid=False)
    plot(fig)
with right:
    sub = shift[(shift.calibration == "blind") & (shift.approval_rate_pct == rate)].set_index("gender")
    vals = [sub.loc[s, "delta_approval_rate_pp"] for s in ["Men", "Women"]]
    fig = go.Figure(go.Bar(
        x=vals, y=["Men", "Women"], orientation="h", marker_color=[MEN, WOMEN], width=0.5,
        text=[m(f"{v:+.2f} pp") for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.2f} pp<extra></extra>"))
    base_layout(fig, height=220, title=f"Change in approval rate: {moved('blind', rate):,} approvals move to men")
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_xaxes(range=[-4, 4], tickvals=[-4, -2, 0, 2, 4], ticktext=["\u22124 pp", "\u22122 pp", "0", "+2 pp", "+4 pp"],
                     title_text="Traditional model → with alternative data")
    fig.update_yaxes(showgrid=False)
    plot(fig)

# ---------------------------------------------------------------- 2. diagnosis
chapter("At the same PD, women default less than men")
choice = st.segmented_control("Model", ["Traditional model", "With alternative data"],
                              default="With alternative data", key="model_2") or "With alternative data"
model = "traditional" if choice == "Traditional model" else "with_ad"
coef = metrics.loc[(model, "blind"), "female_coef"]
chart_col, side, _ = st.columns([0.5, 0.3, 0.2], gap="large")
with side:
    tiles([(f"{math.exp(coef):.2f}×", "Women's default odds vs men's at the same PD")], css_class="stack")
with chart_col:
    plot(calibration_figure(model, "blind", "Predicted vs actual default rate"))

# ---------------------------------------------------------------- 3. why
chapter("Why: a model that cannot see gender gives both groups the average")
left, right, _ = st.columns([0.4, 0.4, 0.2], gap="large")
with left:
    fig = go.Figure(go.Bar(x=[0, 1], y=[2.0, 3.0], width=0.6, marker_color=[WOMEN, MEN],
                           text=["2.00%", "3.00%"], textposition="inside", insidetextanchor="middle", textangle=0,
                           textfont=dict(color="white", size=13), hoverinfo="skip"))
    fig.add_shape(type="line", x0=-0.6, x1=1.6, y0=2.38, y1=2.38, line=dict(color=INK, width=2, dash="dash"))
    fig.add_annotation(x=-0.58, y=2.38, text="Model PD: <b>2.38%</b>", showarrow=False, xanchor="left",
                       yanchor="bottom", yshift=2, font=dict(size=12, color=INK))
    fig.add_annotation(x=0, y=2.19, text="+0.38 pp", showarrow=False, font=dict(size=12, color=INK))
    fig.add_annotation(x=1, y=2.69, text="\u22120.62 pp", showarrow=False, font=dict(size=12, color="white"))
    base_layout(fig, height=300, title="Example: identical-looking applicants")
    fig.update_yaxes(range=[0, 3.4], ticksuffix="%", title_text="Actual default rate")
    fig.update_xaxes(showgrid=False, tickvals=[0, 1], ticktext=["Women<br>62%", "Men<br>38%"], range=[-0.6, 1.6])
    plot(fig)
with right:
    items = ["Actual", "Traditional model", "With alternative data"]
    vals = [gap.loc[i, "odds_gap_pct"] for i in items]
    fig = go.Figure(go.Bar(x=vals, y=["Actual", "Traditional<br>model sees", "With alternative<br>data sees"],
                           orientation="h", width=0.5, marker_color=[INK, BLIND, BLIND],
                           text=[m(f"{v:.0f}%") for v in vals], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x:.1f}%<extra></extra>"))
    base_layout(fig, height=300, title="Women's default odds relative to men's")
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_xaxes(range=[-42, 2], ticksuffix="%", title_text="Gap in default odds, women vs men")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    plot(fig)

# ---------------------------------------------------------------- 4. fix
chapter("The fix: use gender only when turning scores into PDs")
choice = st.segmented_control("PD conversion", ["Gender-blind", "Gender-aware"], default="Gender-aware",
                              key="cal_4") or "Gender-aware"
chart_col, side, _ = st.columns([0.5, 0.3, 0.2], gap="large")
with side:
    tiles([
        (f"{math.exp(blind_ad.female_coef):.2f}× → {math.exp(aware_ad.female_coef):.2f}×",
         "Women's default odds vs men's at the same PD"),
        (f"{blind_ad.auc:.4f} → {aware_ad.auc:.4f}", "AUC"),
        (f"{int(blind_ad.approved_defaults_50):,} → {int(aware_ad.approved_defaults_50):,}",
         "Defaults approved when half of applicants are approved"),
        (f"+{int(aware_ad.value_lg30 - blind_ad.value_lg30):,}", "Lender value when a default costs 30× a good loan's profit"),
    ], css_class="stack")
with chart_col:
    cal = "blind" if choice == "Gender-blind" else "aware"
    plot(calibration_figure("with_ad", cal, f"Predicted vs actual default rate, with alternative data"))

# ---------------------------------------------------------------- 5. payoff
chapter("With the fix, alternative data moves fewer approvals and adds more value")
chart_col, side, _ = st.columns([0.5, 0.3, 0.2], gap="large")
fig = go.Figure()
for cal, color, dash, label in [("blind", BLIND, "dash", "Gender-blind"), ("aware", AWARE, "solid", "Gender-aware")]:
    g = shift[(shift.calibration == cal) & (shift.gender == "All")].sort_values("approval_rate_pct")
    fig.add_trace(go.Scatter(x=g.approval_rate_pct, y=g.female_minus_male_delta_approval_rate_pp, mode="lines",
                             line=dict(color=color, width=2.2, dash=dash), name=legend_name(label),
                             hovertemplate=f"{label}<br>%{{x:.2f}}% approved: %{{y:.2f}} pp<extra></extra>"))
fig.add_hline(y=0, line=dict(color=INK, width=1))
base_layout(fig, height=320, title="Approval shift caused by alternative data, women minus men")
fig.update_layout(showlegend=True, margin=dict(t=40), legend=dict(orientation="h", x=0, y=1.02, xanchor="left",
                                                                    yanchor="bottom"))
fig.update_xaxes(range=[39, 81], tickvals=[40, 50, 60, 70, 80], ticktext=["40%", "50%", "60%", "70%", "80%"], title_text="Share of applicants approved")
fig.update_yaxes(tickvals=[0, -1, -2, -3, -4], ticktext=["0 pp", "\u22121 pp", "\u22122 pp", "\u22123 pp", "\u22124 pp"],
                 title_text="Women's change minus men's")
with chart_col:
    plot(fig)
with side:
    tiles([
        (f"{moved('blind', 50.0):,} → {moved('aware', 50.0):,}", "Approvals moved from women to men at 50% approval"),
        (f"+{int(ad_value.loc[('blind', 30), 'ad_increment_realised_value']):,} → "
         f"+{int(ad_value.loc[('aware', 30), 'ad_increment_realised_value']):,}", "Lender value added by alternative data"),
    ], css_class="stack")
