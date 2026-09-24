"""Recruiter-facing aggregate-results Streamlit showcase."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.showcase_charts import (
    BACKGROUND,
    allocation_figure,
    heterogeneity_figure,
    performance_figure,
    reranking_figure,
    value_figure,
)


ROOT = Path(__file__).resolve().parent
SHOWCASE_DIR = ROOT / "data/showcase"
APP_TITLE = "The Unequal Value and Cost of Alternative Data in Credit Risk Predictions"
VIEWS = [
    "Model performance",
    "Information heterogeneity",
    "Risk re-ranking",
    "Credit re-allocation",
    "Lender-side value",
]
MODEL_FAMILIES = ["Logit (baseline)", "LightGBM", "MLP neural network"]


@st.cache_data(show_spinner=False)
def load_showcase_data() -> dict[str, pd.DataFrame]:
    names = [
        "overall_model_metrics.csv",
        "overall_incremental_gains.csv",
        "subgroup_incremental_gains.csv",
        "risk_migration_5_10_20.csv",
        "fixed_approval_rate_curve.csv",
        "fixed_risk_budget_curve.csv",
        "economic_incremental_value.csv",
    ]
    return {name: pd.read_csv(SHOWCASE_DIR / name) for name in names}


def chart_and_note(figure, lines: list[str], *, heatmap: bool = False) -> None:
    """Render one compact figure, its reading note, and deliberate whitespace."""

    ratios = [0.59, 0.17, 0.24] if heatmap else [0.56, 0.19, 0.25]
    chart_column, note_column, _empty_column = st.columns(ratios, gap="small")
    with chart_column:
        st.plotly_chart(
            figure,
            width="stretch",
            config={"displayModeBar": False, "displaylogo": False, "responsive": True},
        )
    with note_column:
        paragraphs = "".join(f"<p class='reading-line'>{line}</p>" for line in lines)
        st.markdown(
            f"<div class='reading-note'><h3>How to read this chart:</h3>{paragraphs}</div>",
            unsafe_allow_html=True,
        )


def footer() -> None:
    footer_column, _empty_column = st.columns([0.75, 0.25], gap="small")
    with footer_column:
        st.markdown(
            """
            <div class="showcase-footer">
              <div>1,526,659 applications: 1,068,661 training; 114,499 tuning; 114,500 calibration; 228,999 held-out final evaluation.</div>
              <div>Models: regularized Logit (baseline), LightGBM and MLP neural network, each estimated with traditional credit data and with traditional + alternative data.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    f"""
    <style>
      .stApp {{ background: {BACKGROUND}; }}
      .block-container {{ max-width: 1220px; padding: 1.7rem 1.5rem .35rem; }}
      h1 {{ color: #27313A; font-size: 1.78rem !important; line-height: 1.16 !important;
            font-weight: 750 !important; margin: 0 0 .2rem !important; white-space: nowrap; }}
      [data-testid="stWidgetLabel"] p {{ font-weight: 650 !important; color: #27313A; }}
      [data-testid="stSegmentedControl"] button p {{ font-weight: 650 !important; }}
      [data-testid="stSegmentedControl"] button[aria-pressed="true"] p {{ font-weight: 750 !important; }}
      [data-testid="stPlotlyChart"] {{ margin-top: -.55rem; }}
      .results-prompt {{ font-size: .875rem !important; font-weight: 700; color: #27313A; margin: .08rem 0 .3rem; }}
      .st-key-results_navigation [role="radiogroup"] {{ width: 100%; }}
      .st-key-results_navigation [role="radio"] {{ flex: 1 1 0; justify-content: center; min-width: 0; padding: 4px 8px; }}
      .st-key-performance_metric [data-testid="stWidgetLabel"] {{ margin-bottom: 0; }}
      .st-key-risk_group_count [data-testid="stWidgetLabel"] p {{ white-space: nowrap; }}
      .st-key-risk_group_count [role="radio"] {{ flex: 1 1 0; justify-content: center; min-width: 0; }}
      .reading-note {{ padding: 3.55rem .15rem 0 .35rem; max-width: 14rem; }}
      .reading-note h3 {{ font-size: .94rem; line-height: 1.2; margin: 0 0 .6rem; font-weight: 700; color: #27313A; }}
      .reading-line {{ font-size: .78rem; line-height: 1.34; margin: 0 0 .58rem; color: #4F575D; }}
      .showcase-footer {{ color: #6A7075; font-size: .69rem; line-height: 1.3;
                          text-align: left; margin-top: -.55rem; padding-top: .15rem; }}
      .showcase-footer div + div {{ margin-top: .12rem; }}
      footer {{ visibility: hidden; }}
      @media (max-width: 1000px) {{
        h1 {{ white-space: normal; }}
        .reading-note {{ padding-top: .35rem; max-width: none; }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
navigation_region, _empty_navigation = st.columns([0.75, 0.25], gap="small")
with navigation_region:
    with st.container(key="results_navigation"):
        st.markdown('<p class="results-prompt">Choose a result section:</p>', unsafe_allow_html=True)
        view = st.segmented_control(
            "Results section",
            VIEWS,
            default="Model performance",
            label_visibility="collapsed",
            width="stretch",
        )
data = load_showcase_data()

control_region, _empty_controls = st.columns([0.75, 0.25], gap="small")
with control_region:
    if view == "Model performance":
        metric = st.radio(
            "Choose a performance metric to compare the three model families:",
            ["ROC AUC", "Average Precision", "Log Loss", "Brier Score"],
            horizontal=True,
            index=0,
            key="performance_metric",
        )
        figure = performance_figure(data["overall_model_metrics.csv"], metric)
        if metric in {"ROC AUC", "Average Precision"}:
            note = [
                "Each line compares the same model using traditional credit data and traditional + alternative data.",
                "A larger rightward shift indicates stronger predictive performance.",
            ]
        else:
            note = [
                "Each line compares the same model using traditional credit data and traditional + alternative data.",
                "A larger leftward shift indicates a larger reduction in prediction error.",
            ]
    elif view == "Information heterogeneity":
        first, second, third, _spacer = st.columns([1.55, 0.87, 1.15, 0.13], gap="small")
        dimension = first.selectbox(
            "Choose an information dimension:",
            [
                "Number of credit accounts",
                "Credit-history length (years)",
                "Alternative-data richness (number of observed sections)",
            ],
            index=0,
        )
        algorithm = second.selectbox("Choose a model family:", MODEL_FAMILIES, index=1)
        metric = third.selectbox(
            "Choose a performance metric:",
            ["ROC AUC gain", "Average Precision gain", "Log-loss improvement", "Brier improvement"],
        )
        figure = heterogeneity_figure(data["subgroup_incremental_gains.csv"], dimension, algorithm, metric)
        note = [
            "Credit-account count and credit-history length represent the thickness of traditional borrower information.",
            "Lower values indicate fewer accounts or shorter credit histories; alternative-data richness counts observed data sections.",
            "Each bar shows the within-group performance improvement after adding alternative data.",
        ]
    elif view == "Risk re-ranking":
        first, second, _spacer = st.columns([0.8, 0.7, 2.0], gap="small")
        algorithm = first.selectbox("Choose a model family:", MODEL_FAMILIES, index=1)
        n_bins = second.segmented_control(
            "Choose the number of risk groups:",
            [5, 10, 20],
            default=10,
            width="stretch",
            key="risk_group_count",
        )
        figure = reranking_figure(data["risk_migration_5_10_20.csv"], algorithm, int(n_bins))
        note = [
            "Rows show risk groups produced using traditional credit data.",
            "Columns show the new risk groups after alternative data are added.",
            "Each cell is the share of borrowers moving from the original row group to the new column group; each row sums to 100%.",
        ]
    elif view == "Credit re-allocation":
        first, second, _spacer = st.columns([1.35, 0.8, 1.35], gap="small")
        policy = first.segmented_control(
            "Choose a lending decision rule:",
            ["Same approval rate", "Same portfolio risk"],
            default="Same approval rate",
            width="stretch",
        )
        algorithm = second.selectbox("Choose a model family:", MODEL_FAMILIES, index=1)
        figure = allocation_figure(
            data["fixed_approval_rate_curve.csv"],
            data["fixed_risk_budget_curve.csv"],
            policy,
            algorithm,
        )
        if policy == "Same approval rate":
            note = [
                "Both models approve the same proportion of applicants.",
                "A lower curve means fewer realised defaults among approved loans.",
            ]
        else:
            note = [
                "Both models operate under the same portfolio-risk limit.",
                "A higher curve means that more applicants can be approved.",
            ]
    else:
        first, second = st.columns([0.8, 2.7], gap="small")
        algorithm = first.selectbox("Choose a model family:", MODEL_FAMILIES, index=1)
        basis = second.segmented_control(
            "Choose how lender value is measured:",
            [
                "Expected value based on model probabilities",
                "Realised value based on observed outcomes",
            ],
            default="Realised value based on observed outcomes",
            width="stretch",
        )
        figure = value_figure(data["economic_incremental_value.csv"], algorithm, basis)
        note = [
            "The horizontal axis varies the assumed loss from one default relative to the gain from one non-default.",
            "Values are normalized scenario units rather than currency.",
        ]
        if basis == "Realised value based on observed outcomes":
            note.append(
                "A positive break-even value is the maximum alternative-data cost that the realised gain could support."
            )

chart_and_note(figure, note, heatmap=view == "Risk re-ranking")
footer()
