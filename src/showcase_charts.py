"""Plotly figures for the five Task 13 showcase views."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go


T_COLOR = "#2F5D7E"
TAD_COLOR = "#E69F00"
POSITIVE = "#009E73"
NEGATIVE = "#D55E00"
BACKGROUND = "#FFFDF8"
TEXT = "#27313A"
GRID = "#EAE4DA"
NEUTRAL = "#A9A39A"

TRADITIONAL_LABEL = "Traditional credit data"
ALTERNATIVE_LABEL = "Traditional + alternative data"
FAMILY_LABELS = {
    "logit": "Logit (baseline)",
    "lightgbm": "LightGBM",
    "mlp": "MLP neural network",
}
FAMILY_IDS = {
    **{label: family for family, label in FAMILY_LABELS.items()},
    "Logit": "logit",
    "MLP": "mlp",
}

DIMENSIONS = {
    "Number of credit accounts": "account_count",
    "Credit-history length (years)": "bureau_history",
    "Alternative-data richness (number of observed sections)": "ad_richness",
    # Backwards-compatible internal aliases; the app does not display these.
    "Account count": "account_count",
    "Bureau history": "bureau_history",
    "AD richness": "ad_richness",
}

GROUP_LABELS = {
    "account_count": {
        "0": "0 accounts",
        "1": "1 account",
        "2": "2 accounts",
        "3_plus": "3+ accounts",
    },
    "bureau_history": {
        "0_to_0_5": "0–0.5 years",
        "0_5_to_1": "0.5–1 year",
        "1_to_2": "1–2 years",
        "2_to_3": "2–3 years",
        "3_to_4": "3–4 years",
        "4_to_5": "4–5 years",
        "gt_5": "More than 5 years",
    },
    "ad_richness": {
        "0": "0 sections",
        "1": "1 section",
        "2": "2 sections",
        "3": "3 sections",
    },
}


def _family_id(label: str) -> str:
    """Map a public family label to its frozen internal identifier."""

    return FAMILY_IDS[label]


def _linear_range(
    values: np.ndarray | pd.Series | list[float],
    *,
    padding_ratio: float = 0.05,
    zero_baseline: bool = False,
) -> list[float]:
    """Return a finite data-fitted range with modest endpoint padding."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot derive an axis range without finite values")
    low = float(finite.min())
    high = float(finite.max())
    if zero_baseline and low >= 0:
        return [0.0, high * (1 + max(padding_ratio, 0.05))]
    if zero_baseline and high <= 0:
        return [low * (1 + max(padding_ratio, 0.05)), 0.0]
    if zero_baseline:
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low
    if span <= 0:
        span = max(abs(low), 1.0) * 0.1
    padding = span * padding_ratio
    return [low - padding, high + padding]


def _log_range(values: np.ndarray | pd.Series | list[float], padding_ratio: float = 0.04) -> list[float]:
    """Return Plotly log-axis bounds padded multiplicatively in log space."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        raise ValueError("Cannot derive a logarithmic axis range without positive values")
    low = math.log10(float(finite.min()))
    high = math.log10(float(finite.max()))
    span = high - low
    padding = (span if span > 0 else 0.1) * padding_ratio
    return [low - padding, high + padding]


def _finish(
    figure: go.Figure,
    title: str,
    *,
    height: int = 340,
    margins: dict[str, int] | None = None,
    show_legend: bool = True,
    legend_spacing: bool = False,
) -> go.Figure:
    resolved_margins = margins or {"l": 74, "r": 22, "t": 76, "b": 58}
    if legend_spacing:
        resolved_margins = {**resolved_margins, "t": resolved_margins["t"] + 8}
    figure.update_layout(
        title={"text": f"<b>{title}</b>", "x": 0.0, "xanchor": "left", "font": {"size": 18}},
        height=height,
        margin=resolved_margins,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font={"family": "Inter, Source Sans, Arial, sans-serif", "color": TEXT, "size": 12},
        hoverlabel={"bgcolor": BACKGROUND, "font_color": TEXT},
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "y": 1.08,
            "yanchor": "bottom",
            "x": 0,
            "xanchor": "left",
            "font": {"size": 11},
        },
    )
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, nticks=6, title_font={"size": 12})
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, title_font={"size": 12})
    return figure


def performance_figure(metrics: pd.DataFrame, metric: str) -> go.Figure:
    options = {
        "ROC AUC": ("roc_auc", "ROC AUC", False),
        "Average Precision": ("average_precision", "Average Precision", False),
        "Log Loss": ("log_loss", "Log Loss · lower is better", True),
        "Brier Score": ("brier_score", "Brier Score · lower is better", True),
    }
    column, axis_title, lower_better = options[metric]
    figure = go.Figure()
    displayed_values: list[float] = []
    for family in ("logit", "lightgbm", "mlp"):
        subset = metrics.loc[metrics.algorithm_family.eq(family)].set_index("information_set")
        t_value = float(subset.loc["T", column])
        tad_value = float(subset.loc["T_plus_AD", column])
        displayed_values.extend([t_value, tad_value])
        improvement = t_value - tad_value if lower_better else tad_value - t_value
        label = FAMILY_LABELS[family]
        figure.add_trace(go.Scatter(
            x=[t_value, tad_value],
            y=[label, label],
            mode="lines",
            line={"color": NEUTRAL, "width": 5},
            hoverinfo="skip",
            showlegend=False,
        ))
        for value, info, color in (
            (t_value, TRADITIONAL_LABEL, T_COLOR),
            (tad_value, ALTERNATIVE_LABEL, TAD_COLOR),
        ):
            figure.add_trace(go.Scatter(
                x=[value],
                y=[label],
                mode="markers",
                name=info,
                legendgroup=info,
                showlegend=family == "logit",
                marker={"size": 13, "color": color, "line": {"color": BACKGROUND, "width": 1}},
                customdata=[[info, improvement]],
                hovertemplate=(
                    f"{label}<br>%{{customdata[0]}}<br>{metric}: %{{x:.6f}}"
                    "<br>Improvement: %{customdata[1]:+.6f}<extra></extra>"
                ),
            ))
        figure.add_annotation(
            x=tad_value,
            y=label,
            text=f"Δ {improvement:+.4f}",
            showarrow=False,
            xanchor="center",
            yshift=-15,
            font={"size": 11, "color": POSITIVE if improvement >= 0 else NEGATIVE},
        )
    figure.update_xaxes(title=axis_title, range=_linear_range(displayed_values))
    figure.update_yaxes(
        title=None,
        categoryorder="array",
        categoryarray=[FAMILY_LABELS["mlp"], FAMILY_LABELS["lightgbm"], FAMILY_LABELS["logit"]],
    )
    return _finish(
        figure,
        "Model Performance with and without Alternative Data",
        legend_spacing=True,
    )


def heterogeneity_figure(gains: pd.DataFrame, dimension: str, algorithm: str, metric: str) -> go.Figure:
    metrics = {
        "ROC AUC gain": ("roc_auc_gain", "ROC AUC improvement"),
        "Average Precision gain": ("average_precision_gain", "Average Precision improvement"),
        "AP gain": ("average_precision_gain", "Average Precision improvement"),
        "Log-loss improvement": ("log_loss_gain", "Log-loss reduction"),
        "Brier improvement": ("brier_gain", "Brier-score reduction"),
    }
    dimension_id = DIMENSIONS[dimension]
    family = _family_id(algorithm)
    column, axis_title = metrics[metric]
    subset = gains.loc[
        gains.group_dimension.eq(dimension_id) & gains.algorithm_family.eq(family)
    ].sort_values("group_order", ascending=False, kind="stable")
    labels = [GROUP_LABELS[dimension_id][str(label)] for label in subset.group_label]
    values = subset[column].to_numpy(float)
    colors = [TAD_COLOR if value >= 0 else NEGATIVE for value in values]
    figure = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        width=0.48,
        marker={"color": colors},
        customdata=np.column_stack([subset.sample_count.to_numpy(), values]),
        hovertemplate=(
            "%{y}<br>Applications: %{customdata[0]:,.0f}"
            "<br>Improvement: %{customdata[1]:+.6f}<extra></extra>"
        ),
        showlegend=False,
    ))
    figure.add_vline(x=0, line={"color": NEUTRAL, "width": 1})
    figure.update_xaxes(title=axis_title, range=_linear_range(values, padding_ratio=0.08, zero_baseline=True))
    figure.update_yaxes(
        title=None,
        type="category",
        categoryorder="array",
        categoryarray=labels,
        autorange="reversed",
    )
    height = 350 if len(labels) > 4 else 315
    return _finish(
        figure,
        "Who Gains More from Alternative Data?",
        height=height,
        margins={"l": 150, "r": 22, "t": 58, "b": 56},
        show_legend=False,
    )


def _annotation_text(value: float) -> str:
    if 0 < value < 0.01:
        return "<1%"
    return f"{value:.0%}"


def reranking_figure(migration: pd.DataFrame, algorithm: str, n_bins: int) -> go.Figure:
    family = _family_id(algorithm)
    subset = migration.loc[migration.algorithm_family.eq(family) & migration.n_bins.eq(n_bins)].copy()
    index = range(1, n_bins + 1)
    z = subset.pivot(
        index="t_risk_bin", columns="t_plus_ad_risk_bin", values="share_within_t_bin"
    ).reindex(index=index, columns=index)
    counts = subset.pivot(
        index="t_risk_bin", columns="t_plus_ad_risk_bin", values="case_count"
    ).reindex(index=index, columns=index)
    rates = subset.pivot(
        index="t_risk_bin", columns="t_plus_ad_risk_bin", values="observed_default_rate"
    ).reindex(index=index, columns=index)
    custom = np.dstack([counts.to_numpy(), rates.to_numpy()])
    colorscale = [
        [0.00, "#FFFDF8"],
        [0.05, "#FBE8BC"],
        [0.10, "#F6D58A"],
        [0.20, "#EDB64A"],
        [0.30, TAD_COLOR],
        [0.45, "#C87500"],
        [0.60, "#9B5100"],
        [1.00, "#5E2C00"],
    ]
    figure = go.Figure(go.Heatmap(
        z=z.to_numpy(),
        x=list(index),
        y=list(index),
        customdata=custom,
        colorscale=colorscale,
        colorbar={
            "title": {"text": "Origin-row<br>share", "side": "right"},
            "tickformat": ".0%",
            "tickvals": [0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 1.0],
            "len": 0.84,
            "thickness": 14,
            "x": 0.82,
            "xpad": 2,
        },
        hovertemplate=(
            "Traditional-data risk group %{y} → combined-data risk group %{x}"
            "<br>Applicants: %{customdata[0]:,.0f}<br>Origin-row share: %{z:.1%}"
            "<br>Observed default rate: %{customdata[1]:.2%}<extra></extra>"
        ),
        zmin=0,
        zmax=1,
    ))
    annotation_size = {5: 11, 10: 9, 20: 7}[n_bins]
    for origin_group in index:
        for destination_group in index:
            if abs(destination_group - origin_group) <= 2:
                value = float(z.loc[origin_group, destination_group])
                figure.add_annotation(
                    x=destination_group,
                    y=origin_group,
                    text=_annotation_text(value),
                    showarrow=False,
                    font={"size": annotation_size, "color": "#FFFDF8" if value >= 0.45 else TEXT},
                )
    for group in index:
        figure.add_shape(
            type="rect",
            x0=group - 0.5,
            x1=group + 0.5,
            y0=group - 0.5,
            y1=group + 0.5,
            line={"color": "rgba(39,49,58,0.45)", "width": 1},
            fillcolor="rgba(0,0,0,0)",
        )
    figure.update_xaxes(
        title=(
            "Risk group using traditional + alternative data<br>"
            f"(1 = lowest risk, {n_bins} = highest risk)"
        ),
        dtick=1,
        constrain="domain",
        domain=[0.0, 0.80],
    )
    figure.update_yaxes(
        title=(
            "Risk group using traditional credit data<br>"
            f"(1 = lowest risk, {n_bins} = highest risk)"
        ),
        dtick=1,
        autorange="reversed",
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )
    return _finish(
        figure,
        "How Risk Rankings Change after Adding Alternative Data",
        height=405,
        margins={"l": 118, "r": 58, "t": 56, "b": 82},
        show_legend=False,
    )


def allocation_figure(
    approval: pd.DataFrame,
    risk: pd.DataFrame,
    policy: str,
    algorithm: str,
) -> go.Figure:
    family = _family_id(algorithm)
    t_model, tad_model = f"{family}_T", f"{family}_T_plus_AD"
    if policy in {"Same approval rate", "Fixed approval rate"}:
        frame, x_col, y_col = approval, "approval_rate_percent", "realised_approved_default_rate"
        title = "Portfolio Risk at the Same Approval Rate"
        x_title = "Approval rate (%)"
        y_title = "Realised approved default rate (%)"
        x_hover = "Approval rate"
        y_multiplier = 100
        same_approval = True
    else:
        frame, x_col, y_col = risk, "common_risk_budget_percent", "maximum_feasible_approval_rate_percent"
        title = "Credit Access at the Same Portfolio Risk"
        x_title = "Portfolio-risk limit (%)"
        y_title = "Maximum feasible approval rate (%)"
        x_hover = "Portfolio-risk limit"
        y_multiplier = 1
        same_approval = False
    figure = go.Figure()
    displayed_x: list[float] = []
    for index, (model, label, color) in enumerate((
        (t_model, TRADITIONAL_LABEL, T_COLOR),
        (tad_model, ALTERNATIVE_LABEL, TAD_COLOR),
    )):
        subset = frame.loc[frame.model_id.eq(model)].sort_values(x_col, kind="stable")
        displayed_x.extend(subset[x_col].astype(float).tolist())
        custom = np.column_stack([
            subset.approved_count.to_numpy(),
            subset.realised_approved_defaults.to_numpy(),
            subset.boundary_calibrated_pd_cutoff.to_numpy(),
        ])
        figure.add_trace(go.Scatter(
            x=subset[x_col],
            y=subset[y_col] * y_multiplier,
            mode="lines",
            name=label,
            line={"color": color, "width": 3},
            fill="tonexty" if index == 1 else None,
            fillcolor="rgba(230,159,0,0.09)" if index == 1 else None,
            customdata=custom,
            hovertemplate=(
                f"{label}<br>{x_hover}: %{{x:.2f}}%<br>Approval rate: "
                + ("%{x:.2f}%" if same_approval else "%{y:.2f}%")
                + "<br>Approved: %{customdata[0]:,.0f}<br>Realised defaults: %{customdata[1]:,.0f}"
                + "<br>Boundary probability of default: %{customdata[2]:.3%}<extra></extra>"
            ),
        ))
        anchor_subset = subset.loc[subset.is_external_display_point.astype(bool)]
        figure.add_trace(go.Scatter(
            x=anchor_subset[x_col],
            y=anchor_subset[y_col] * y_multiplier,
            mode="markers",
            marker={"size": 8, "color": color, "line": {"color": BACKGROUND, "width": 1}},
            hoverinfo="skip",
            showlegend=False,
        ))
    figure.update_xaxes(title=x_title, range=_linear_range(displayed_x, padding_ratio=0.03))
    figure.update_yaxes(title=y_title)
    return _finish(figure, title, legend_spacing=True)


def value_figure(economic: pd.DataFrame, algorithm: str, basis: str) -> go.Figure:
    family = _family_id(algorithm)
    subset = economic.loc[economic.algorithm_family.eq(family)].sort_values(
        "loss_to_gain_ratio", kind="stable"
    )
    if basis in {"Expected value based on model probabilities", "Expected"}:
        y_column = "predicted_expected_value_per_evaluation_case_change"
        y_title = "Expected incremental value per evaluated application"
    else:
        y_column = "normalized_break_even_ad_cost_per_scored_case"
        y_title = "Maximum normalized alternative-data cost per application"
    x = subset.loss_to_gain_ratio.to_numpy(float)
    y = subset[y_column].to_numpy(float)
    figure = go.Figure(go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        line={"color": TAD_COLOR, "width": 3},
        marker={"size": 9, "color": TAD_COLOR},
        customdata=np.column_stack([
            subset.approval_rate_percent_change.to_numpy(),
            subset.realised_default_count_change.to_numpy(),
        ]),
        hovertemplate=(
            "Default loss relative to non-default gain: %{x:g}"
            "<br>Incremental value: %{y:+.6f}<br>Approval-rate change: %{customdata[0]:+.3f} pp"
            "<br>Realised default-count change: %{customdata[1]:+,.0f}<extra></extra>"
        ),
        showlegend=False,
    ))
    low, high = min(float(y.min()), 0), max(float(y.max()), 0)
    padding = max(high - low, 0.001) * 0.08
    figure.add_hrect(
        y0=0,
        y1=high + padding,
        fillcolor="rgba(0,158,115,0.055)",
        line_width=0,
        layer="below",
    )
    figure.add_hrect(
        y0=low - padding,
        y1=0,
        fillcolor="rgba(213,94,0,0.05)",
        line_width=0,
        layer="below",
    )
    figure.add_hline(y=0, line={"color": NEUTRAL, "width": 1})
    figure.update_xaxes(
        title="Default loss relative to non-default gain",
        type="log",
        tickvals=x.tolist(),
        range=_log_range(x),
    )
    figure.update_yaxes(title=y_title, range=[low - padding, high + padding])
    return _finish(
        figure,
        "Lender-side Value of Alternative Data across Loss Scenarios",
        show_legend=False,
    )
