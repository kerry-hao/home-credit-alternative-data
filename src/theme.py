"""Shared colours, typography and Plotly styling for the portfolio app."""

import plotly.graph_objects as go

INK = "#24292F"
INK_2 = "#57606A"
GRID = "#E9EBEE"
AXIS = "#C4C9CF"
REFERENCE = "#9AA1A9"
SURFACE = "#FFFFFF"

WOMEN = "#209993"
MEN = "#AC5853"
WOMEN_TINT = "#E3F3F1"
MEN_TINT = "#F7E7E5"
BLIND = "#9AA1A9"
AWARE = INK

FONT = "Source Sans Pro, Source Sans 3, Helvetica Neue, Arial, sans-serif"
GENDER_COLOR = {"Women": WOMEN, "Men": MEN}


def base_layout(fig: go.Figure, *, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=16, b=8),
        meta=dict(title=title or ""),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=14, color=INK),
        showlegend=False,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(family=FONT, size=13, color=INK)),
    )
    fig.update_xaxes(automargin=True, showgrid=True, gridcolor=GRID, zeroline=False, showline=True, linecolor=AXIS,
                     ticks="outside", tickcolor=AXIS, title_font=dict(size=14, color=INK_2),
                     tickfont=dict(size=13, color=INK_2))
    fig.update_yaxes(automargin=True, showgrid=True, gridcolor=GRID, zeroline=False, showline=True, linecolor=AXIS,
                     ticks="outside", tickcolor=AXIS, title_font=dict(size=14, color=INK_2),
                     tickfont=dict(size=13, color=INK_2))
    return fig


CSS = f"""
<style>
  .block-container {{ max-width: 1080px; padding-top: 4.6rem; padding-bottom: 4rem; }}
  h1, h2, h3, p, div {{ font-family: {FONT}; }}
  .kicker {{ font-size: 1rem; color: {INK}; margin: 0 0 .1rem; }}
  .hero-title {{ font-size: 2.6rem; font-weight: 600; color: {INK}; line-height: 1.1; margin: 0 0 1.4rem; }}
  .tiles {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 1.2rem; }}
  .tile {{ background: #F4F5F7; border-radius: 10px; padding: 14px 16px; }}
  .tile .num {{ font-size: 1.7rem; font-weight: 600; color: {INK}; line-height: 1.15; }}
  .tile .lab {{ font-size: .92rem; color: {INK}; margin-top: 4px; line-height: 1.3; }}
  .terms {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px;
            border-top: 1px solid {GRID}; padding: 12px 0 0; margin-bottom: .4rem; }}
  .terms .t {{ font-size: .95rem; color: {INK}; line-height: 1.35; }}
  .terms .t b {{ font-weight: 600; }}
  .chapter {{ font-size: 1.55rem; font-weight: 600; color: {INK}; margin: 1.6rem 0 .8rem; line-height: 1.25; }}
  .chapter-rule {{ border-top: 1px solid {GRID}; margin-top: 2.2rem; }}
  .pair {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: .4rem; }}
  .pair .tile .num {{ font-size: 1.35rem; }}
  .stack {{ display: block; margin-top: .9rem; }}
  .stack .tile {{ margin-bottom: 12px; }}
  .stack .tile .num {{ font-size: 1.35rem; }}
  .page-title {{ font-size: 2.1rem; font-weight: 600; color: {INK}; line-height: 1.15; margin: 0 0 .8rem; }}
  .chart-title {{ font-size: 1rem; font-weight: 600; color: {INK}; margin: .6rem 0 .2rem; line-height: 1.35; }}
  .sub {{ font-size: 1.18rem; font-weight: 600; color: {INK}; margin: 1.8rem 0 .4rem; }}
  .lead {{ font-size: 1.02rem; color: {INK}; line-height: 1.55; margin: 0 0 .9rem; max-width: 760px; }}
  .tbl-wrap {{ overflow-x: auto; margin: .4rem 0 .2rem; }}
  .tbl {{ border-collapse: collapse; width: 100%; font-size: .93rem; color: {INK}; }}
  .tbl thead th {{ font-weight: 600; border-top: 1.5px solid {INK}; border-bottom: 1px solid {INK}; padding: 8px 10px;
                   vertical-align: bottom; }}
  .tbl tr, .tbl td, .tbl th {{ background: none !important; }}
  .tbl tr, .tbl tbody tr {{ border: none !important; }}
  .tbl td {{ padding: 7px 10px; border: none !important; vertical-align: top; }}
  .tbl thead th {{ border-left: none !important; border-right: none !important; }}
  .tbl tbody tr:last-child td {{ border-bottom: 1.5px solid {INK} !important; }}
  .tbl tr.panel td {{ font-style: italic; padding-top: 12px; }}
  .tbl-note {{ font-size: .85rem; color: {INK_2}; margin: .4rem 0 0; line-height: 1.45; max-width: 860px; }}
  .tbl.doc {{ font-size: .9rem; }}
  .tbl.doc thead th {{ border-top: none !important; border-bottom: none !important; font-weight: 600; padding: 6px 10px; }}
  .tbl.doc thead tr:first-child th {{ border-top: 1.5px solid {INK} !important; }}
  .tbl.doc thead tr:last-child th {{ border-bottom: 1px solid {INK} !important; }}
  .tbl.doc thead th.rule {{ border-bottom: 1px solid {INK} !important; }}
  .tbl.doc td {{ padding: 3px 10px; white-space: nowrap; }}
  .tbl.doc td:first-child {{ white-space: normal; }}
  .cap {{ font-size: 1.05rem; color: {INK}; margin: .2rem 0 .3rem; }}
  .cap b {{ font-weight: 600; }}
  .toc a {{ color: {INK}; text-decoration: none; border-bottom: 1px solid {GRID}; }}
  .toc a:hover {{ border-bottom-color: {INK}; }}
  .toc div {{ margin: .25rem 0; font-size: 1rem; }}
  .eq-def {{ font-size: .88rem; color: {INK_2}; margin: -.2rem 0 .6rem; line-height: 1.45; }}
  [data-testid="stWidgetLabel"] p {{ font-size: .95rem; color: {INK}; }}
  @media (max-width: 900px) {{
    .tiles, .pair {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .terms {{ grid-template-columns: 1fr; }}
    .hero-title {{ font-size: 2.1rem; }}
  }}
</style>
"""


def legend_name(text: str) -> str:
    """Pad legend labels so Plotly does not clip them when the web font loads after measuring."""
    return text + "\u00a0\u00a0\u00a0"
