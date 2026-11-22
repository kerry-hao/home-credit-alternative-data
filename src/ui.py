"""Small rendering helpers shared by the pages."""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data" / "showcase_v2"
CHART_CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(show_spinner=False)
def load() -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_csv(p) for p in DATA.glob("*.csv")}


def plot(fig) -> None:
    """Render a chart title as wrapping HTML, then the chart."""
    title = (fig.layout.meta or {}).get("title", "") if isinstance(fig.layout.meta, dict) else ""
    if title:
        st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, config=CHART_CONFIG, width="stretch")


def m(text: str) -> str:
    """Use a true minus sign in displayed numbers."""
    return text.replace("-", "−")


def tiles(items: list[tuple[str, str]], css_class: str = "tiles") -> None:
    cells = "".join(f'<div class="tile"><div class="num">{n}</div><div class="lab">{l}</div></div>' for n, l in items)
    st.markdown(f'<div class="{css_class}">{cells}</div>', unsafe_allow_html=True)


def chapter(text: str, first: bool = False) -> None:
    rule = "" if first else '<div class="chapter-rule"></div>'
    st.markdown(f'{rule}<div class="chapter">{text}</div>', unsafe_allow_html=True)


def lead(text: str) -> None:
    st.markdown(f'<p class="lead">{text}</p>', unsafe_allow_html=True)


def page_title(kicker: str, title: str) -> None:
    st.markdown(f'<div class="kicker">{kicker}</div><div class="page-title">{title}</div>', unsafe_allow_html=True)


def table(header: list[str], rows: list, align: list[str] | None = None, note: str | None = None,
          max_width: str = "100%") -> None:
    align = align or ["left"] + ["right"] * (len(header) - 1)
    head = "".join(f'<th style="text-align:{a}">{h}</th>' for h, a in zip(header, align))
    body = ""
    for r in rows:
        if isinstance(r, str):
            body += f'<tr class="panel"><td colspan="{len(header)}">{r}</td></tr>'
        else:
            body += "<tr>" + "".join(f'<td style="text-align:{a}">{c}</td>' for c, a in zip(r, align)) + "</tr>"
    html = f'<div class="tbl-wrap" style="max-width:{max_width}"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    if note:
        html += f'<p class="tbl-note">{note}</p>'
    st.markdown(html, unsafe_allow_html=True)


def doc_table(t: dict) -> None:
    """Render a table extracted from the Word write-up, keeping its header spans and panel rows."""
    import html as _h

    def cell(c, tag, first, extra=""):
        cls = []
        if c.get("rule") and c.get("span", 1) > 1 and tag == "th":
            cls.append("rule")
        align = "left" if first else "center"
        span = f' colspan="{c["span"]}"' if c.get("span", 1) > 1 else ""
        txt = _h.escape(c["text"])
        return f'<{tag}{span} class="{" ".join(cls)}" style="text-align:{align}"{extra}>{txt}</{tag}>'

    ncol = sum(c["span"] for c in t["header"][-1])
    head = ""
    for r in t["header"]:
        head += "<tr>" + "".join(cell(c, "th", j == 0) for j, c in enumerate(r)) + "</tr>"
    body = ""
    for r in t["body"]:
        if len(r) == 1 and r[0]["span"] == ncol:
            body += f'<tr class="panel"><td colspan="{ncol}">{_h.escape(r[0]["text"])}</td></tr>'
        else:
            body += "<tr>" + "".join(cell(c, "td", j == 0) for j, c in enumerate(r)) + "</tr>"
    st.markdown(
        f'<div class="tbl-wrap"><table class="tbl doc"><thead>{head}</thead><tbody>{body}</tbody></table></div>'
        f'<p class="tbl-note"><i>Note:</i> {_h.escape(t["note"])}</p>', unsafe_allow_html=True)
