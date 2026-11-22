"""Regressions page: the full tables behind the gender results, each with its equation."""

import json

import streamlit as st

from src.ui import DATA, chapter, doc_table, lead, page_title

TABLES = json.loads((DATA / "rg_tables.json").read_text(encoding="utf-8"))

EQUATIONS = {
    1: (r"""\begin{aligned}
&\text{Panel A:}\quad \Pr(D_i=1)=\Lambda\big(a+b\,\mathrm{logit}(\mathrm{PD}_i)+g\,\mathrm{Female}_i\big)\\[2pt]
&\text{Panel B:}\quad 100\,(D_i-\mathrm{PD}_i)=a+g\,\mathrm{Female}_i+\varepsilon_i\\[2pt]
&\text{Panel C:}\quad \Pr(D_{im}=1)=\Lambda\big(a_m+b_m\,\mathrm{logit}(\mathrm{PD}_{im})+g_m\,\mathrm{Female}_i\big),
\quad m\in\{\mathrm{T},\ \mathrm{T{+}AD}\},\quad \text{reported: } g_{\mathrm{T+AD}}-g_{\mathrm{T}}
\end{aligned}""",
        "Column (1) omits Female. In Panel C the two models' rows are stacked."),
    2: (r"""\Pr(D_i=1)=\Lambda\big(a+b\,\mathrm{logit}(\mathrm{PD}_i)+g\,\mathrm{Female}_i+\mu_{t(i)}+\nu_{p(i)}\big)""",
        "μ: decision-month fixed effects; ν: product-type fixed effects. Each enters only in the rows that name it."),
    3: (r"""\Pr(D_i=1)=\Lambda\Big(a+b\,\mathrm{logit}(\mathrm{PD}_i)+g\,\mathrm{Female}_i+c_1\,\mathrm{Age}_i+c_2\,\mathrm{Age}_i^2/100
+\textstyle\sum_k d_k\,\mathrm{Family}_{ik}+\sum_l e_l\,\mathrm{Employ}_{il}\Big)""",
        "Controls enter one block at a time, columns (3) to (5). Missing family state and employment length are "
        "separate categories."),
    4: (r"""s_i=\phi_0+\sum_{j}\phi_{ij},\qquad \Delta_j=\bar\phi_j^{\,\mathrm{women}}-\bar\phi_j^{\,\mathrm{men}},
\qquad \sum_j\Delta_j=\bar s^{\,\mathrm{women}}-\bar s^{\,\mathrm{men}}""",
        "s: the model score in log-odds; φ<sub>j</sub>: SHAP contribution of variable family j. This is an exact decomposition, "
        "not a regression."),
    5: (r"""\Pr(D_i=1)=\Lambda\big(a+b\,(L_i-\bar L)+g\,\mathrm{Female}_i+h\,\mathrm{Female}_i\,(L_i-\bar L)
+\mathbf{x}_i'\boldsymbol{\theta}\big),\qquad L_i=\mathrm{logit}(\mathrm{PD}_i)""",
        "Column (1) has no interaction. x: demographic controls, column (3) only."),
    6: (r"""\Pr(D_i=1)=\Lambda\big(a_k+b_k\,\mathrm{logit}(\mathrm{PD}_i)+g_k\,\mathrm{Female}_i\big),\quad i\in\text{group }k;
\qquad H_0:\ g_1=g_2=\dots=g_K""",
        "Estimated separately in each group; the equality test uses a fully interacted pooled logit."),
    7: (r"""\Pr(D_i=1)=\Lambda\Big(a+\sum_j c_j\,(S_{ij}-\bar S_j)+g\,\mathrm{Female}_i
+\sum_j h_j\,\mathrm{Female}_i\,(S_{ij}-\bar S_j)\Big)""",
        "S<sub>j</sub>: the part of the score from variable family j. Men's coefficient is c, women's is c + h."),
}

page_title("Regressions", "Regression tables")
lead("The full tables behind the Evidence page, as in the research write-up. All regressions use the final sample "
     "of 228,999 applications (7,199 defaults) unless the note says otherwise. D<sub>i</sub> = 1 if application i "
     "defaults, and Λ(x) = 1 / (1 + e<sup>−x</sup>) is the logistic function.")
links = "".join(f'<div><a href="#table-{t["number"]}">Table {t["number"]}. {t["caption"]}</a></div>' for t in TABLES)
st.markdown(f'<div class="toc">{links}</div>', unsafe_allow_html=True)

for t in TABLES:
    n = t["number"]
    st.markdown(f'<div id="table-{n}"></div>', unsafe_allow_html=True)
    chapter(f"Table {n}. {t['caption']}")
    eq, defs = EQUATIONS[n]
    st.latex(eq)
    st.markdown(f'<p class="eq-def">{defs}</p>', unsafe_allow_html=True)
    doc_table(t)
