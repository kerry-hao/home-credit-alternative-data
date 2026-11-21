"""Portfolio app: gender bias in a gender-blind credit model."""

import streamlit as st

from src.theme import CSS

st.set_page_config(page_title="Blind is not neutral", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)
page = st.navigation(
    [
        st.Page("views/story.py", title="Story", default=True),
        st.Page("views/evidence.py", title="Evidence"),
        st.Page("views/method.py", title="Method"),
    ],
    position="top",
)
page.run()
