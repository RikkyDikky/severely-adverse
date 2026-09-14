"""
CCAR Stress Testing Assistant -- Streamlit app.

Wraps the hybrid search + citation-bearing generation pipeline from
02_search_and_generate.py in a simple chat interface, matching how
Topic 1's project shipped.

Uses @st.cache_resource to load the pipeline (Chroma client, BM25 index,
Voyage/Anthropic clients) exactly once per server process instead of on
every rerun -- Streamlit reruns the whole script on each interaction, and
rebuilding the BM25 index from scratch every time would be slow and
pointless since the underlying data doesn't change between questions.
"""

import importlib.util
import pathlib
import streamlit as st

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


@st.cache_resource
def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "search_and_generate", str(SCRIPT_DIR / "02_search_and_generate.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sg = load_pipeline()

st.set_page_config(page_title="Severely Adverse", page_icon="🏦")
st.title("Severely Adverse")
st.caption("A CCAR/DFAST stress-testing Q&A assistant")
st.caption(
    "Answers questions about the Federal Reserve's 2026 CCAR/DFAST cycle, "
    "grounded in five public documents: the 2026 Stress Test Scenarios, "
    "2026 Stress Test Methodology, 2026 DFAST Results, SR 15-18, and the "
    "CCAR Q&As. Every answer cites its source document and page."
)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a question about CCAR / DFAST...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the source documents..."):
            retrieved = sg.hybrid_search(question, top_k=5)
            answer = sg.generate_answer(question, retrieved)
        st.markdown(answer)
        with st.expander("Sources retrieved for this answer"):
            for c in retrieved:
                st.markdown(f"- **{c['document']}**, page {c['page']}")

    st.session_state.history.append({"role": "assistant", "content": answer})
