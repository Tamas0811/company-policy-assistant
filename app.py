import pandas as pd
import streamlit as st

from src.llm_direct import LLMDirectSearcher
from src.llm_rag import LLMRAGSearcher
from src.policies import DECLINED, ROOT, SUPPORTED, UNSUPPORTED
from src.rules_search import RulesSearcher

st.set_page_config(page_title="Company Policy Assistant", layout="wide")


@st.cache_resource
def get_engines():
    return RulesSearcher(), LLMDirectSearcher(), LLMRAGSearcher()


GROUNDING_TEXT = {
    SUPPORTED: "✅ Supported: the answer cites a policy that exists in the database",
    DECLINED: "⛔ Declined: no policy in the database covers this question",
    UNSUPPORTED: "⚠️ Unsupported: answered without a valid policy from the database",
    "Error": "❗ The model could not be reached",
}


def show_result(title, res, box):
    st.header(title)
    st.subheader(f"Policy: {res['policy']}")
    st.write(f"**Category:** {res['category']}")
    box(f"**Answer:** {res['answer']}")
    c1, c2 = st.columns(2)
    c1.metric("Response time", f"{res['response_time']}s")
    c2.metric("Token use", f"{res['token_usage']['total_tokens']:,}")
    st.caption(GROUNDING_TEXT.get(res["grounding"], res["grounding"]))
    if res.get("retrieved"):
        st.caption("Retrieved from index: " + ", ".join(res["retrieved"]))


st.title("🏢 Company Policy Assistant: Comparison")
st.markdown(
    "Compare **rules-based search**, an **LLM without a vector index** (all policies in the prompt) "
    "and an **LLM with a vector index (RAG)** side by side."
)

query = st.text_input("Ask a company policy question:", value="How many vacation days do I get?")

if st.button("Submit question", type="primary") and query.strip():
    with st.spinner("Running all three approaches..."):
        rules_engine, direct_engine, rag_engine = get_engines()
        results = [
            ("1. Rules-based (BM25)", rules_engine.search(query), st.info),
            ("2. LLM without index", direct_engine.search(query), st.warning),
            ("3. LLM + vector index (RAG)", rag_engine.search(query), st.success),
        ]
    for col, (title, res, box) in zip(st.columns(3), results):
        with col:
            show_result(title, res, box)

st.divider()

st.subheader("Evaluation: 10 questions the policies cover, 6 they don't")
results_path = ROOT / "evaluation_results.csv"
if results_path.exists():
    st.dataframe(pd.read_csv(results_path), hide_index=True, use_container_width=True)
    details_path = ROOT / "evaluation_details.csv"
    if details_path.exists():
        st.download_button("Download per-question results (CSV)", details_path.read_bytes(),
                           file_name="evaluation_details.csv", mime="text/csv")
    st.caption(
        "An *unsupported answer* is any answer to a question the policy database does not cover "
        "(instead of saying no policy was found), or an answer that cites no valid policy."
    )
else:
    st.info("No evaluation results yet. Run `python -m src.evaluate` and commit evaluation_results.csv.")

st.subheader("Comparison and preferred approach")
st.markdown("""
On the 16 test questions in the table above, rules-based BM25 search answered instantly and used no tokens, but it found the right policy for only 6 of the 10 questions the database covers, because it matches words rather than meaning (a question about a "baby" never reaches the Parental Leave Policy). It also answered 4 of the 6 questions the database does not cover by returning an unrelated policy that happened to share a keyword. Both LLM approaches identified the correct policy for all 10 covered questions and declined all 6 uncovered ones, so on this policy file neither produced answers unsupported by the database. The cost was very different, however: the LLM without a vector index sends all 98 policies with every question, using about 1,900 tokens per question against about 200 for the LLM with a vector index, with a median response time of 5.2 seconds against 3.3 (the no-index average of 8.2 seconds is inflated by one 63-second call, most likely a rate-limit delay).

The preferred approach is therefore the LLM with a vector index (RAG). It matched the no-index LLM on accuracy and grounding while using roughly a ninth of the tokens and a shorter response time, because semantic retrieval narrows each question to the three most relevant policies before the model answers, and it cites the policy it used so every answer can be traced back to the database. The gap will also widen as the policy library grows: the no-index approach gets more expensive and slower with every policy added, while RAG's cost per question stays roughly flat. Its main risk is that a missed retrieval leads to a missed answer, which did not happen in this test but remains the thing to monitor, whereas rules-based search is only suitable as a free fallback for exact keyword lookups.
""")

