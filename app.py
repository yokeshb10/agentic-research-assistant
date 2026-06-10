from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from pypdf import PdfReader
import io

from agent import build_index_from_text, run_research, run_quick

@st.cache_resource
def get_index(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = "".join(page.extract_text() for page in reader.pages)
    return build_index_from_text(text)

# ---------- UI ----------
st.title("Agentic Research Assistant")
st.write("Upload a PDF, then ask a question. The assistant plans, retrieves, verifies, and answers.")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is not None:
    file_bytes = uploaded.read()
    with st.spinner("Indexing document..."):
        indexed = get_index(file_bytes)
    st.success(f"Indexed {len(indexed)} sections. Ask away!")

    mode = st.radio(
        "Mode",
        ["Quick answer", "Deep research (splits complex questions, slower)"],
        horizontal=True,
    )
    question = st.text_input("Your question:")

    if question:
        try:
            with st.spinner("Working... (deep research can take ~30-60s on the free tier)"):
                if mode.startswith("Quick"):
                    result = run_quick(question, indexed)
                else:
                    result = run_research(question, indexed)

            # show the plan only if it actually split into multiple sub-questions
            if len(result["sub_questions"]) > 1:
                with st.expander(f"🧭 Research plan ({len(result['sub_questions'])} sub-questions)"):
                    for i, sq in enumerate(result["sub_questions"], 1):
                        st.write(f"{i}) {sq}")

            st.subheader("Answer")
            st.write(result["final_answer"])

            st.subheader("Details")
            for r in result["sub_results"]:
                with st.expander(f"📄 {r['question']}"):
                    if r["status"] == "verified":
                        st.success("✓ Verified against sources")
                    elif r["status"] == "unverified":
                        st.warning("⚠ Could not be verified")
                    else:
                        st.info("ℹ️ Not found in document")

                    if r["status"] == "no_answer":
                        st.write("Not found in the document.")
                    else:
                        st.write(r["answer"])

                    st.caption("Sources:")
                    for i, it in enumerate(r["sources"], 1):
                        snippet = it["text"][:200].replace("\n", " ")
                        st.markdown(f"**[{i}]** (score {it['score']:.3f}) — {snippet}...")
        except Exception as e:
            st.error(f"Error: {type(e).__name__}: {e}")
else:
    st.info("Please upload a PDF to begin.")