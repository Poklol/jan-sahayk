from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from memory import append_message, init_state, profile_snapshot, reset_state
from orchestrator import JanSahayakOrchestrator
from rag import RAGAgent
from web_search import WebSearchAgent


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
st.set_page_config(page_title="Jan-Sahayak AI", page_icon="\U0001F1EE\U0001F1F3", layout="wide")


def _bootstrap_agents() -> None:
    if "rag_agent" not in st.session_state:
        st.session_state.rag_agent = RAGAgent()
    if "web_agent" not in st.session_state:
        st.session_state.web_agent = WebSearchAgent()
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = JanSahayakOrchestrator(
            st.session_state.rag_agent,
            st.session_state.web_agent,
        )
    if "index_status" not in st.session_state:
        st.session_state.index_status = st.session_state.rag_agent.ensure_index()


def _render_sidebar() -> None:
    st.sidebar.header("Session")
    st.sidebar.markdown("### User Profile")
    st.sidebar.markdown(profile_snapshot(st.session_state.user_profile))

    st.sidebar.markdown("### RAG Index")
    st.sidebar.json(st.session_state.index_status)

    if st.sidebar.button("Index / Rebuild RAG Data"):
        with st.spinner("Indexing local scheme documents..."):
            st.session_state.index_status = st.session_state.rag_agent.ensure_index(
                force_reindex=True
            )
        st.rerun()

    status = str(st.session_state.index_status.get("status", ""))
    if status == "no_docs":
        st.sidebar.info(
            "Add .txt or .md files under docs/ and click 'Index / Rebuild RAG Data'."
        )
    elif status == "missing_gemini_key":
        st.sidebar.error("Set GEMINI_API_KEY in .env, then restart the app.")

    if st.sidebar.button("Reset Conversation"):
        for key in ["orchestrator", "rag_agent", "web_agent"]:
            if key in st.session_state:
                del st.session_state[key]
        reset_state()
        st.rerun()


def _render_chat_history() -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    init_state()
    _bootstrap_agents()

    st.title("Jan-Sahayak AI")
    st.caption("AI-powered welfare scheme assistant with RAG + official web updates")

    _render_chat_history()

    prompt = st.chat_input("Describe your situation and benefit need...")
    
    if prompt:
        append_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Analyzing profile, checking schemes, and gathering evidence..."):
            result = st.session_state.orchestrator.handle_query(
                user_query=prompt,
                user_profile=st.session_state.user_profile,
                chat_history=st.session_state.chat_history,
            )

        st.session_state.user_profile = result.get("updated_profile", st.session_state.user_profile)
        response_markdown = result.get("response_markdown", "I could not process that request.")

        with st.chat_message("assistant"):
            if "rag_evidence" in result and result["rag_evidence"]:
                with st.expander("🔍 RAG Debugger: Retrieved Chunks"):
                    for i, doc in enumerate(result["rag_evidence"]):
                        st.markdown(f"**Chunk {i+1} | Score: {doc.get('score', 0):.2f} | Source: {doc.get('source', '')}**")
                        st.markdown(f"_{doc.get('quote', '')[:200]}..._")

            if hasattr(response_markdown, '__iter__') and not isinstance(response_markdown, str):
                # Stream the response natively
                assistant_reply = st.write_stream(response_markdown)
            else:
                assistant_reply = str(response_markdown)
                st.markdown(assistant_reply)

        append_message("assistant", assistant_reply)

    # Render sidebar at the end so it uses the most up-to-date user_profile state
    _render_sidebar()

if __name__ == "__main__":
    main()
