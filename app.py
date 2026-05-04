from __future__ import annotations

import re
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv
import streamlit as st

from memory import append_message, init_state, reset_state
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


def _init_ui_state() -> None:
    if "ui_tab" not in st.session_state:
        st.session_state.ui_tab = "HOME"
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False
    if "sync_state_name" not in st.session_state:
        st.session_state.sync_state_name = "Tamil Nadu"
    if "selected_scheme_slug" not in st.session_state:
        st.session_state.selected_scheme_slug = ""


def _inject_styles() -> None:
    st.markdown(
        """
<style>
:root {
  --bg: #040608;
  --panel: #12161c;
  --panel2: #0d1117;
  --text: #e8ecf1;
  --muted: #8e95a2;
  --mint: #2fd2aa;
  --yellow: #f3d21b;
}
.stApp {
  background: radial-gradient(1000px 500px at 85% -10%, #0a2020 0%, var(--bg) 60%);
  color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1200px; padding-top: 1rem; padding-bottom: 8rem; }

.brand-row {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 1rem;
}
.brand-title {
  font-family: Georgia, 'Times New Roman', serif;
  color: #f3f6f9; font-size: 2rem; margin: 0;
}
.brand-sub { color: var(--muted); margin-top: 0.3rem; }

.hero-box {
  background: linear-gradient(135deg, #070b11 0%, #10171e 100%);
  border: 1px solid #1b2430;
  border-radius: 22px;
  padding: 2rem;
  margin-bottom: 1.2rem;
}
.hero-title {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 4rem;
  line-height: 1.05;
  color: #eaf2ef;
  margin: 0;
}
.hero-accent { color: var(--yellow); }
.hero-copy { color: var(--muted); font-size: 1.15rem; margin-top: 0.9rem; }

.section-title {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 2.4rem;
  color: #e9edf2;
  margin: 0.2rem 0 1rem 0;
}

.scheme-card {
  background: linear-gradient(135deg, #141a21 0%, #11161c 100%);
  border: 1px solid #202833;
  border-radius: 22px;
  padding: 1.4rem;
  margin-bottom: 0.9rem;
}
.scheme-badge {
  display: inline-block;
  background: #0f6f45;
  color: #9df6c8;
  border-radius: 999px;
  padding: 0.2rem 0.8rem;
  font-weight: 700;
  font-size: 0.8rem;
  letter-spacing: 0.12em;
}
.scheme-name {
  color: #f3f6fb;
  font-size: 1.8rem;
  font-family: Georgia, 'Times New Roman', serif;
  margin: 0.7rem 0 0.35rem 0;
}
.scheme-copy { color: var(--muted); margin-bottom: 0.9rem; }
.apply-btn {
  display: inline-block;
  background: var(--yellow);
  color: #161204;
  font-weight: 700;
  border-radius: 12px;
  padding: 0.65rem 1.2rem;
  text-decoration: none;
}

.status-panel {
  background: linear-gradient(135deg, #181d25 0%, #12161d 100%);
  border: 1px solid #283241;
  border-radius: 22px;
  padding: 1.4rem;
  margin-bottom: 1rem;
}
.status-kicker { color: var(--yellow); letter-spacing: 0.14em; font-weight: 700; }
.status-row {
  border-top: 1px solid #263143;
  margin-top: 0.8rem;
  padding-top: 0.8rem;
}
.status-ok { color: #9ff1c6; font-weight: 700; }

.help-panel {
  background: linear-gradient(135deg, #087744 0%, #055835 100%);
  border-radius: 22px;
  padding: 1.5rem;
  color: #d3f6e6;
}

.ms-home {
  background: linear-gradient(160deg, #242833 0%, #2a2f3b 100%);
  border: 1px solid #373e4e;
  border-radius: 24px;
  overflow: hidden;
  margin-bottom: 1.2rem;
}
.ms-strip {
  background: linear-gradient(90deg, #2a2f3a 0%, #242a34 100%);
  border-bottom: 1px solid #3a4255;
  padding: 0.9rem 1.3rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}
.ms-brand {
  color: #e7edf7;
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 1.9rem;
  margin: 0;
}
.ms-search {
  background: #232833;
  border: 1px solid #8b94a8;
  border-radius: 12px;
  padding: 0.55rem 0.85rem;
  color: #dde6f3;
  min-width: 320px;
}
.ms-hero {
  background: radial-gradient(1400px 500px at 80% 10%, #204a23 0%, #10131b 45%, #090c12 100%);
  padding: 2.2rem 1.4rem;
}
.ms-kicker {
  letter-spacing: 0.12em;
  color: #8ad89a;
  font-weight: 700;
  margin: 0;
}
.ms-title {
  font-family: Georgia, 'Times New Roman', serif;
  color: #f1f5fb;
  font-size: 3rem;
  line-height: 1.06;
  margin: 0.4rem 0;
}
.ms-copy {
  color: #d5deec;
  font-size: 1.05rem;
}
.ms-cta {
  display: inline-block;
  background: #3d8b44;
  color: #eef8ed;
  padding: 0.65rem 1.2rem;
  border-radius: 12px;
  text-decoration: none;
  font-weight: 700;
}
.ms-stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.9rem;
  margin: 1rem 0;
}
.ms-stat-card {
  background: #040d1b;
  border: 1px solid #1a2c49;
  border-radius: 18px;
  padding: 1rem;
  text-align: center;
}
.ms-stat-n {
  color: #eef3fb;
  font-size: 2.1rem;
  font-weight: 800;
}
.ms-stat-l {
  color: #d6deea;
}
.ms-tabs {
  display: flex;
  gap: 0.6rem;
  justify-content: center;
  margin: 0.8rem 0 1rem 0;
}
.ms-tab {
  background: #2a2f3a;
  color: #d7deea;
  border: 1px solid #404a60;
  border-radius: 10px;
  padding: 0.35rem 0.8rem;
  font-weight: 700;
}
.ms-heading {
  text-align: center;
  color: #f1f5fb;
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 2.2rem;
  margin: 0.2rem 0 0.9rem 0;
}
.ms-categories {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.8rem;
}
.ms-cat {
  background: linear-gradient(135deg, #2a2f3a 0%, #272c36 100%);
  border: 1px solid #3b4458;
  border-radius: 14px;
  padding: 0.7rem;
  min-height: 130px;
}
.ms-cat-ico { font-size: 1.6rem; }
.ms-cat-n { color: #62d989; font-weight: 700; }
.ms-cat-t { color: #e3e9f3; font-size: 1rem; }
.ms-how {
  background: linear-gradient(135deg, #262b35 0%, #2c323f 100%);
  border: 1px solid #3a4357;
  border-radius: 22px;
  padding: 1.2rem;
  margin-top: 1rem;
}
.ms-steps {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.8rem;
}
.ms-step {
  background: linear-gradient(135deg, #303541 0%, #2a2f3a 100%);
  border: 1px solid #454f66;
  border-radius: 14px;
  padding: 0.9rem;
  text-align: center;
}
.scheme-shell {
  background: linear-gradient(135deg, #121720 0%, #1a202a 100%);
  border: 1px solid #2a3240;
  border-radius: 18px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.85rem;
}
.scheme-head {
  color: #edf2f8;
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 2rem;
  margin: 0;
}
.scheme-meta {
  color: #95a4b8;
  font-size: 1rem;
  margin-top: 0.4rem;
}
.scheme-desc {
  color: #d7dfea;
  margin-top: 0.55rem;
  line-height: 1.45;
}
.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 0.65rem;
}
.tag-pill {
  display: inline-block;
  border: 1px solid #2f9a55;
  color: #8ee9b2;
  background: rgba(41, 110, 70, 0.22);
  border-radius: 999px;
  padding: 0.2rem 0.62rem;
  font-size: 0.84rem;
  font-weight: 700;
}
.scheme-detail-box {
  background: linear-gradient(135deg, #111925 0%, #1b2533 100%);
  border: 1px solid #2b3a52;
  border-radius: 16px;
  padding: 1rem;
  margin-bottom: 0.85rem;
}

.chat-lead {
  font-family: Georgia, 'Times New Roman', serif;
  color: #40dcb4;
  font-size: 4.2rem;
  line-height: 1.05;
  margin: 0.5rem 0 0.6rem 0;
}
.chat-copy { color: #8f96a3; font-size: 1.1rem; margin-bottom: 1.2rem; }

.bubble-wrap { display: flex; margin: 0.8rem 0; }
.bubble-wrap.user { justify-content: flex-end; }
.bubble-wrap.assistant { justify-content: flex-start; }
.bubble {
  max-width: 74%;
  background: #171b23;
  border: 1px solid #202635;
  color: #e7edf5;
  border-radius: 18px;
  padding: 1rem 1.2rem;
  line-height: 1.5;
  white-space: pre-wrap;
}
.bubble-wrap.user .bubble {
  background: linear-gradient(135deg, #03342b 0%, #0a4a39 100%);
  border-color: #d8c129;
}

.tip-card {
  background: #171c25;
  border: 1px solid #26313f;
  border-radius: 16px;
  padding: 0.95rem 1.1rem;
  color: #d7e5df;
}

.scheme-guide {
  background: linear-gradient(135deg, #080d13 0%, #0f141c 100%);
  border: 1px solid #202938;
  border-radius: 24px;
  padding: 1.8rem;
  margin-bottom: 1.2rem;
}
.step-title {
  font-family: Georgia, 'Times New Roman', serif;
  color: var(--yellow);
  font-size: 2rem;
  margin-bottom: 0.5rem;
}
.step-copy { color: var(--muted); font-size: 1.08rem; }

.bottom-cta {
  margin-top: 1.5rem;
  background: linear-gradient(135deg, #02222a 0%, #0a2f2d 100%);
  border: 1px solid #184a56;
  border-radius: 28px;
  padding: 2rem;
  text-align: center;
}
.bottom-cta h3 {
  font-family: Georgia, 'Times New Roman', serif;
  color: #ebf5f0;
  font-size: 3rem;
  margin: 0 0 1rem 0;
}

[data-testid="stButton"] > button {
  border-radius: 14px !important;
  background: #151a22 !important;
  color: #d6e3dc !important;
  border: 1px solid #273142 !important;
  min-height: 44px;
}

.chat-input-note {
  color: #838a95;
  font-size: 0.9rem;
  margin-top: 0.4rem;
}

.settings-bar {
  background: linear-gradient(135deg, #0f141c 0%, #131a23 100%);
  border: 1px solid #253043;
  border-radius: 14px;
  padding: 0.75rem 0.9rem;
  margin-bottom: 1rem;
}
.settings-title {
  color: #dce5ef;
  font-weight: 700;
  letter-spacing: 0.08em;
  margin: 0 0 0.4rem 0;
}
.settings-sub {
  color: #8994a1;
  margin: 0;
  font-size: 0.92rem;
}

@media (max-width: 900px) {
  .hero-title { font-size: 2.9rem; }
  .chat-lead { font-size: 2.7rem; }
  .bubble { max-width: 92%; }
  .ms-search { min-width: 0; width: 100%; }
  .ms-stats { grid-template-columns: 1fr; }
  .ms-categories { grid-template-columns: 1fr 1fr; }
  .ms-steps { grid-template-columns: 1fr; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _parse_schemes_from_text(text: str) -> List[Dict[str, str]]:
    if not text:
        return []
    name_matches = re.findall(r"🌾\s*\*?([^\n*]+)\*?", text)
    link_matches = re.findall(r"Apply(?:\s+Now)?:\s*(https?://\S+)", text)

    schemes: List[Dict[str, str]] = []
    for i, name in enumerate(name_matches[:3]):
        link = link_matches[i] if i < len(link_matches) else ""
        schemes.append(
            {
                "name": name.strip(),
                "desc": "Matched from your profile and current eligibility context.",
                "link": link,
                "badge": "PRIORITY MATCH" if i == 0 else f"OPTION {i+1}",
            }
        )
    return schemes


def _latest_assistant_text() -> str:
    for msg in reversed(st.session_state.chat_history):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def _profile_is_complete() -> bool:
    required = ["name", "occupation", "gender", "income", "loan_amount", "state"]
    for key in required:
        val = str(st.session_state.user_profile.get(key, "")).strip().lower()
        if not val or val in {"unknown", "not specified", "skip", "skipped"}:
            return False
    return True


def _has_scheme_recommendation() -> bool:
    text = _latest_assistant_text().lower()
    markers = ["top 3 schemes", "best match", "recommended for you", "apply now", "🌾"]
    return any(marker in text for marker in markers)


def _render_top_nav() -> None:
    sbtn, left, spacer, n1, n2, n3 = st.columns([1, 2.1, 2.9, 1, 1, 1])
    with sbtn:
        if st.button("⚙ Settings", use_container_width=True):
            st.session_state.show_settings = not st.session_state.show_settings
            st.rerun()
    with left:
        st.markdown('<h2 class="brand-title">Guardian Welfare</h2>', unsafe_allow_html=True)
    with n1:
        if st.button("HOME", use_container_width=True):
            st.session_state.ui_tab = "HOME"
            st.rerun()
    with n2:
        if st.button("SCHEMES", use_container_width=True):
            st.session_state.ui_tab = "SCHEMES"
            st.rerun()
    with n3:
        if st.button("ASSISTANT", use_container_width=True):
            st.session_state.ui_tab = "ASSISTANT"
            st.rerun()


def _render_settings_bar() -> None:
    if not st.session_state.show_settings:
        return
    st.markdown(
        """
<div class="settings-bar">
  <p class="settings-title">SETTINGS PANEL</p>
  <p class="settings-sub">Quick controls for indexing and conversation reset.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    left, _ = st.columns([1.25, 3.75])
    with left:
        st.caption(f"Index status: {st.session_state.index_status.get('status', 'unknown')}")
        states = _supported_states_for_sync()
        states_with_bulk = ["All States (Bulk)"] + states
        selected_state = st.selectbox(
            "State for myScheme Sync",
            options=states_with_bulk,
            index=states_with_bulk.index(st.session_state.sync_state_name) if st.session_state.sync_state_name in states_with_bulk else 0,
        )
        st.session_state.sync_state_name = selected_state
        if st.button("Sync State Schemes", use_container_width=True):
            with st.spinner(f"Syncing {selected_state} schemes from myScheme API..."):
                if selected_state == "All States (Bulk)":
                    sync_result = _sync_all_states_from_myscheme(states)
                else:
                    sync_result = _sync_state_schemes_from_myscheme(selected_state)
                st.session_state.rag_agent._source_docs = None
                st.session_state.index_status = st.session_state.rag_agent.ensure_index(force_reindex=True)
            if sync_result.get("ok"):
                if selected_state == "All States (Bulk)":
                    st.success(
                        f"Bulk sync complete: {sync_result.get('saved_schemes', 0)} schemes from {sync_result.get('synced_states', 0)} states."
                    )
                else:
                    st.success(
                        f"Synced {sync_result.get('saved_schemes', 0)} schemes for {selected_state} and rebuilt index."
                    )
            else:
                st.warning(f"State sync failed: {sync_result.get('detail', 'Unknown error')}")
            st.rerun()
        if st.button("Rebuild Index", use_container_width=True):
            with st.spinner("Indexing local scheme documents..."):
                st.session_state.rag_agent._source_docs = None
                st.session_state.index_status = st.session_state.rag_agent.ensure_index(force_reindex=True)
            st.rerun()
        if st.button("Reset Chat", use_container_width=True):
            for key in ["orchestrator", "rag_agent", "web_agent"]:
                if key in st.session_state:
                    del st.session_state[key]
            reset_state()
            st.session_state.pending_prompt = ""
            st.session_state.ui_tab = "ASSISTANT"
            st.rerun()


def _render_home_view() -> None:
    st.markdown(
        """
<div class="ms-home">
  <div class="ms-strip">
    <h2 class="ms-brand">Jan-Sahayak Welfare</h2>
    <div class="ms-search">Enter scheme name to search...</div>
  </div>
  <div class="ms-hero">
    <p class="ms-kicker">#GOVERNMENTSCHEMES / #SCHEMESFORYOU</p>
    <h1 class="ms-title">Find Relevant<br/>Government Schemes</h1>
    <p class="ms-copy">Profile-based discovery, Tamil voice support, and guided application steps.</p>
    <a class="ms-cta" href="#">Find Schemes For You</a>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="ms-stats">
  <div class="ms-stat-card"><div class="ms-stat-n">4600+</div><div class="ms-stat-l">Indexed Scheme Signals</div></div>
  <div class="ms-stat-card"><div class="ms-stat-n">650+</div><div class="ms-stat-l">Central Scheme Signals</div></div>
  <div class="ms-stat-card"><div class="ms-stat-n">4000+</div><div class="ms-stat-l">State/UT Scheme Signals</div></div>
</div>
<div class="ms-tabs">
  <span class="ms-tab">Categories</span>
  <span class="ms-tab">States/UTs</span>
  <span class="ms-tab">Ministries</span>
</div>
<h3 class="ms-heading">Find schemes based on categories</h3>
        """,
        unsafe_allow_html=True,
    )

    categories = [
        ("🌾", "Agriculture, Rural & Environment", "840+"),
        ("🏦", "Banking & Financial Services", "329+"),
        ("🤝", "Business & Entrepreneurship", "747+"),
        ("🎓", "Education & Learning", "1086+"),
        ("❤️", "Health & Wellness", "287+"),
        ("🏠", "Housing & Shelter", "134+"),
        ("⚖️", "Public Safety, Law & Justice", "33+"),
        ("🧪", "Science, IT & Communication", "109+"),
        ("📈", "Skills & Employment", "399+"),
        ("✊", "Social Welfare & Empowerment", "1438+"),
    ]
    cat_html = ['<div class="ms-categories">']
    for icon, title, count in categories:
        cat_html.append(
            f'<div class="ms-cat"><div class="ms-cat-ico">{icon}</div><div class="ms-cat-n">{count} Schemes</div><div class="ms-cat-t">{title}</div></div>'
        )
    cat_html.append("</div>")
    st.markdown("".join(cat_html), unsafe_allow_html=True)

    st.markdown(
        """
<div class="ms-how">
  <p class="ms-kicker" style="color:#6d7f70;">How it works</p>
  <h3 class="ms-heading" style="font-size:2rem;">Easy steps to apply for government schemes</h3>
  <div class="ms-steps">
    <div class="ms-step"><div class="ms-cat-ico">📝</div><h4 style="color:#6fe08c;">Enter Details</h4><p style="color:#d3deec;">Share profile basics and preferences.</p></div>
    <div class="ms-step"><div class="ms-cat-ico">🔎</div><h4 style="color:#6fe08c;">Search</h4><p style="color:#d3deec;">Assistant finds best-fit schemes.</p></div>
    <div class="ms-step"><div class="ms-cat-ico">✅</div><h4 style="color:#6fe08c;">Select & Apply</h4><p style="color:#d3deec;">Get eligibility, docs, and apply links.</p></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_schemes_view() -> None:
    st.markdown('<h2 class="chat-lead" style="font-size:3rem;">All Schemes</h2>', unsafe_allow_html=True)
    st.caption("State-wise scheme listing from myScheme sync. Click any scheme for details.")

    states_available = _available_synced_states()
    if not states_available:
        st.warning("No synced scheme files found. Use Settings -> Sync State Schemes first.")
        return

    selected_state = st.selectbox("Choose state", options=states_available, index=0)
    schemes = _load_synced_schemes_for_state(selected_state)
    st.markdown(f"### We found {len(schemes)} schemes for {selected_state}")

    selected_slug = st.session_state.selected_scheme_slug
    if selected_slug:
        chosen = next((s for s in schemes if s.get("slug") == selected_slug), None)
        if chosen:
            st.markdown(
                f"""
<div class="scheme-detail-box">
  <h3 class="scheme-head" style="font-size:1.7rem;">{chosen.get('name', 'Scheme')}</h3>
  <div class="scheme-meta">State: {chosen.get('state', selected_state)} | Category: {chosen.get('category', 'Not specified')}</div>
  <div class="scheme-meta">For: {chosen.get('scheme_for', 'Individual')} | Close Date: {chosen.get('close_date', 'Not specified')}</div>
  <div class="scheme-desc">{chosen.get('brief', 'No description available.')}</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            link = chosen.get("official_link", "")
            if link and link.startswith("http"):
                st.markdown(f"[Open Official Scheme Page]({link})")
            if st.button("Ask Assistant About This Scheme", key=f"ask_{selected_slug}", use_container_width=False):
                st.session_state.pending_prompt = f"Explain eligibility and documents for {chosen.get('name', 'this scheme')}"
                st.session_state.ui_tab = "ASSISTANT"
                st.rerun()
            st.markdown("---")

    for idx, scheme in enumerate(schemes):
        with st.container():
            tags = [t.strip() for t in scheme.get("tags", "").split(",") if t.strip()]
            tag_html = "".join([f'<span class="tag-pill">{t}</span>' for t in tags[:8]])
            st.markdown(
                f"""
<div class="scheme-shell">
  <h3 class="scheme-head">{scheme.get('name', 'Scheme')}</h3>
  <div class="scheme-meta">{scheme.get('state', selected_state)}</div>
  <div class="scheme-desc">{scheme.get("brief", "No description available.")}</div>
  <div class="tag-row">{tag_html}</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("View Details", key=f"view_{selected_state}_{idx}", use_container_width=True):
                    st.session_state.selected_scheme_slug = scheme.get("slug", "")
                    st.rerun()
            with c2:
                link = scheme.get("official_link", "")
                if link and link.startswith("http"):
                    st.markdown(f"[Open Link]({link})")
                else:
                    st.caption("Official link unavailable")
            st.markdown("---")


def _render_chat_message(role: str, text: str) -> None:
    safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f"""
<div class="bubble-wrap {role}">
  <div class="bubble">{safe_text}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _fetch_url_text(url: str, headers: Dict[str, str] | None = None) -> str:
    req = Request(url, headers=headers or {"User-Agent": "Mozilla/5.0 Jan-Sahayak Sync Bot"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _supported_states_for_sync() -> List[str]:
    return [
        "Andaman and Nicobar Islands",
        "Andhra Pradesh",
        "Arunachal Pradesh",
        "Assam",
        "Bihar",
        "Chandigarh",
        "Chhattisgarh",
        "Dadra & Nagar Haveli and Daman & Diu",
        "Delhi",
        "Goa",
        "Gujarat",
        "Haryana",
        "Himachal Pradesh",
        "Jammu and Kashmir",
        "Jharkhand",
        "Karnataka",
        "Kerala",
        "Ladakh",
        "Lakshadweep",
        "Madhya Pradesh",
        "Maharashtra",
        "Manipur",
        "Meghalaya",
        "Mizoram",
        "Nagaland",
        "Odisha",
        "Puducherry",
        "Punjab",
        "Rajasthan",
        "Sikkim",
        "Tamil Nadu",
        "Telangana",
        "Tripura",
        "Uttar Pradesh",
        "Uttarakhand",
        "West Bengal",
    ]


def _sync_state_schemes_from_myscheme(state_name: str) -> Dict[str, Any]:
    filters = json.dumps(
        [
            {"identifier": "beneficiaryState", "value": state_name},
            {"identifier": "level", "value": "State"},
        ]
    )
    q = quote(filters, safe="")
    from_idx = 0
    size = 100
    all_items: List[Dict[str, Any]] = []
    api_key = "tYTy5eEhlu9rFjyxuCr7ra7ACp4dv1RH8gWuHTDc"
    headers = {
        "x-api-key": api_key,
        "Origin": "https://www.myscheme.gov.in",
        "Referer": "https://www.myscheme.gov.in/",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 Jan-Sahayak State Sync Bot",
    }

    while True:
        url = (
            "https://api.myscheme.gov.in/search/v6/schemes"
            f"?lang=en&q={q}&keyword=&sort=&from={from_idx}&size={size}"
        )
        try:
            raw = _fetch_url_text(url, headers=headers)
            payload = json.loads(raw)
        except Exception as exc:
            return {"ok": False, "detail": f"API fetch error: {exc}"}

        hits = payload.get("data", {}).get("hits", {})
        items = hits.get("items", []) or []
        if not items:
            break
        all_items.extend(items)

        page = hits.get("page", {}) or {}
        total = int(page.get("total", len(all_items)) or len(all_items))
        from_idx += size
        if from_idx >= total:
            break

    if not all_items:
        return {"ok": False, "detail": f"No schemes returned for {state_name}."}

    safe_state = re.sub(r"[^a-z0-9]+", "_", state_name.strip().lower()).strip("_")
    sections: List[str] = [f"# Schemes - {state_name} (myScheme API)\n"]
    saved = 0
    for item in all_items:
        fields = item.get("fields", {}) if isinstance(item, dict) else {}
        scheme_name = str(fields.get("schemeName", "")).strip()
        if not scheme_name:
            continue
        slug = str(fields.get("slug", "")).strip()
        level = str(fields.get("level", "State")).strip()
        categories = fields.get("schemeCategory", [])
        if not isinstance(categories, list):
            categories = []
        tags = fields.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        brief = str(fields.get("briefDescription", "")).strip()
        scheme_for = str(fields.get("schemeFor", "Individual")).strip()
        close_date = str(fields.get("schemeCloseDate", "") or "Not specified")
        states = fields.get("beneficiaryState", [])
        if not isinstance(states, list):
            states = [str(states)]
        scheme_url = f"https://www.myscheme.gov.in/schemes/{slug}" if slug else "Not available"

        sections.append(f"## {scheme_name}\n")
        sections.append(f"Scheme Type: {level.lower()}_support")
        sections.append(f"State Origin: {state_name}")
        sections.append(f"Category: {', '.join(categories) if categories else 'Not specified'}")
        sections.append(
            f"Good Match Signals: {', '.join(tags) if tags else 'general_support'}"
        )
        sections.append(
            f"Eligibility: Beneficiary state should include {state_name}. Scheme is for {scheme_for}. "
            f"Check final criteria and age/category limits on the official page."
        )
        sections.append(f"Benefits: {brief or 'Benefit details are available on the official page.'}")
        sections.append("Application Steps: Open the official link, verify eligibility, and submit required documents.")
        sections.append(f"Official Link: {scheme_url}")
        sections.append(f"\n### Brief\n{brief or 'No brief description provided.'}\n")
        saved += 1

    docs_dir = Path(__file__).resolve().parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    target = docs_dir / f"schemes_{safe_state}.md"
    target.write_text("\n".join(sections), encoding="utf-8")
    return {"ok": True, "saved_schemes": saved, "file": str(target)}


def _sync_all_states_from_myscheme(states: List[str]) -> Dict[str, Any]:
    total_saved = 0
    synced_states = 0
    failures: List[str] = []
    for state in states:
        result = _sync_state_schemes_from_myscheme(state)
        if result.get("ok"):
            total_saved += int(result.get("saved_schemes", 0))
            synced_states += 1
        else:
            failures.append(f"{state}: {result.get('detail', 'error')}")
    if synced_states == 0:
        return {"ok": False, "detail": "No states synced successfully."}
    detail = " | ".join(failures[:5]) if failures else "All states synced."
    return {
        "ok": True,
        "saved_schemes": total_saved,
        "synced_states": synced_states,
        "detail": detail,
    }


def _available_synced_states() -> List[str]:
    docs_dir = Path(__file__).resolve().parent / "docs"
    if not docs_dir.exists():
        return []
    states: List[str] = []
    for path in docs_dir.glob("schemes_*.md"):
        state = path.stem.replace("schemes_", "").replace("_", " ").title()
        states.append(state)
    return sorted(set(states))


def _load_synced_schemes_for_state(state_name: str) -> List[Dict[str, str]]:
    safe_state = re.sub(r"[^a-z0-9]+", "_", state_name.strip().lower()).strip("_")
    path = Path(__file__).resolve().parent / "docs" / f"schemes_{safe_state}.md"
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"^##\s+", content, flags=re.MULTILINE)
    schemes: List[Dict[str, str]] = []
    for block in blocks[1:]:
        lines = block.splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        data: Dict[str, str] = {
            "name": name,
            "state": state_name,
            "category": "",
            "scheme_for": "",
            "close_date": "",
            "tags": "",
            "official_link": "",
            "brief": "",
            "slug": "",
        }
        for i, line in enumerate(lines[1:], start=1):
            raw = line.strip()
            if raw.startswith("Category:"):
                data["category"] = raw.replace("Category:", "", 1).strip()
            elif raw.startswith("Scheme For:"):
                data["scheme_for"] = raw.replace("Scheme For:", "", 1).strip()
            elif raw.startswith("Close Date:"):
                data["close_date"] = raw.replace("Close Date:", "", 1).strip()
            elif raw.startswith("Good Match Signals:"):
                data["tags"] = raw.replace("Good Match Signals:", "", 1).strip()
            elif raw.startswith("Official Link:"):
                link = raw.replace("Official Link:", "", 1).strip()
                data["official_link"] = link
                if "/schemes/" in link:
                    data["slug"] = link.rsplit("/", 1)[-1].strip()
            elif raw.startswith("### Brief"):
                brief = "\n".join(lines[i + 1 :]).strip()
                data["brief"] = re.sub(r"\s+", " ", brief)
                break
        if not data["slug"]:
            data["slug"] = re.sub(r"[^a-z0-9]+", "-", data["name"].lower()).strip("-")
        schemes.append(data)
    return schemes


def _contains_tamil_script(text: str) -> bool:
    return bool(re.search(r"[\u0B80-\u0BFF]", text or ""))


def _send_query(prompt: str) -> None:
    normalized_prompt = (prompt or "").strip()
    if _contains_tamil_script(normalized_prompt):
        normalized_prompt = _translate_tamil_to_english(normalized_prompt)

    append_message("user", normalized_prompt)
    result = st.session_state.orchestrator.handle_query(
        user_query=normalized_prompt,
        user_profile=st.session_state.user_profile,
        chat_history=st.session_state.chat_history,
    )
    st.session_state.user_profile = result.get("updated_profile", st.session_state.user_profile)
    response_markdown = result.get("response_markdown", "I could not process that request.")
    assistant_reply = str(response_markdown)
    append_message("assistant", assistant_reply)


@st.cache_resource(show_spinner=False)
def _get_tamil_asr_pipeline():
    try:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        return ("omnilingual", ASRInferencePipeline(model_card="omniASR_CTC_300M_v2"))
    except Exception:
        from faster_whisper import WhisperModel

        return ("faster_whisper", WhisperModel("small", device="cpu", compute_type="int8"))


def _transcribe_tamil_audio(audio_file: Any) -> str:
    if audio_file is None:
        return ""
    suffix = Path(getattr(audio_file, "name", "audio.wav")).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_file.getvalue())
        tmp_path = Path(tmp.name)
    try:
        backend, asr_model = _get_tamil_asr_pipeline()
        if backend == "omnilingual":
            transcription = asr_model.transcribe([str(tmp_path)], lang=["tam_Taml"], batch_size=1)
            if transcription and isinstance(transcription, list):
                return str(transcription[0]).strip()
            return ""

        segments, _ = asr_model.transcribe(str(tmp_path), language="ta", task="transcribe")
        text = " ".join(seg.text.strip() for seg in segments if seg.text).strip()
        return text
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _translate_tamil_to_english(text: str) -> str:
    clean_text = (text or "").strip()
    if not clean_text:
        return ""

    # Fast-path common greeting so UI behavior is predictable.
    greeting_map = {
        "வணக்கம்": "Hi",
        "வணககம்": "Hi",
    }
    if clean_text in greeting_map:
        return greeting_map[clean_text]

    # First fallback: direct translator (works even without Gemini model object).
    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source="ta", target="en").translate(clean_text)
        translated = (translated or "").strip()
        if translated:
            return translated
    except Exception:
        pass

    model = getattr(st.session_state.orchestrator, "model", None)
    if model is None:
        return clean_text
    prompt = (
        "Translate the following Tamil text into natural English for a welfare-assistant chatbot. "
        "Return only the translated English text, no extra notes.\n\n"
        f"Tamil text: {clean_text}"
    )
    try:
        result = model.generate_content(prompt)
        translated = str(getattr(result, "text", "")).strip()
        return translated or clean_text
    except Exception:
        return clean_text


def _render_assistant_view() -> None:
    st.markdown('<h2 class="chat-lead">I am here to help</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="chat-copy">How can I assist you today? We can discuss welfare schemes, update your profile, and guide your application step-by-step.</p>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.chat_history:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        _render_chat_message(role, str(msg.get("content", "")))

    if _profile_is_complete() and _has_scheme_recommendation():
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Check Eligibility Requirements", use_container_width=True):
                _send_query("check eligibility requirements")
                st.rerun()
        with b2:
            if st.button("View Other Available Schemes", use_container_width=True):
                _send_query("show more schemes")
                st.rerun()

    st.markdown("**Tamil Voice Input**")
    audio_file = st.audio_input("Speak in Tamil and click transcribe.")
    if st.button("Transcribe Tamil Voice", use_container_width=True):
        if audio_file is None:
            st.warning("Please record Tamil audio first.")
        else:
            tamil_prompt = ""
            with st.spinner("Transcribing Tamil voice..."):
                try:
                    tamil_prompt = _transcribe_tamil_audio(audio_file)
                except Exception as exc:
                    st.error(f"Voice transcription failed: {exc}")
            if tamil_prompt:
                english_prompt = _translate_tamil_to_english(tamil_prompt)
                st.session_state.pending_prompt = english_prompt
                st.caption(f"Tamil transcript: {tamil_prompt}")
                st.caption(f"English sent to chatbot: {english_prompt}")
                st.success("Tamil voice converted and translated to English.")
                st.rerun()
            else:
                st.warning("No speech was detected from the audio.")

    prompt = st.chat_input("Type your message...")
    if not prompt and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""

    if prompt:
        with st.spinner("Analyzing profile, checking schemes, and gathering evidence..."):
            _send_query(prompt)
        st.rerun()

    st.markdown('<p class="chat-input-note">Tip: You can ask for documents, application steps, or more schemes.</p>', unsafe_allow_html=True)


def main() -> None:
    init_state()
    _bootstrap_agents()
    _init_ui_state()
    _inject_styles()

    _render_top_nav()
    _render_settings_bar()

    if st.session_state.ui_tab == "HOME":
        _render_home_view()
    elif st.session_state.ui_tab == "SCHEMES":
        _render_schemes_view()
    else:
        _render_assistant_view()


if __name__ == "__main__":
    main()
