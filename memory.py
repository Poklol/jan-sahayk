from __future__ import annotations

import random
from typing import Dict, List

import streamlit as st

DEFAULT_PROFILE: Dict[str, str] = {
    "name": "",
    "occupation": "",
    "gender": "",
    "income": "",
    "loan_amount": "",
    "state": "",
}


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = DEFAULT_PROFILE.copy()


def append_message(role: str, content: str) -> None:
    st.session_state.chat_history.append({"role": role, "content": content})


def update_profile(profile_updates: Dict[str, str]) -> Dict[str, str]:
    profile = st.session_state.user_profile.copy()
    for key, value in profile_updates.items():
        if key in profile and isinstance(value, str) and value.strip():
            profile[key] = value.strip()
    st.session_state.user_profile = profile
    return profile


def get_missing_required_fields(profile: Dict[str, str]) -> List[str]:
    required = ["name", "occupation", "gender", "income", "loan_amount", "state"]
    missing = []
    for field in required:
        val = profile.get(field, "").strip()
        if not val or val.lower() in ("unknown", "skipped", "not specified"):
            if not val:
                missing.append(field)
    return missing


def next_followup_question(missing_fields: List[str], profile: Dict[str, str] | None = None) -> str:
    if not missing_fields:
        return ""

    field_map = {
        "occupation": "profession",
        "loan_amount": "loan",
    }
    field = field_map.get(missing_fields[0], missing_fields[0])
    if field == "income":
        occupation = ((profile or {}).get("occupation", "") or "").strip().lower()
        if occupation == "student":
            return "To match student schemes correctly - what's your family yearly income?"
    return ask_missing_field(field)


def ask_missing_field(field: str) -> str:
    prompts = {
        "name": [
            "Hey! What should I call you? 🙂",
            "Before we continue - your name?",
        ],
        "profession": [
            "Got it 👍 What kind of work do you do?",
            "Nice - what's your profession?",
        ],
        "gender": [
            "Please share your gender (male/female) so I can avoid ineligible schemes.",
            "Quick check: your gender (male/female)?",
        ],
        "income": [
            "To match the right schemes - roughly your yearly income?",
            "Just to narrow it down - your income range?",
        ],
        "loan": [
            "How much support are you roughly looking for?",
            "What kind of financial help do you need?",
        ],
        "state": [
            "Which state are you in?",
            "Where are you currently based?",
        ],
    }
    fallback = "Could you share one quick detail so I can help better?"
    return random.choice(prompts.get(field, [fallback]))


def profile_snapshot(profile: Dict[str, str]) -> str:
    lines = [
        f"- Name: {profile.get('name') or 'Not provided'}",
        f"- Profession: {profile.get('occupation') or 'Not provided'}",
        f"- Gender: {profile.get('gender') or 'Not provided'}",
        f"- Income: {profile.get('income') or 'Not provided'}",
        f"- Loan Needed: {profile.get('loan_amount') or 'Not provided'}",
        f"- State: {profile.get('state') or 'Not provided'}",
    ]
    return "\n".join(lines)


def reset_state() -> None:
    st.session_state.chat_history = []
    st.session_state.user_profile = DEFAULT_PROFILE.copy()
