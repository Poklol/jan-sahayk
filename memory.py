from __future__ import annotations

from typing import Dict, List

import streamlit as st

DEFAULT_PROFILE: Dict[str, str] = {
    "name": "",
    "occupation": "",
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
    required = ["name", "occupation", "income", "loan_amount", "state"]
    missing = []
    for field in required:
        val = profile.get(field, "").strip()
        if not val or val.lower() in ("unknown", "skipped", "not specified"):
            if not val:
                missing.append(field)
    return missing


def next_followup_question(missing_fields: List[str]) -> str:
    if not missing_fields:
        return ""
        
    question_map = {
        "name": "What is your name?",
        "occupation": "What is your profession or occupation (for example: farmer, student, small business owner)?",
        "income": "What is your yearly family income range? (If you're not sure, it's okay to say 'I don't know').",
        "loan_amount": "How much loan or financial assistance are you looking for?",
        "state": "Which state do you currently live in?",
    }
    
    if len(missing_fields) > 1:
        return f"{question_map.get(missing_fields[0], 'Could you provide some more details?')} This will help me find the best loan or welfare schemes for you."
        
    return question_map.get(missing_fields[0], "Could you provide one more detail so I can continue?")


def profile_snapshot(profile: Dict[str, str]) -> str:
    lines = [
        f"- Name: {profile.get('name') or 'Not provided'}",
        f"- Profession: {profile.get('occupation') or 'Not provided'}",
        f"- Income: {profile.get('income') or 'Not provided'}",
        f"- Loan Needed: {profile.get('loan_amount') or 'Not provided'}",
        f"- State: {profile.get('state') or 'Not provided'}",
    ]
    return "\n".join(lines)


def reset_state() -> None:
    st.session_state.chat_history = []
    st.session_state.user_profile = DEFAULT_PROFILE.copy()
